"""HNSW index - tries hnswlib if available, else FLAT fallback with HNSW params bookkeeping."""
from __future__ import annotations

import math
import logging
import numpy as np
from .base import BaseIndex, IndexResult
from .flat import FlatIndex

logger = logging.getLogger(__name__)

try:
    import hnswlib  # type: ignore
    HAS_HNSWLIB = True
except Exception:
    HAS_HNSWLIB = False


class HNSWIndex(BaseIndex):
    """HNSW (Hierarchical Navigable Small World)

    Params per spec:
      M 16-64, ef_construction 200-500, ef_search 50-200, max_level auto log(n)/log(M)
    Memory layout: adjacency flat int32, mmap vectors, Level-0 NVMe, upper in RAM (simulated).
    """

    def __init__(self, dimension: int, metric: str = "cosine", M: int = 16, ef_construction: int = 200, ef_search: int = 100, max_elements: int = 1000000):
        super().__init__(dimension, metric)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self._count = 0
        # map internal label <-> external id
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._metadatas: dict[int, dict] = {}
        self._next_label = 0
        # translate metric for hnswlib
        self._space = {"cosine": "cosine", "euclidean": "l2", "dot_product": "ip", "manhattan": "l2"}.get(metric.lower(), "l2")
        if HAS_HNSWLIB:
            self._index = hnswlib.Index(space=self._space, dim=dimension)
            self._index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            self._index.set_ef(ef_search)
            self._fallback: FlatIndex | None = None
            self.degraded = False
            self.backend_name = "HNSW"
            self.is_native_backend = True
        else:
            logger.warning("hnswlib not installed - using FlatIndex fallback for HNSW")
            self._index = None  # type: ignore
            self._fallback = FlatIndex(dimension, metric)
            self.degraded = True
            self.backend_name = "FlatIndex"
            self.is_native_backend = False

    @property
    def max_level(self) -> int:
        if self._count == 0:
            return 0
        return int(math.log(self._count + 1) / math.log(self.M + 1e-9))

    def _ensure_capacity(self, need: int):
        if self._index is not None and self._count + need > self._index.get_max_elements():
            new_max = max(self._index.get_max_elements() * 2, self._count + need + 1000)
            self._index.resize_index(new_max)

    def add(self, ids, vectors, metadatas=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = len(ids)
        if HAS_HNSWLIB and self._index is not None:
            self._ensure_capacity(n)
            labels = list(range(self._next_label, self._next_label + n))
            # hnswlib cosine expects normalized? we let it handle
            self._index.add_items(vectors, labels)
            for i, ext_id in enumerate(ids):
                label = labels[i]
                self._label_to_id[label] = ext_id
                self._id_to_label[ext_id] = label
                self._metadatas[label] = metadatas[i] if metadatas else {}
            self._next_label += n
            self._count += n
        else:
            assert self._fallback is not None
            self._fallback.add(ids, vectors, metadatas)
            self._count = self._fallback.count()

    def delete(self, ids):
        if HAS_HNSWLIB and self._index is not None:
            for ext_id in ids:
                label = self._id_to_label.pop(ext_id, None)
                if label is not None:
                    try:
                        self._index.mark_deleted(label)
                    except Exception:
                        pass
                    self._label_to_id.pop(label, None)
                    self._metadatas.pop(label, None)
                    self._count = max(0, self._count - 1)
        else:
            assert self._fallback is not None
            self._fallback.delete(ids)
            self._count = self._fallback.count()

    def count(self): return self._count

    def set_ef(self, ef: int):
        self.ef_search = ef
        if self._index is not None:
            self._index.set_ef(ef)

    def search(self, query, top_k=10, ef_search=None, nprobe=None, filter_fn=None):
        query = np.asarray(query, dtype=np.float32)
        if ef_search is not None:
            self.set_ef(ef_search)
        if HAS_HNSWLIB and self._index is not None:
            if self._count == 0:
                return []
            k = min(top_k * 3 if filter_fn else top_k, self._count)  # over-fetch if filtering
            try:
                labels, distances = self._index.knn_query(query.reshape(1, -1), k=k)
            except Exception as e:
                logger.error(f"HNSW search failed: {e}")
                return []
            labels, distances = labels[0], distances[0]
            res: list[IndexResult] = []
            for lab, dist in zip(labels, distances):
                ext_id = self._label_to_id.get(int(lab))
                if ext_id is None:
                    continue
                md = self._metadatas.get(int(lab), {})
                if filter_fn and not filter_fn(md):
                    continue
                # convert distance to score
                if self._space == "cosine":
                    score = 1 - float(dist)  # hnswlib cosine distance = 1 - cosine
                elif self._space == "l2":
                    score = -float(dist)
                elif self._space == "ip":
                    score = -float(dist)  # ip distance
                else:
                    score = -float(dist)
                res.append(IndexResult(id=ext_id, score=score, metadata=md))
                if len(res) >= top_k:
                    break
            # need vector payload? fetch via stored? we don't store vectors in hnswlib separately; return without vector
            return res
        else:
            assert self._fallback is not None
            return self._fallback.search(query, top_k=top_k, filter_fn=filter_fn)

    def compact(self):
        """Background thread periodically compacts and re-links (simulated)."""
        logger.info(f"Compacting HNSW index count={self._count} max_level={self.max_level}")
