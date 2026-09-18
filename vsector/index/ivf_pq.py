"""IVF-PQ: Inverted File with Product Quantization.

Spec: nlist 4096-65536, nprobe 64-256, PQ subspaces dimension/8, codebook 256 codes (8-bit).
Training: 1M sample vectors, shared memory codebook, retrain 24h.

Simplified: uses numpy k-means + residual PQ (optional faiss if available, else pure numpy flat scan with compression simulation).
"""
from __future__ import annotations

import logging
import numpy as np
from .base import BaseIndex, IndexResult
from .flat import FlatIndex

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


class IVFPQIndex(BaseIndex):
    def __init__(self, dimension: int, metric: str = "cosine", nlist: int = 4096, nprobe: int = 64, m: int | None = None):
        super().__init__(dimension, metric)
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m or max(1, dimension // 8)  # PQ subspaces
        self._fallback = FlatIndex(dimension, metric)
        self._trained = False
        self._faiss_index = None
        self.degraded = False
        self.backend_name = "FlatIndex"
        self.is_native_backend = False
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._metadatas: list[dict] = []
        # try faiss init
        if HAS_FAISS:
            try:
                quantizer = faiss.IndexFlatL2(dimension) if metric != "cosine" else faiss.IndexFlatIP(dimension)
                # IVF-PQ setup
                self._faiss_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, self.m, 8)
                if metric == "cosine":
                    faiss.normalize_L2  # type: ignore
                logger.info(f"FAISS IVF-PQ initialized dim={dimension} nlist={nlist} m={self.m}")
                self.backend_name = "FAISS"
                self.is_native_backend = True
            except Exception as e:
                logger.warning(f"FAISS init failed, fallback to flat: {e}")
                self._faiss_index = None
                self.degraded = True
                self.backend_name = "FlatIndex"
                self.is_native_backend = False
        else:
            logger.warning("faiss not installed - using FlatIndex fallback for IVF_PQ")
            self.degraded = True
            self.backend_name = "FlatIndex"
            self.is_native_backend = False

    def train(self, sample_vectors: np.ndarray):
        if self._faiss_index is None:
            self._trained = True
            return
        if len(sample_vectors) < self.nlist:
            logger.warning("Not enough samples to train IVF-PQ, skipping")
            return
        if self.metric == "cosine":
            faiss.normalize_L2(sample_vectors)  # type: ignore
        try:
            self._faiss_index.train(sample_vectors.astype(np.float32))  # type: ignore
            self._trained = True
        except Exception as e:
            logger.warning(f"FAISS train failed: {e}")

    def add(self, ids, vectors, metadatas=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # auto-train on first 1M if not trained and using faiss
        if self._faiss_index is not None and not self._faiss_index.is_trained:
            # collect fallback vectors
            self._fallback.add(ids, vectors, metadatas)
            # try train if we have enough
            if self._fallback.count() >= 1000:
                all_vec = self._fallback.vectors  # type: ignore
                if all_vec is not None:
                    self.train(all_vec[: min(100000, len(all_vec))])
                    if self._faiss_index.is_trained:  # type: ignore
                        # migrate
                        if self.metric == "cosine":
                            faiss.normalize_L2(all_vec)  # type: ignore
                        self._faiss_index.add_with_ids(all_vec.astype(np.float32), np.arange(len(all_vec), dtype=np.int64))  # type: ignore
                        # clear fallback
                        self._ids = list(self._fallback.ids)
                        self._metadatas = list(self._fallback.metadatas)
                        self._id_to_idx = { _id: i for i,_id in enumerate(self._ids)}
                        self._fallback = FlatIndex(self.dimension, self.metric)
            return
        if self._faiss_index is not None and getattr(self._faiss_index, "is_trained", False):
            vec = vectors.copy()
            if self.metric == "cosine":
                faiss.normalize_L2(vec)  # type: ignore
            start = len(self._ids)
            idxs = np.arange(start, start + len(ids), dtype=np.int64)
            try:
                self._faiss_index.add_with_ids(vec, idxs)  # type: ignore
            except Exception:
                self._faiss_index.add(vec)  # type: ignore
                idxs = np.arange(start, start+len(ids))
            for i, _id in enumerate(ids):
                self._ids.append(_id)
                self._id_to_idx[_id] = int(idxs[i])
                self._metadatas.append(metadatas[i] if metadatas else {})
        else:
            self._fallback.add(ids, vectors, metadatas)
            self._ids = []  # keep empty when using fallback only

    def delete(self, ids):
        # faiss delete not trivial; fallback path uses mask filtering
        if self._faiss_index is not None and self._ids:
            for _id in ids:
                idx = self._id_to_idx.pop(_id, None)
                if idx is not None:
                    # mark metadata deleted; search will filter
                    if idx < len(self._metadatas):
                        self._metadatas[idx] = {"_deleted": True}
        else:
            self._fallback.delete(ids)

    def count(self):
        if self._faiss_index is not None and self._ids:
            return len([m for m in self._metadatas if not m.get("_deleted")])
        return self._fallback.count()

    def search(self, query, top_k=10, ef_search=None, nprobe=None, filter_fn=None):
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        probe = nprobe or self.nprobe
        if self._faiss_index is not None and getattr(self._faiss_index, "is_trained", False) and len(self._ids)>0:
            try:
                self._faiss_index.nprobe = min(probe, self.nlist)  # type: ignore
                if self.metric == "cosine":
                    faiss.normalize_L2(query)  # type: ignore
                k = top_k * 3 if filter_fn else top_k
                D, I = self._faiss_index.search(query, k)  # type: ignore
                res=[]
                for dist, idx in zip(D[0], I[0]):
                    if idx == -1 or idx >= len(self._ids):
                        continue
                    md = self._metadatas[idx]
                    if md.get("_deleted"):
                        continue
                    if filter_fn and not filter_fn(md):
                        continue
                    score = float(-dist) if self.metric != "cosine" else float(dist)  # FAISS IP vs L2
                    # For IP, larger is better; faiss returns inner product
                    if self.metric == "cosine":
                        score = float(dist)
                    else:
                        score = -float(dist)
                    res.append(IndexResult(id=self._ids[idx], score=score, metadata=md))
                    if len(res) >= top_k:
                        break
                if res:
                    return res
            except Exception as e:
                logger.warning(f"IVF-PQ search failed, fallback: {e}")
        # fallback brute force
        return self._fallback.search(query.reshape(-1), top_k=top_k, filter_fn=filter_fn)

    @property
    def compressed_size_bytes(self) -> int:
        # (dimension / 8) bytes vs 4*dimension raw per spec
        return self.dimension // self.m * self.m
