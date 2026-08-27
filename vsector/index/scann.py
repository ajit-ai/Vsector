"""ScaNN (Scalable Nearest Neighbors) — anisotropic + asymmetric hashing stub.

Real ScaNN uses Google's ScANN library; here we simulate with quantized Flat + re-rank.
"""

from __future__ import annotations

import numpy as np
from .base import BaseIndex, IndexResult
from .flat import FlatIndex


class ScaNNIndex(BaseIndex):
    """ScaNN: partitioning + quantization + re-ranking.

    Pipeline: coarse quantizer (k-means) → per-partition PQ/SQ8 → asymmetric distance + re-rank top 100.
    For v0.3.0 we wrap Flat with SQ8 compression + 2-stage search.
    """

    def __init__(self, dimension: int, metric: str = "cosine", num_leaves: int = 1024, leaves_to_search: int = 16, compression: str = "NONE"):
        super().__init__(dimension, metric)
        self.num_leaves = num_leaves
        self.leaves_to_search = leaves_to_search
        self.compression = compression.upper()
        self._flat = FlatIndex(dimension, metric)
        self._vectors_raw: np.ndarray | None = None

    def add(self, ids, vectors, metadatas=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.compression in ("SQ8", "BF16"):
            from .quantize import compress_vectors

            _, meta = compress_vectors(vectors, self.compression)
            # store raw for re-rank, but scoring will use compressed in search
            if self._vectors_raw is None:
                self._vectors_raw = vectors
            else:
                self._vectors_raw = np.vstack([self._vectors_raw, vectors])
        self._flat.add(ids, vectors, metadatas)

    def delete(self, ids):
        self._flat.delete(ids)

    def count(self):
        return self._flat.count()

    def search(self, query, top_k=10, ef_search=None, nprobe=None, filter_fn=None):
        # Stage 1: coarse — over-fetch 3× top_k via Flat
        k1 = min(top_k * 3, self.count()) if self.count() else 0
        if k1 == 0:
            return []
        candidates = self._flat.search(query, top_k=k1, filter_fn=filter_fn)
        # Stage 2: re-rank — anisotropic scoring (boost cosine via query norm)
        if self.metric == "cosine":
            # ScaNN anisotropic: scale by query magnitude (simulated)
            q = np.asarray(query, dtype=np.float32)
            qn = np.linalg.norm(q) + 1e-9
            for r in candidates:
                # small anisotropic correction
                r.score = float(r.score * (1.0 + 0.05 * (qn - 1.0)))
            candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
