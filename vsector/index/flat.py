"""FLAT brute-force index."""
from __future__ import annotations

import numpy as np
from .base import BaseIndex, IndexResult


class FlatIndex(BaseIndex):
    def __init__(self, dimension: int, metric: str = "cosine"):
        super().__init__(dimension, metric)
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None
        self.metadatas: list[dict] = []
        self.id_to_idx: dict[str, int] = {}

    def add(self, ids, vectors, metadatas=None):
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.shape[1] == self.dimension
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        start = len(self.ids)
        for i, _id in enumerate(ids):
            self.ids.append(_id)
            self.id_to_idx[_id] = start + i
            self.metadatas.append(metadatas[i] if metadatas else {})

    def delete(self, ids):
        keep = [i for i, _id in enumerate(self.ids) if _id not in set(ids)]
        self.ids = [self.ids[i] for i in keep]
        self.metadatas = [self.metadatas[i] for i in keep]
        if self.vectors is not None and keep:
            self.vectors = self.vectors[keep]
        elif not keep:
            self.vectors = None
        self.id_to_idx = { _id: idx for idx, _id in enumerate(self.ids)}

    def count(self): return len(self.ids)

    def search(self, query, top_k=10, ef_search=None, nprobe=None, filter_fn=None):
        if self.vectors is None or len(self.ids)==0:
            return []
        query = np.asarray(query, dtype=np.float32)
        scores = self._score(query, self.vectors)
        # filter
        if filter_fn is not None:
            mask = np.array([filter_fn(self.metadatas[i]) for i in range(len(self.ids))])
            # set filtered scores to -inf so they sink
            scores = np.where(mask, scores, -1e9)
        # top_k
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k-1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        res=[]
        for j in idx:
            if scores[j] <= -1e8:
                continue
            res.append(IndexResult(id=self.ids[j], score=float(scores[j]), vector=self.vectors[j].tolist(), metadata=self.metadatas[j]))
        return res
