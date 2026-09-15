"""Pluggable index engine base."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IndexResult:
    id: str
    score: float
    vector: list[float] | None = None
    metadata: dict[str, Any] | None = None


class BaseIndex(abc.ABC):
    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric.lower()
        self.backend_name = "abstract"
        self.is_native_backend = False

    @abc.abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict] | None = None) -> None: ...

    @abc.abstractmethod
    def search(self, query: np.ndarray, top_k: int = 10, ef_search: int | None = None, nprobe: int | None = None, filter_fn=None) -> list[IndexResult]: ...

    @abc.abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return v / norms
        return v

    def _score(self, q: np.ndarray, db: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            # cosine similarity (higher better)
            qn = q / (np.linalg.norm(q) + 1e-9)
            dbn = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
            return dbn @ qn
        elif self.metric == "euclidean":
            # convert distance to score (negative distance)
            return -np.linalg.norm(db - q, axis=1)
        elif self.metric == "dot_product":
            return db @ q
        elif self.metric == "manhattan":
            return -np.sum(np.abs(db - q), axis=1)
        else:
            return -np.linalg.norm(db - q, axis=1)
