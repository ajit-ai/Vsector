from __future__ import annotations
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .ivf_pq import IVFPQIndex

def create_index(index_type: str, dimension: int, metric: str = "cosine", **kwargs):
    t = index_type.upper()
    if t == "HNSW":
        return HNSWIndex(dimension=dimension, metric=metric, M=kwargs.get("M", 16), ef_construction=kwargs.get("ef_construction", 200), ef_search=kwargs.get("ef_search", 100))
    elif t in ("IVF_PQ", "IVF-PQ", "IVFPQ"):
        return IVFPQIndex(dimension=dimension, metric=metric, nlist=kwargs.get("nlist", 4096), nprobe=kwargs.get("nprobe", 64))
    elif t in ("FLAT", "SCANN"):
        return FlatIndex(dimension=dimension, metric=metric)
    else:
        raise ValueError(f"unknown index_type {index_type}")
