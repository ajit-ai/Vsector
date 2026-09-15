from __future__ import annotations
import logging
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .ivf_pq import IVFPQIndex
from .scann import ScaNNIndex

logger = logging.getLogger(__name__)

def create_index(index_type: str, dimension: int, metric: str = "cosine", **kwargs):
    t = index_type.upper()
    compression = kwargs.get("compression", "NONE")
    if t == "HNSW":
        # Wrap HNSW with optional compression (SQ8/BF16 via quantize helpers if needed)
        idx = HNSWIndex(dimension=dimension, metric=metric, M=kwargs.get("M", 16), ef_construction=kwargs.get("ef_construction", 200), ef_search=kwargs.get("ef_search", 100))
        idx.compression = compression  # type: ignore
        if getattr(idx, "degraded", False):
            logger.warning("HNSW index created but DEGRADED: hnswlib missing -> FlatIndex fallback. Install optional 'ann' extras: pip install -e '.[ann]'")
        return idx
    elif t in ("IVF_PQ", "IVF-PQ", "IVFPQ"):
        idx = IVFPQIndex(dimension=dimension, metric=metric, nlist=kwargs.get("nlist", 4096), nprobe=kwargs.get("nprobe", 64))
        if getattr(idx, "degraded", False):
            logger.warning("IVF_PQ index created but DEGRADED: faiss missing -> FlatIndex fallback. Install optional 'ann' extras: pip install -e '.[ann]'")
        return idx
    elif t == "SCANN":
        idx = ScaNNIndex(dimension=dimension, metric=metric, num_leaves=kwargs.get("num_leaves", 1024), leaves_to_search=kwargs.get("leaves_to_search", 16), compression=compression)
        if getattr(idx, "degraded", True):
            logger.warning("SCANN index is a simulation wrapper over FlatIndex (no real ScaNN/GPU backend in v0.9.0)")
        return idx
    elif t == "FLAT":
        return FlatIndex(dimension=dimension, metric=metric)
    else:
        raise ValueError(f"unknown index_type {index_type}")
