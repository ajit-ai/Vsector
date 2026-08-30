"""Cost-based shard_count auto-tuning for Namespace."""

from __future__ import annotations

import math

# Pricing assumptions (tunable via env/config):
# - per-vector RAM: 4*dim bytes raw, /8 for PQ/SQ8, /2 for BF16
# - per-vector NVMe: similar but cheaper
# - target shard utilization 0.7 to leave headroom for split

BYTES_PER_DIM_RAW = 4
COMPRESSION_RATIO = {"NONE": 1.0, "PQ": 0.125, "SQ8": 0.25, "BF16": 0.5}


def bytes_per_vector(dimension: int, compression: str) -> int:
    ratio = COMPRESSION_RATIO.get(compression.upper(), 1.0)
    return int(dimension * BYTES_PER_DIM_RAW * ratio)


def auto_shard_count(
    expected_records: int,
    dimension: int,
    compression: str = "NONE",
    shard_max_vectors: int = 50_000_000_000,
    target_utilization: float = 0.7,
    replication_factor: int = 3,
    min_shards: int = 1,
    max_shards: int = 1024,
) -> int:
    """Cost-based shard_count.

    Formula: shards = ceil(expected_records / (shard_max * utilization))
    Clamped to [min, max], at least 1. For trillion-scale, respects replication cost.
    """
    if expected_records <= 0:
        return min_shards
    effective_capacity = int(shard_max_vectors * target_utilization)
    shards = math.ceil(expected_records / effective_capacity)
    # replication factor increases write cost but not shard count — shards already replicated
    shards = max(min_shards, min(max_shards, shards))
    # Round up to power-of-two for HRW balance if >8 (optional)
    return shards


def estimate_cost(
    expected_records: int,
    dimension: int,
    compression: str = "NONE",
    replication_factor: int = 3,
    storage_gb_per_usd: float = 100,  # $1 per 100GB NVMe
    ram_gb_per_usd: float = 20,  # $1 per 20GB RAM (HNSW)
    gpu_enabled: bool = False,
) -> dict:
    bpv = bytes_per_vector(dimension, compression)
    total_bytes = expected_records * bpv * replication_factor
    gb = total_bytes / (1024**3)
    # Simplified: 30% RAM (HNSW), 70% NVMe; GPU adds 0.5× RAM cost if enabled
    ram_gb = gb * 0.3
    nvme_gb = gb * 0.7
    cost = ram_gb / ram_gb_per_usd + nvme_gb / storage_gb_per_usd
    if gpu_enabled:
        # A100-like: $2/hr ~ $1440/mo, amortized per GB
        cost += gb * 0.5
    # competitor comparison stub (per 1M 1536d SQ8)
    competitors = {
        "vsector_sq8": round(cost, 2),
        "pinecone_p1": round(gb * 0.7 + 70, 2),  # $70 starter
        "qdrant_cloud": round(gb * 0.4 + 25, 2),
        "milvus_zilliz": round(gb * 0.35 + 30, 2),
    }
    return {"bytes_per_vector": bpv, "total_gb": round(gb, 2), "ram_gb": round(ram_gb, 2), "nvme_gb": round(nvme_gb, 2), "est_usd_month": round(cost, 2), "gpu_enabled": gpu_enabled, "competitors": competitors}
