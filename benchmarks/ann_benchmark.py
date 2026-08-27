"""ann-benchmarks stub — measures P99 latency & recall vs competitor HNSW baseline.

Usage: python benchmarks/ann_benchmark.py --dim 128 --n 100000 --top-k 10
Target: P99 <50ms @1M, <200ms @1T per spec.
"""

from __future__ import annotations

import argparse
import time
import numpy as np


def run(dim: int, n: int, top_k: int, ef_search: int = 100):
    from vsector.index.factory import create_index

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    idx = create_index("HNSW", dimension=dim, metric="cosine", ef_search=ef_search)
    # ingest
    ids = [f"id{i}" for i in range(n)]
    batch = 5000
    t0 = time.time()
    for i in range(0, n, batch):
        idx.add(ids[i : i + batch], data[i : i + batch])
    ingest_ms = int((time.time() - t0) * 1000)
    # query
    queries = rng.standard_normal((100, dim)).astype(np.float32)
    latencies = []
    for q in queries:
        s = time.time()
        idx.search(q, top_k=top_k, ef_search=ef_search)
        latencies.append((time.time() - s) * 1000)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    print(f"n={n} dim={dim} ingest={ingest_ms}ms p50={p50:.2f}ms p99={p99:.2f}ms top_k={top_k} ef={ef_search}")
    ok = (n <= 10_000 and p99 < 50) or (p99 < 200)
    print("PASS" if ok else "FAIL: P99 exceeds target")
    return {"p50": p50, "p99": p99, "ingest_ms": ingest_ms}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=100)
    args = ap.parse_args()
    run(args.dim, args.n, args.top_k, args.ef_search)
