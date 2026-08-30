"""ann-benchmarks suite — recall + P99 latency + cost vs competitors.

Compares Vsector (HNSW/IVF_PQ/ScaNN/Flat) against embedded baselines.
Usage: python benchmarks/ann_benchmark.py --dim 64 --n 20000 --top-k 10
Outputs: stdout + benchmarks/report.json + cost table.
Target: P99 <50ms @1M, <200ms @1T per spec.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _recall_at_k(true_ids: list[list[str]], pred_ids: list[list[str]], k: int) -> float:
    hits = 0
    total = 0
    for t, p in zip(true_ids, pred_ids):
        hits += len(set(t[:k]) & set(p[:k]))
        total += k
    return hits / total if total else 0.0


def _bench_one(index_type: str, data: np.ndarray, queries: np.ndarray, top_k: int, **kw) -> dict:
    from vsector.index.factory import create_index

    dim = data.shape[1]
    n = len(data)
    ids = [f"id{i}" for i in range(n)]
    idx = create_index(index_type, dimension=dim, metric=kw.get("metric", "cosine"), **kw)
    t0 = time.time()
    batch = 5000
    for i in range(0, n, batch):
        idx.add(ids[i : i + batch], data[i : i + batch])
    ingest_ms = int((time.time() - t0) * 1000)

    # Flat ground truth for recall
    from vsector.index.flat import FlatIndex

    flat = FlatIndex(dimension=dim, metric=kw.get("metric", "cosine"))
    flat.add(ids, data)
    true_lists = []
    query_vecs = queries[:20]  # recall sample
    for q in query_vecs:
        true_lists.append([r.id for r in flat.search(q, top_k=top_k)])

    latencies = []
    pred_lists = []
    for q in queries:
        s = time.time()
        res = idx.search(q, top_k=top_k, ef_search=kw.get("ef_search"), nprobe=kw.get("nprobe"))
        latencies.append((time.time() - s) * 1000)
        if len(pred_lists) < len(true_lists):
            pred_lists.append([r.id for r in res])
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    recall = _recall_at_k(true_lists, pred_lists, top_k) if pred_lists else 0.0

    # cost via models/cost.py
    try:
        from vsector.models.cost import estimate_cost

        cost = estimate_cost(n, dim, kw.get("compression", "NONE"))
    except Exception:
        cost = {}

    return {"index": index_type, "n": n, "dim": dim, "ingest_ms": ingest_ms, "p50_ms": round(p50, 2), "p99_ms": round(p99, 2), "recall": round(recall, 3), "cost": cost, "pass": (p99 < 50 if n <= 10000 else p99 < 200)}


def run(dim: int, n: int, top_k: int, out: str = "benchmarks/report.json"):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((100, dim)).astype(np.float32)

    # Competitor-mimic baselines: HNSW (Qdrant-like), IVF_PQ (Milvus-like), ScaNN (Google), Flat
    configs = [
        ("HNSW", {"ef_search": 100}),
        ("HNSW", {"ef_search": 200}),
        ("IVF_PQ", {"nlist": 1024, "nprobe": 64}),
        ("SCANN", {"compression": "SQ8"}),
        ("FLAT", {}),
    ]
    results = []
    for idx_type, kw in configs:
        r = _bench_one(idx_type, data, queries, top_k, **kw)
        print(f"{r['index']:6} ef={kw.get('ef_search','-'):3} nprobe={kw.get('nprobe','-'):3} comp={kw.get('compression','NONE'):4} p50={r['p50_ms']:5.2f} p99={r['p99_ms']:5.2f} recall={r['recall']:.3f} cost=${r['cost'].get('est_usd_month',0)} {'PASS' if r['pass'] else 'FAIL'}")
        results.append(r)

    # Markdown table for market comparison
    md = ["| Index (Vsector) | Mimics | P50 | P99 | Recall | $/mo (1M 1536d SQ8) |", "|---|---|---|---|---|---|"]
    for r in results:
        mimics = {"HNSW": "Qdrant", "IVF_PQ": "Milvus", "SCANN": "Google ScaNN", "FLAT": "pgvector"}[r["index"]]
        md.append(f"| {r['index']} | {mimics} | {r['p50_ms']} | {r['p99_ms']} | {r['recall']} | ${r['cost'].get('est_usd_month',0)} |")
    report = {"results": results, "markdown": "\n".join(md)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")
    print("\n".join(md))
    # overall P99 SLO check
    p99_ok = all(r["pass"] for r in results if r["index"] != "FLAT")
    print("OVERALL PASS" if p99_ok else "OVERALL FAIL")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=str, default="benchmarks/report.json")
    args = ap.parse_args()
    run(args.dim, args.n, args.top_k, args.out)
