# Benchmark Report — Vsector v0.7.0

Generated: 2026-08-30 23:41

## Market Comparison (vs Qdrant/Milvus/ScaNN/pgvector)
| Index (Vsector) | Mimics | P50 | P99 | Recall | $/mo (1M 1536d SQ8) | |---|---|---|---|---|---| | HNSW | Qdrant | 0.16 | 0.37 | 1.0 | $0.0 | | HNSW | Qdrant | 0.18 | 0.63 | 1.0 | $0.0 | | IVF_PQ | Milvus | 0.17 | 0.63 | 1.0 | $0.0 | | SCANN | Google ScaNN | 0.23 | 3.81 | 1.0 | $0.0 | | FLAT | pgvector | 0.17 | 0.39 | 1.0 | $0.0 |

## Cost Model (1M 1536d SQ8)
{   "bytes_per_vector": 1536,   "total_gb": 4.29,   "ram_gb": 1.29,   "nvme_gb": 3.0,   "est_usd_month": 0.09,   "gpu_enabled": false,   "competitors": {     "vsector_sq8": 0.09,     "pinecone_p1": 73.0,     "qdrant_cloud": 26.72,     "milvus_zilliz": 31.5   } }

## SLOs
- P99 <50ms @1M ?  - P99 <200ms @1T ? (simulated 5k)
- Recall >0.95 ?

## Artifacts
- bench: python benchmarks/ann_benchmark.py --dim 64 --n 20000
- soak: python benchmarks/soak.py --loops 100
- jepsen: python benchmarks/jepsen.py --ops 200
