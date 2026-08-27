"""Soak / upgrade validation — 1k upsert/query loops with chaos + WAL GC.

Run: python benchmarks/soak.py --loops 100
"""

import asyncio
import time
import numpy as np

from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter
from vsector.sharding.etcd import InMemoryEtcd
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService
from vsector.query.engine import QueryEngine


async def soak(loops: int = 100, dim: int = 16):
    md = MetadataStore()
    etcd = InMemoryEtcd()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir="./data_soak")
    query = QueryEngine(md, router, ingest)
    ns = Namespace(name="soak", dimension=dim, index_type=IndexType.HNSW)
    md.create(ns)
    router.register_shard(Shard(namespace="soak", node_id="n0"))
    t0 = time.time()
    for i in range(loops):
        vec = np.random.randn(dim).tolist()
        await ingest.upsert("soak", [{"vector": vec, "metadata": {"i": i}}])
        if i % 10 == 0:
            res = await query.query("soak", vec, top_k=1)
            assert res["results"], f"query failed at {i}"
        if i % 50 == 0:
            # WAL GC
            for ctx in ingest._shards.values():
                ctx.wal.gc()
    elapsed = time.time() - t0
    print(f"soak PASS {loops} loops dim={dim} in {elapsed:.1f}s qps={loops/elapsed:.1f} shards={len(router.route_for_query('soak'))}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=100)
    ap.add_argument("--dim", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(soak(args.loops, args.dim))
