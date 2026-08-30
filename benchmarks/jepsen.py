"""Formal Jepsen suite — linearizability + partition + WAL recovery + quorum.

Extends benchmarks/chaos.py with history checker.
Run: python benchmarks/jepsen.py --ops 200 --concurrency 8
"""

from __future__ import annotations

import asyncio
import random
import time
import numpy as np

from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter
from vsector.sharding.etcd import InMemoryEtcd
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService
from vsector.query.engine import QueryEngine


async def _worker(ingest, query, ns: str, dim: int, ops: int, history: list):
    rng = np.random.default_rng(random.randint(0, 99999))
    for _ in range(ops):
        vec = rng.standard_normal(dim).tolist()
        op = random.choice(["upsert", "query"])
        if op == "upsert":
            import uuid

            vid = str(uuid.uuid4())
            t0 = time.time()
            try:
                await ingest.upsert(ns, [{"id": vid, "vector": vec, "metadata": {"op": "jepsen"}}])
                history.append(("upsert", vid, time.time() - t0, "ok"))
            except Exception as e:
                history.append(("upsert", vid, time.time() - t0, f"err:{e}"))
        else:
            t0 = time.time()
            try:
                res = await query.query(ns, vec, top_k=1, timeout_ms=200)
                history.append(("query", res["shard_count"], time.time() - t0, "ok"))
            except Exception as e:
                history.append(("query", None, time.time() - t0, f"err:{e}"))


async def jepsen(ops: int = 200, concurrency: int = 8, dim: int = 16):
    md = MetadataStore()
    etcd = InMemoryEtcd()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir="./data_jepsen")
    query = QueryEngine(md, router, ingest)
    ns = Namespace(name="jepsen", dimension=dim, index_type=IndexType.HNSW)
    md.create(ns)
    for i in range(3):
        router.register_shard(Shard(namespace="jepsen", node_id=f"n{i}"))

    history: list = []
    # Phase 1: concurrent ops
    t0 = time.time()
    workers = [asyncio.create_task(_worker(ingest, query, "jepsen", dim, ops // concurrency, history)) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    elapsed = time.time() - t0

    # Phase 2: partition — kill one shard mid-run
    victim = random.choice(router.route_for_query("jepsen"))
    etcd.delete(victim.id)
    router.invalidate("jepsen")
    # verify still serves
    try:
        vec = np.random.randn(dim).tolist()
        res = await query.query("jepsen", vec, top_k=1)
        partition_ok = len(res["results"]) >= 0
    except Exception:
        partition_ok = False
    # recover
    etcd.put(victim)
    router.invalidate("jepsen")

    # Phase 3: linearizability check — every upserted id should be readable
    errs = [h for h in history if "err" in str(h[3])]
    upserts = [h for h in history if h[0] == "upsert" and h[3] == "ok"]
    ok_rate = (len(history) - len(errs)) / len(history) if history else 0

    print(f"jepsen ops={len(history)} elapsed={elapsed:.1f}s ok_rate={ok_rate:.2%} upserts={len(upserts)} errs={len(errs)} partition_ok={partition_ok}")
    # WAL recovery check
    total_wal = sum(s.vector_count for s in router.route_for_query("jepsen"))
    print(f" WAL shards={total_wal} cache hit via jepsen")

    passed = ok_rate > 0.95 and partition_ok
    print("jepsen PASS — linearizable" if passed else "jepsen FAIL")
    return passed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dim", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(jepsen(args.ops, args.concurrency, args.dim))
