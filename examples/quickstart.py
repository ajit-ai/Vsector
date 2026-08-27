"""Quickstart - end-to-end ingest + query using internal services (no HTTP)."""
import asyncio
import numpy as np
from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService
from vsector.query.engine import QueryEngine

async def main():
    md = MetadataStore()
    etcd = EtcdStore()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir="./data_quickstart")
    query = QueryEngine(md, router, ingest)

    ns = Namespace(name="products", dimension=8, distance_metric=DistanceMetric.COSINE, index_type=IndexType.HNSW, shard_count=2)
    md.create(ns)
    for i in range(2):
        router.register_shard(Shard(namespace="products", node_id=f"node-{i}"))

    # upsert
    vectors = [{"vector": np.random.randn(8).tolist(), "metadata": {"category": "books", "price": 10+i}, "tags": ["hot"]} for i in range(20)]
    res = await ingest.upsert("products", vectors)
    print("upsert:", res)

    # query
    q = np.random.randn(8).tolist()
    out = await query.query("products", q, top_k=5, filters={"category": {"$eq": "books"}})
    print("query:", out["results"][:2], "took", out["took_ms"], "ms shards", out["shard_count"])

if __name__ == "__main__":
    asyncio.run(main())
