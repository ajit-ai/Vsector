import asyncio, numpy as np, pytest
from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService
from vsector.query.engine import QueryEngine

@pytest.mark.asyncio
async def test_ingest_query():
    md = MetadataStore()
    etcd = EtcdStore()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir="./data_test")
    query = QueryEngine(md, router, ingest)
    ns = Namespace(name="ns1", dimension=4, index_type=IndexType.FLAT)
    md.create(ns)
    router.register_shard(Shard(namespace="ns1", node_id="n1"))
    vectors = [{"vector": np.random.randn(4).tolist(), "metadata": {"cat": "a"}} for _ in range(10)]
    await ingest.upsert("ns1", vectors)
    out = await query.query("ns1", np.random.randn(4).tolist(), top_k=3)
    assert len(out["results"]) <=3
    assert out["shard_count"]==1
