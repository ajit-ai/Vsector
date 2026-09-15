"""VS-09.1: WAL recovery / crash-consistency tests.

Each scenario simulates a process restart by shutting down the WAL flusher of the
current environment and constructing a brand-new stack (fresh MetadataStore,
ShardRouter/EtcdStore, IngestService, QueryEngine) over the SAME durable on-disk
directories. Only the on-disk state (SSTables + WAL) carries across a "restart".
"""
from __future__ import annotations

import os
import uuid

import pytest

from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService
from vsector.query.engine import QueryEngine

SHARD_ID = "shard-fixed-recovery"


def _env(base_dir: str, ns_name: str = "ns1", dimension: int = 4):
    """Build a full stack over `base_dir`. Namespace persisted to disk when first created."""
    shard_id = f"{SHARD_ID}-{ns_name}"
    md = MetadataStore(path=f"{base_dir}/metadata.json")
    etcd = EtcdStore()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir=base_dir)
    query = QueryEngine(md, router, ingest)
    if not md.get(ns_name):
        ns = Namespace(name=ns_name, dimension=dimension, index_type=IndexType.FLAT, distance_metric=DistanceMetric.COSINE)
        md.create(ns)
    if not router.etcd.list_by_namespace(ns_name):
        router.register_shard(Shard(namespace=ns_name, node_id="n1", id=shard_id))
    return ingest, query


def _shutdown(ingest: IngestService) -> None:
    """Stop background WAL flusher threads so a fresh process can take over the files."""
    for ctx in ingest._shards.values():
        try:
            ctx.wal.close()
        except Exception:
            pass


def _vec(x, dim: int = 4) -> list[float]:
    # distinct direction per x so cosine NN results are deterministic
    v = [0.0] * dim
    v[0] = 1.0
    v[1] = float(x)
    return v


def _uid(i) -> str:
    return str(uuid.UUID(int=i))


def _ctx(ingest: IngestService, ns: str = "ns1"):
    return ingest._get_or_create_ctx(ns, ingest.router.route(ns, _uid(0)))


# --- 1. empty state ---------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_state_recovers_cleanly(tmp_path):
    ingest, _ = _env(str(tmp_path))
    assert _ctx(ingest).recovery.records == []
    _shutdown(ingest)

    ingest2, query2 = _env(str(tmp_path))
    ctx2 = _ctx(ingest2)
    assert ctx2.recovery.records == []
    assert ctx2.recovery.recovered_from_wal == 0
    assert await ingest2.fetch("ns1", [_uid(9)]) == []
    out = await query2.query("ns1", _vec(1), top_k=5, include_vector=True, timeout_ms=5000)
    assert out["results"] == []
    _shutdown(ingest2)


# --- 2. WAL-only records survive a restart ----------------------------------

@pytest.mark.asyncio
async def test_wal_only_records_recover_after_restart(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    ids = [_uid(i) for i in (1, 2, 3)]
    await ingest.upsert("ns1", [{"id": ids[i], "vector": _vec(i)} for i in range(3)])
    # durable at this point: WAL flushed, no SSTable yet (memtable threshold 10k)
    assert _ctx(ingest).segments.memtable.count() == 3
    assert _ctx(ingest).segments.durable_records_by_id() == {}
    _shutdown(ingest)

    ingest2, query2 = _env(base)
    ctx2 = _ctx(ingest2)
    assert ctx2.recovery.recovered_from_wal == 3
    assert ctx2.recovery.records and len(ctx2.recovery.records) == 3
    # served from MemTable (reinserted) so fetch works
    fetched = await ingest2.fetch("ns1", ids)
    assert sorted(r["id"] for r in fetched) == sorted(ids)
    out = await query2.query("ns1", _vec(1), top_k=3, include_vector=True, timeout_ms=5000)
    assert out["results"] and out["results"][0]["id"] == ids[1]
    _shutdown(ingest2)


# --- 3. multiple records across batches -------------------------------------

@pytest.mark.asyncio
async def test_multiple_records_across_batches_recover(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    ids = [_uid(i + 1) for i in range(10)]
    await ingest.upsert("ns1", [{"id": ids[i], "vector": _vec(i)} for i in range(5)])
    await ingest.upsert("ns1", [{"id": ids[5 + i], "vector": _vec(i + 10)} for i in range(5)])
    _shutdown(ingest)

    ingest2, query2 = _env(base)
    ctx2 = _ctx(ingest2)
    assert ctx2.recovery.recovered_from_wal == 10
    out = await query2.query("ns1", _vec(13), top_k=1, include_vector=True, timeout_ms=5000)
    assert out["results"] and out["results"][0]["id"] == ids[8]  # the 13-vector
    _shutdown(ingest2)


# --- 4. update recovers latest version --------------------------------------

@pytest.mark.asyncio
async def test_updated_record_recovers_latest_version(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    a = _uid(1)
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(1), "metadata": {"gen": 1}}])
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(1), "metadata": {"gen": 2}}])
    _shutdown(ingest)

    ingest2, _ = _env(base)
    fetched = await ingest2.fetch("ns1", [a])
    assert len(fetched) == 1
    assert fetched[0]["metadata"]["gen"] == 2  # last WAL write wins, not the SSTable/base
    _shutdown(ingest2)


# --- 5. delete remains deleted ----------------------------------------------

@pytest.mark.asyncio
async def test_delete_remains_deleted_after_restart(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    a, b = _uid(1), _uid(2)
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(1)}, {"id": b, "vector": _vec(2)}])
    await ingest.delete("ns1", ids=[a])
    _shutdown(ingest)

    ingest2, query2 = _env(base)
    assert await ingest2.fetch("ns1", [a]) == []
    assert (await ingest2.fetch("ns1", [b]))[0]["id"] == b
    out = await query2.query("ns1", _vec(1), top_k=5, include_vector=True, timeout_ms=5000)
    assert [r["id"] for r in out["results"]] == [b]
    _shutdown(ingest2)


# --- 6. update then delete --------------------------------------------------

@pytest.mark.asyncio
async def test_update_then_delete_recovers_as_deleted(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    a = _uid(1)
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(1), "metadata": {"gen": 1}}])
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(1), "metadata": {"gen": 2}}])
    await ingest.delete("ns1", ids=[a])
    _shutdown(ingest)

    ingest2, query2 = _env(base)
    assert await ingest2.fetch("ns1", [a]) == []
    out = await query2.query("ns1", _vec(1), top_k=5, include_vector=True, timeout_ms=5000)
    assert out["results"] == []
    _shutdown(ingest2)


# --- 7. delete then re-insert ------------------------------------------------

@pytest.mark.asyncio
async def test_delete_then_reinsert_recovers_reinserted(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    a = _uid(1)
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(1), "metadata": {"gen": 1}}])
    await ingest.delete("ns1", ids=[a])
    # re-insert: final WAL state must be "present", tombstones do not shadow the later upsert
    await ingest.upsert("ns1", [{"id": a, "vector": _vec(5), "metadata": {"gen": 3}}])
    _shutdown(ingest)

    ingest2, query2 = _env(base)
    fetched = await ingest2.fetch("ns1", [a])
    assert len(fetched) == 1
    assert fetched[0]["metadata"]["gen"] == 3
    out = await query2.query("ns1", _vec(5), top_k=1, include_vector=True, timeout_ms=5000)
    assert out["results"] and out["results"][0]["id"] == a
    _shutdown(ingest2)


# --- 8. idempotent recovery --------------------------------------------------

@pytest.mark.asyncio
async def test_recovery_is_idempotent(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    ids = [_uid(i + 1) for i in range(6)]
    await ingest.upsert("ns1", [{"id": ids[i], "vector": _vec(i)} for i in range(6)])
    await ingest.delete("ns1", ids=[ids[2]])
    _shutdown(ingest)

    ingest1, _ = _env(base)
    ctx1 = _ctx(ingest1)
    first_state = sorted((str(r.id), r.metadata) for r in ctx1.recovery.records)
    first_count = ctx1.recovery.recovered_from_wal
    # restart again from the SAME durable files
    _shutdown(ingest1)
    ingest2, query2 = _env(base)
    ctx2 = _ctx(ingest2)
    assert sorted((str(r.id), r.metadata) for r in ctx2.recovery.records) == first_state
    assert ctx2.recovery.recovered_from_wal == first_count
    out = await query2.query("ns1", _vec(3), top_k=5, include_vector=True, timeout_ms=5000)
    assert sorted(r["id"] for r in out["results"]) == sorted(id_ for i, id_ in enumerate(ids) if i != 2)
    _shutdown(ingest2)


# --- 9. multiple namespaces stay isolated ------------------------------------

@pytest.mark.asyncio
async def test_multiple_namespaces_remain_isolated(tmp_path):
    base = str(tmp_path)
    ing_a, _ = _env(base, ns_name="alpha")
    ing_b, _ = _env(base, ns_name="beta")
    a1, b1 = _uid(101), _uid(102)
    await ing_a.upsert("alpha", [{"id": a1, "vector": _vec(1)}])
    await ing_b.upsert("beta", [{"id": b1, "vector": _vec(9)}])
    _shutdown(ing_a)
    _shutdown(ing_b)

    ingest2, query2 = _env(base, ns_name="alpha")
    ib2, _ = _env(base, ns_name="beta")
    assert (await ingest2.fetch("alpha", [a1]))[0]["id"] == a1
    assert await ingest2.fetch("alpha", [b1]) == []
    assert (await ib2.fetch("beta", [b1]))[0]["id"] == b1
    assert await ib2.fetch("beta", [a1]) == []
    assert (await query2.query("alpha", _vec(1), top_k=5, include_vector=True, timeout_ms=5000))["shard_count"] == 1
    _shutdown(ingest2)
    _shutdown(ib2)


# --- 10. resilient to corrupt / truncated WAL tail ---------------------------

@pytest.mark.asyncio
async def test_resilient_to_corrupt_and_truncated_wal_tail(tmp_path):
    base = str(tmp_path)
    ingest, _ = _env(base)
    import struct
    import time as _t

    ids = [_uid(i + 1) for i in range(3)]
    await ingest.upsert("ns1", [{"id": ids[i], "vector": _vec(i)} for i in range(3)])
    ctx = _ctx(ingest)
    ctx.wal.flush()
    # simulate a crash mid-write: corrupt tail after the last valid record
    log_dir = f"{base}/wal/{SHARD_ID}-ns1"
    seg = os.path.join(log_dir, "wal-000000.log")
    with open(seg, "ab") as f:
        # 1) entry with a bad checksum (skipped during replay)
        bad_payload = b'{"bad": true}'
        bad = struct.pack(">IIQ", len(bad_payload), 0xDEADBEEF, _t.time_ns()) + bad_payload
        f.write(bad)
        # 2) truncated header with payload length exceeding the remaining file (replay breaks cleanly)
        f.write(struct.pack(">IIQ", 4096, 0, _t.time_ns()))
    _shutdown(ingest)

    ingest2, query2 = _env(base)
    # valid records still recovered, no exception, no index corruption
    fetched = await ingest2.fetch("ns1", ids)
    assert sorted(r["id"] for r in fetched) == sorted(ids)
    out = await query2.query("ns1", _vec(2), top_k=3, include_vector=True, timeout_ms=5000)
    assert len(out["results"]) == 3
    _shutdown(ingest2)


# --- 11. REST readiness reflects startup recovery (spec §21 / §26) ----------

@pytest.mark.asyncio
async def test_ready_endpoint_reports_recovery_outcome(tmp_path, monkeypatch):
    import vsector.api.rest as rest_module
    from vsector.storage.metadata import MetadataStore as MD

    base = str(tmp_path)
    md = MD(path=f"{base}/metadata.json")
    etcd = EtcdStore()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir=f"{base}/data")
    ns = Namespace(name="ns_ready", dimension=4, index_type=IndexType.FLAT, distance_metric=DistanceMetric.COSINE)
    md.create(ns)
    router.register_shard(Shard(namespace="ns_ready", node_id="n1", id="shard-ready"))

    # simulate a prior uptime that wrote durable WAL data, then "crashed"
    rid = _uid(1)
    await ingest.upsert("ns_ready", [{"id": rid, "vector": _vec(1)}])
    assert ingest.stats()["ns_ready:shard-ready"]["recovered_from_wal"] == 0
    _shutdown(ingest)

    # fresh process: new metadata/router/ingest, then run the boot-time recovery hook
    md2 = MD(path=f"{base}/metadata.json")
    etcd2 = EtcdStore()
    router2 = ShardRouter(etcd2)
    ingest2 = IngestService(md2, router2, base_dir=f"{base}/data")
    router2.register_shard(Shard(namespace="ns_ready", node_id="n1", id="shard-ready"))
    monkeypatch.setattr(rest_module, "metadata_store", md2)
    monkeypatch.setattr(rest_module, "router", router2)
    monkeypatch.setattr(rest_module, "ingest", ingest2)
    monkeypatch.setattr(rest_module, "recovery_status", {})

    rest_module._run_startup_recovery()
    resp = await rest_module.ready()
    assert resp["ready"] is True
    assert resp["namespaces"] == 1
    assert resp["recovery"]["ns_ready:shard-ready"] == "recovered:1"
    _shutdown(ingest2)