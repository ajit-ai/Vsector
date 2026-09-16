"""VS-09.2: real replication & failure semantics tests.

Primary writes are pushed to independently represented in-process replica
contexts through a real application path (WAL + SegmentStore + index on the
replica); acknowledgements reflect actual results.  No ``sleep + True``.
"""
from __future__ import annotations

import uuid

import pytest

from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService

SHARD_ID = "shard-repl-1"


def _uid(i: int) -> str:
    return str(uuid.UUID(int=i))


def _vec(x: float, dim: int = 4) -> list[float]:
    # distinct direction per x so a re-upsert differs in checksum
    v = [0.0] * dim
    v[0] = 1.0
    v[1] = float(x)
    return v


def _env(base_dir: str, ns_name: str = "ns_repl", dimension: int = 4, required_acks: int = 1,
         replicas: tuple[str, ...] = ("replica-node-1",)):
    md = MetadataStore(path=f"{base_dir}/metadata.json")
    router = ShardRouter(EtcdStore())
    ingest = IngestService(md, router, base_dir=base_dir)
    ns = md.get(ns_name)
    if ns is None:
        ns = Namespace(name=ns_name, dimension=dimension, index_type=IndexType.FLAT,
                       distance_metric=DistanceMetric.COSINE, required_acks=required_acks)
        md.create(ns)
    shard = Shard(namespace=ns_name, node_id="primary-node", id=SHARD_ID, replicas=list(replicas))
    if not router.etcd.list_by_namespace(ns_name):
        router.register_shard(shard)
    return ingest, ns, shard


def _replica_ep(ingest: IngestService, ns_name: str, shard: Shard, node_id: str):
    return ingest._get_or_create_replica_endpoint(ns_name, shard, node_id)


def _shutdown_all(ingest: IngestService) -> None:
    """Stop every WAL flusher (primary + replicas) so a fresh process can take over files."""
    for ctx in list(ingest._shards.values()):
        try:
            ctx.wal.close()
        except Exception:
            pass
    for ep in list(ingest._replica_endpoints.values()):
        try:
            ep.close()
        except Exception:
            pass


# --- 1. primary + replica success ------------------------------------------

@pytest.mark.asyncio
async def test_primary_and_replica_replicated_upsert(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_ok")
    rid = _uid(1)
    res = await ingest.upsert("ns_ok", [{"id": rid, "vector": _vec(1)}])
    assert res["upserted"] == 1
    assert res["errors"] == []

    repl = res["replication"]
    assert repl["success"] is True
    assert repl["degraded"] is False
    assert repl["acknowledged"] == 1
    assert repl["required_acks"] == 1
    assert repl["failed_replicas"] == []

    # replica independently contains the same logical record
    ep = _replica_ep(ingest, "ns_ok", shard, "replica-node-1")
    assert ep.contains(rid)
    rec = ep.record(rid)
    assert rec is not None and rec.vector == _vec(1)
    # primary contains it too
    fetched = await ingest.fetch("ns_ok", [rid])
    assert len(fetched) == 1 and fetched[0]["id"] == rid
    _shutdown_all(ingest)


# --- 2. replica state is independent ---------------------------------------

@pytest.mark.asyncio
async def test_replica_state_is_independent(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_indep")
    ep = _replica_ep(ingest, "ns_indep", shard, "replica-node-1")
    rid = _uid(2)
    pctx = ingest._get_or_create_ctx("ns_indep", shard)

    # distinct object graph: never the same WAL / SegmentStore / index / tombstone set
    assert pctx is not ep.ctx
    assert pctx.wal is not ep.ctx.wal
    assert pctx.wal.dir != ep.ctx.wal.dir
    assert pctx.segments is not ep.ctx.segments
    assert pctx.index is not ep.ctx.index

    # before replication the replica is genuinely empty -> state is NOT shared with the primary
    assert ep.ctx.segments.count() == 0
    assert ep.contains(rid) is False

    await ingest.upsert("ns_indep", [{"id": rid, "vector": _vec(2)}])
    # convergence happens through the replication path, not shared objects
    assert ep.contains(rid) is True
    assert ep.ctx.segments.count() == 1
    _shutdown_all(ingest)


# --- 3. replica failure, primary-only acknowledgement -----------------------

@pytest.mark.asyncio
async def test_replica_failure_primary_only_ack(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_deg", required_acks=1)
    ingest.replicator.fail_node("replica-node-1")
    rid = _uid(3)

    res = await ingest.upsert("ns_deg", [{"id": rid, "vector": _vec(3)}])
    assert res["upserted"] == 1
    repl = res["replication"]
    assert repl["success"] is True          # primary durable write is sufficient
    assert repl["degraded"] is True         # replica failure is NOT hidden
    assert repl["acknowledged"] == 0
    assert "replica-node-1" in repl["failed_replicas"]

    # primary data remains available
    fetched = await ingest.fetch("ns_deg", [rid])
    assert len(fetched) == 1
    # replica genuinely never received it
    ep = _replica_ep(ingest, "ns_deg", shard, "replica-node-1")
    assert ep.contains(rid) is False
    _shutdown_all(ingest)


# --- 4. replica failure when acknowledgement is required --------------------

@pytest.mark.asyncio
async def test_replica_failure_when_ack_required(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_require", required_acks=2)
    ingest.replicator.fail_node("replica-node-1")
    rid = _uid(4)

    res = await ingest.upsert("ns_require", [{"id": rid, "vector": _vec(4)}])
    assert res["errors"] == []              # primary write itself succeeded
    repl = res["replication"]
    assert repl["success"] is False         # required ack not satisfied -> no fake success
    assert repl["degraded"] is True
    assert repl["acknowledged"] == 0
    assert repl["required_acks"] == 2
    assert "replica-node-1" in repl["failed_replicas"]

    # primary durable state: the write IS persisted locally (no silent rollback, no ambiguity)
    fetched = await ingest.fetch("ns_require", [rid])
    assert len(fetched) == 1 and fetched[0]["id"] == rid
    assert ingest._shards["ns_require:shard-repl-1"].shard.vector_count == 1

    # retry after the replica becomes available converges to full replication
    ingest.replicator.heal_node("replica-node-1")
    res2 = await ingest.upsert("ns_require", [{"id": rid, "vector": _vec(4)}])
    assert res2["replication"]["success"] is True
    assert res2["replication"]["degraded"] is False
    assert res2["replication"]["acknowledged"] == 1
    ep = _replica_ep(ingest, "ns_require", shard, "replica-node-1")
    assert ep.contains(rid)
    _shutdown_all(ingest)


# --- 5. primary failure -----------------------------------------------------

@pytest.mark.asyncio
async def test_primary_failure_no_false_ack(tmp_path, monkeypatch):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_primary_fail", required_acks=1)
    rid = _uid(5)
    pctx = ingest._get_or_create_ctx("ns_primary_fail", shard)

    def _boom(*_a, **_k):
        raise RuntimeError("primary local apply failed")

    monkeypatch.setattr(pctx.segments, "put", _boom)
    res = await ingest.upsert("ns_primary_fail", [{"id": rid, "vector": _vec(5)}])
    assert res["upserted"] == 0
    assert res["errors"] and "primary local apply failed" in res["errors"][0]["error"]

    repl = res["replication"]
    assert repl["success"] is False         # do not claim replicated success
    assert repl["attempted"] == 0           # replication never proceeded
    assert repl["acknowledged"] == 0
    assert repl["failed_replicas"] == []
    ep = _replica_ep(ingest, "ns_primary_fail", shard, "replica-node-1")
    assert ep.contains(rid) is False
    _shutdown_all(ingest)


# --- 6. idempotent replicated upsert ---------------------------------------

@pytest.mark.asyncio
async def test_idempotent_replicated_upsert(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_idem")
    rid = _uid(6)
    payload = [{"id": rid, "vector": _vec(6), "metadata": {"gen": 1}}]

    res1 = await ingest.upsert("ns_idem", payload)
    res2 = await ingest.upsert("ns_idem", payload)  # same logical operation delivered twice
    assert res1["replication"]["success"] is True
    assert res2["replication"]["success"] is True

    ep = _replica_ep(ingest, "ns_idem", shard, "replica-node-1")
    # one logical record, not a duplicate
    assert ep.ctx.segments.memtable.count() == 1
    assert ep.ctx.index.count() == 1
    assert ep.record(rid).metadata == {"gen": 1}
    _shutdown_all(ingest)


# --- 7. replicated delete ---------------------------------------------------

@pytest.mark.asyncio
async def test_replicated_delete(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_del")
    rid = _uid(7)
    await ingest.upsert("ns_del", [{"id": rid, "vector": _vec(7)}])
    ep = _replica_ep(ingest, "ns_del", shard, "replica-node-1")
    assert ep.contains(rid)

    res = await ingest.delete("ns_del", ids=[rid])
    assert res["deleted"] == 1
    assert res["replication"]["success"] is True

    # both primary and replica hide the record
    assert await ingest.fetch("ns_del", [rid]) == []
    assert ep.contains(rid) is False
    assert rid in ep.ctx.tombstones
    assert ep.ctx.index.count() == 0
    _shutdown_all(ingest)


# --- 8. replicated delete survives replica restart --------------------------

@pytest.mark.asyncio
async def test_replicated_delete_survives_replica_restart(tmp_path):
    base = str(tmp_path)
    ns_name = "ns_del_restart"
    rid = _uid(8)

    ingest1, _, shard1 = _env(base, ns_name)
    await ingest1.upsert(ns_name, [{"id": rid, "vector": _vec(8)}])
    await ingest1.delete(ns_name, ids=[rid])   # tombstone replicated + durable on replica
    _shutdown_all(ingest1)

    # fresh process over the same durable replica directory
    ingest2, _, shard2 = _env(base, ns_name)
    ep2 = _replica_ep(ingest2, ns_name, shard2, "replica-node-1")
    # recovery found the WAL entries (upsert + tombstone) -> record stays deleted
    assert ep2.ctx.recovery.wal_entries_read == 2
    assert ep2.ctx.recovery.recovered_from_wal == 0
    assert ep2.contains(rid) is False
    assert rid in ep2.deleted_ids


    # re-upsert resurrection on the restarted replica
    res = await ingest2.upsert(ns_name, [{"id": rid, "vector": _vec(9)}])
    assert res["replication"]["success"] is True
    assert ep2.contains(rid) is True
    assert ep2.record(rid).vector == _vec(9)
    _shutdown_all(ingest2)


# --- 9. replicated upsert survives replica restart --------------------------

@pytest.mark.asyncio
async def test_replicated_upsert_survives_replica_restart(tmp_path):
    base = str(tmp_path)
    ns_name = "ns_up_restart"
    rid = _uid(9)

    ingest1, _, shard1 = _env(base, ns_name)
    res = await ingest1.upsert(ns_name, [{"id": rid, "vector": _vec(10)}])
    assert res["replication"]["acknowledged"] == 1
    _shutdown_all(ingest1)

    ingest2, _, shard2 = _env(base, ns_name)
    ep2 = _replica_ep(ingest2, ns_name, shard2, "replica-node-1")
    # VS-09.1 recovery re-established the replicated record from the replica WAL
    assert ep2.ctx.recovery.wal_entries_read == 1
    assert ep2.ctx.recovery.recovered_from_wal == 1
    assert ep2.contains(rid) is True
    rec = ep2.record(rid)
    assert rec is not None and rec.vector == _vec(10)
    _shutdown_all(ingest2)


# --- optional: multiple replicas / partial success / retry ------------------

@pytest.mark.asyncio
async def test_partial_replica_success_and_retry(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_partial", required_acks=2, replicas=("rep-a", "rep-b"))
    ingest.replicator.fail_node("rep-a")
    rid = _uid(11)

    # primary + rep-b = 2 acks >= required 2 -> degraded success (rep-a failed)
    res = await ingest.upsert("ns_partial", [{"id": rid, "vector": _vec(11)}])
    repl = res["replication"]
    assert repl["success"] is True
    assert repl["degraded"] is True
    assert repl["acknowledged"] == 1
    assert sorted(repl["failed_replicas"]) == ["rep-a"]

    # a third replica would be needed for the stricter policy -> honest failure
    res2 = await ingest.upsert("ns_partial", [{"id": _uid(12), "vector": _vec(12)}])
    assert res2["replication"]["acknowledged"] == 1  # only rep-b applied

    # after the failed node recovers, full replication is achievable
    ingest.replicator.heal_node("rep-a")
    res3 = await ingest.upsert("ns_partial", [{"id": _uid(13), "vector": _vec(13)}])
    assert res3["replication"]["success"] is True
    assert res3["replication"]["degraded"] is False
    assert res3["replication"]["acknowledged"] == 2
    _shutdown_all(ingest)


# --- optional: acknowledgement counting via stats ---------------------------

@pytest.mark.asyncio
async def test_stats_expose_replication_counts(tmp_path):
    base = str(tmp_path)
    ingest, _, _ = _env(base, "ns_stats", required_acks=1)
    ingest.replicator.fail_node("replica-node-1")
    await ingest.upsert("ns_stats", [{"id": _uid(14), "vector": _vec(14)}])
    stats = ingest.stats()
    key = "ns_stats:shard-repl-1"
    assert key in stats
    rep = stats[key]["replication"]
    assert rep["attempts"] >= 1
    assert rep["failures"] >= 1
    assert rep["degraded"] >= 1
    _shutdown_all(ingest)