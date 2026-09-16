"""VS-09.2 / VS-09.3 / VS-09.4: real replication, failure, and health observability tests.

Primary writes are pushed to independently represented in-process replica
contexts through a real application path (WAL + SegmentStore + index on the
replica); acknowledgements reflect actual results.  No ``sleep + True``.
VS-09.4 covers stable health representation, failure/recovery sequence
observability, the required-ack readiness matrix, metrics consistency, and the
read-only inspection contract.
"""
from __future__ import annotations

import json
import types
import uuid

import pytest

from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard
from vsector.sharding.replication import InProcessReplicaTransport
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
    # VS-09.3: health view is additive in stats and matches the degraded transport state
    health = stats[key]["replication_health"]
    assert health["ready"] is True and health["degraded"] is True
    assert health["configured_replicas"] == 1 and health["healthy_replicas"] == 0
    _shutdown_all(ingest)


# ===========================================================================
# VS-09.3: replication health & readiness
# ===========================================================================


# --- Test 1: all replicas healthy ------------------------------------------

@pytest.mark.asyncio
async def test_health_all_replicas_healthy(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_ok", required_acks=2, replicas=("rep-a",))
    res = await ingest.upsert("ns_health_ok", [{"id": _uid(100), "vector": _vec(100)}])
    assert res["replication"]["success"] is True

    health = ingest.replicator.replication_health("ns_health_ok", shard, required_acks=2)
    assert health.ready is True
    assert health.degraded is False
    assert health.healthy_replicas == health.configured_replicas == 1
    assert health.required_acks == 2
    assert health.replicas[0].healthy is True
    assert health.replicas[0].available is True
    assert health.replicas[0].last_error is None
    assert health.last_failure is None

    # response embeds the additive health snapshot
    rh = res["replication"]["health"]
    assert rh["ready"] is True
    assert rh["degraded"] is False
    assert rh["healthy_replicas"] == 1
    _shutdown_all(ingest)


# --- Test 2: optional replica failure --------------------------------------

@pytest.mark.asyncio
async def test_health_optional_replica_failure(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_opt", required_acks=1, replicas=("rep-a",))
    ingest.replicator.fail_node("rep-a")
    res = await ingest.upsert("ns_health_opt", [{"id": _uid(101), "vector": _vec(101)}])
    assert res["replication"]["success"] is True    # primary-only ack suffices
    assert res["replication"]["degraded"] is True

    health = ingest.replicator.replication_health("ns_health_opt", shard, required_acks=1)
    assert health.ready is True                     # degraded != not-ready
    assert health.degraded is True
    assert health.healthy_replicas == 0
    assert health.replicas[0].healthy is False
    assert health.replicas[0].available is False
    assert "replica unavailable" in (health.replicas[0].last_error or "")
    assert health.last_failure is not None

    # the primary remains usable
    assert len(await ingest.fetch("ns_health_opt", [_uid(101)])) == 1
    rh = res["replication"]["health"]
    assert rh["ready"] is True and rh["degraded"] is True
    _shutdown_all(ingest)


# --- Test 3: required replica failure ---------------------------------------

@pytest.mark.asyncio
async def test_health_required_replica_failure(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_req", required_acks=2, replicas=("rep-a",))
    ingest.replicator.fail_node("rep-a")
    res = await ingest.upsert("ns_health_req", [{"id": _uid(102), "vector": _vec(102)}])
    assert res["replication"]["success"] is False

    health = ingest.replicator.replication_health("ns_health_req", shard, required_acks=2)
    assert health.ready is False
    assert health.degraded is True
    assert health.healthy_replicas == 0
    _shutdown_all(ingest)


# --- Test 4: recovery of a failed replica -----------------------------------

@pytest.mark.asyncio
async def test_health_recovers_after_failure(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_rec", required_acks=2, replicas=("rep-a",))
    ingest.replicator.fail_node("rep-a")
    await ingest.upsert("ns_health_rec", [{"id": _uid(103), "vector": _vec(103)}])
    assert ingest.replicator.replication_health("ns_health_rec", shard, required_acks=2).ready is False

    # healing alone is not a fake recovery: no subsequent success yet
    ingest.replicator.heal_node("rep-a")
    stale = ingest.replicator.replication_health("ns_health_rec", shard, required_acks=2)
    assert stale.ready is False
    assert stale.replicas[0].healthy is False

    # a real successful delivery restores health
    res = await ingest.upsert("ns_health_rec", [{"id": _uid(104), "vector": _vec(104)}])
    assert res["replication"]["success"] is True
    fine = ingest.replicator.replication_health("ns_health_rec", shard, required_acks=2)
    assert fine.ready is True
    assert fine.degraded is False
    assert fine.replicas[0].healthy is True
    assert fine.replicas[0].available is True
    assert fine.replicas[0].last_error is None
    _shutdown_all(ingest)


# --- Test 5: multiple replicas -----------------------------------------------

@pytest.mark.asyncio
async def test_health_multiple_replicas(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_multi", required_acks=1, replicas=("rep-a", "rep-b"))
    await ingest.upsert("ns_health_multi", [{"id": _uid(105), "vector": _vec(105)}])
    h = ingest.replicator.replication_health("ns_health_multi", shard, required_acks=1)
    assert h.ready is True and h.degraded is False
    assert h.healthy_replicas == h.configured_replicas == 2

    # one failed
    ingest.replicator.fail_node("rep-a")
    await ingest.upsert("ns_health_multi", [{"id": _uid(106), "vector": _vec(106)}])
    h1 = ingest.replicator.replication_health("ns_health_multi", shard, required_acks=1)
    assert h1.ready is True and h1.degraded is True
    assert h1.healthy_replicas == 1

    # both failed with the stricter policy -> not ready (1 primary + 0 healthy < 2)
    ingest.replicator.fail_node("rep-b")
    await ingest.upsert("ns_health_multi", [{"id": _uid(107), "vector": _vec(107)}])
    h2 = ingest.replicator.replication_health("ns_health_multi", shard, required_acks=2)
    assert h2.ready is False and h2.degraded is True
    assert h2.healthy_replicas == 0
    # same underlying state, relaxed policy -> degraded but ready again
    h3 = ingest.replicator.replication_health("ns_health_multi", shard, required_acks=1)
    assert h3.ready is True and h3.degraded is True
    _shutdown_all(ingest)


# --- Test 6: primary failure ------------------------------------------------

@pytest.mark.asyncio
async def test_health_primary_failure_no_fake_replica_status(tmp_path, monkeypatch):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_primary", required_acks=2, replicas=("rep-a",))
    rid = _uid(108)
    pctx = ingest._get_or_create_ctx("ns_health_primary", shard)

    def _boom(*_a, **_k):
        raise RuntimeError("primary local apply failed")

    monkeypatch.setattr(pctx.segments, "put", _boom)
    res = await ingest.upsert("ns_health_primary", [{"id": rid, "vector": _vec(108)}])
    assert res["replication"]["attempted"] == 0
    assert res["replication"]["success"] is False

    # replication was never attempted -> replicas are NOT falsely marked failed
    h = ingest.replicator.replication_health("ns_health_primary", shard, required_acks=2)
    assert h.replicas[0].healthy is True
    assert h.replicas[0].available is True
    assert h.replicas[0].last_error is None
    assert h.degraded is False
    _shutdown_all(ingest)


# --- Test 7: health inspection is non-mutating read-only --------------------

@pytest.mark.asyncio
async def test_health_inspection_is_non_mutating(tmp_path):
    base = str(tmp_path)
    ingest, _, shard = _env(base, "ns_health_read", required_acks=1, replicas=("rep-a",))
    rid = _uid(109)
    await ingest.upsert("ns_health_read", [{"id": rid, "vector": _vec(109)}])
    ep = _replica_ep(ingest, "ns_health_read", shard, "rep-a")

    # spy on the endpoint provider: inspection must never invoke it
    real_provider = ingest.replicator.endpoint_provider
    provider_calls: list[str] = []

    def _spy_provider(ns, s, node):
        provider_calls.append(node)
        return real_provider(ns, s, node)

    ingest.replicator.endpoint_provider = _spy_provider

    before_entries = [(str(e.payload), e.timestamp) for e in ep.ctx.wal.read_all()]
    before_segments = ep.ctx.segments.count()
    before_index = ep.ctx.index.count()
    before_endpoints = dict(ingest._replica_endpoints)

    h1 = ingest.replicator.replication_health("ns_health_read", shard, required_acks=1).to_dict()
    h2 = ingest.replicator.replication_health("ns_health_read", shard, required_acks=1).to_dict()

    after_entries = [(str(e.payload), e.timestamp) for e in ep.ctx.wal.read_all()]
    assert h1 == h2                              # deterministic and idempotent
    assert after_entries == before_entries       # WAL contents unchanged
    assert ep.ctx.segments.count() == before_segments
    assert ep.ctx.index.count() == before_index
    assert dict(ingest._replica_endpoints) == before_endpoints  # no endpoints created
    assert provider_calls == []                  # endpoint_provider never invoked

    # counters and health timestamps unchanged by repeated inspection
    live = ingest.replicator._health[("ns_health_read", shard.id, "rep-a")]
    snap0 = (live.successes, live.failures, live.last_success, live.last_failure)
    for _ in range(5):
        ingest.replicator.replication_health("ns_health_read", shard, required_acks=1)
    snap1 = (live.successes, live.failures, live.last_success, live.last_failure)
    assert snap1 == snap0

    # inspection returns fresh copies: no shared mutable reference into transport state
    returned = ingest.replicator.replication_health("ns_health_read", shard, required_acks=1)
    assert returned.replicas[0] is not live

    # full representation is plain, JSON-serializable data (no objects, no reprs)
    dumped = json.dumps(returned.to_dict())
    assert "object at 0x" not in dumped
    _shutdown_all(ingest)


# --- Test 8: health consistent after replica restart/recovery ---------------

@pytest.mark.asyncio
async def test_health_after_replica_restart(tmp_path):
    base = str(tmp_path)
    ns_name = "ns_health_restart"
    rid = _uid(110)

    ingest1, _, shard1 = _env(base, ns_name, required_acks=1, replicas=("rep-a",))
    await ingest1.upsert(ns_name, [{"id": rid, "vector": _vec(110)}])
    assert ingest1.replicator.replication_health(ns_name, shard1, required_acks=1).ready is True
    _shutdown_all(ingest1)

    # fresh process over the same durable replica directory
    ingest2, _, shard2 = _env(base, ns_name, required_acks=1, replicas=("rep-a",))
    ep2 = _replica_ep(ingest2, ns_name, shard2, "rep-a")
    assert ep2.contains(rid)                     # VS-09.1 recovery re-established the record
    health = ingest2.replicator.replication_health(ns_name, shard2, required_acks=1)
    assert health.ready is True
    assert health.degraded is False
    assert health.healthy_replicas == 1
    assert health.replicas[0].last_error is None
    _shutdown_all(ingest2)


# ===========================================================================
# VS-09.4: operational observability & semantics
# ===========================================================================

# --- helpers -----------------------------------------------------------------

def _fake_shard(node_id: str = "primary-node", namespace: str = "ns_fake",
                shard_id: str = "shard-fake", replicas=()):
    return types.SimpleNamespace(node_id=node_id, namespace=namespace, id=shard_id, replicas=list(replicas))


class _NoopEndpoint:
    def apply_upsert(self, _payload):
        pass

    def apply_delete(self, _payload):
        pass


def _healthy_transport(node_ids):
    """Transport + fake shard with exactly ``node_ids`` replicas already healthy via a real delivery."""
    t = InProcessReplicaTransport(lambda _ns, _shard, _node: _NoopEndpoint())
    shard = _fake_shard(replicas=node_ids)
    if node_ids:
        t.replicate_upsert("ns_fake", shard, b"x", required_acks=1)
    return t, shard


# --- Test 9: required-ack readiness matrix -----------------------------------

@pytest.mark.parametrize("configured,required_acks,primary_available,expected_ready", [
    (0, 1, True,  True),   # primary available, 0 healthy replicas, R=1  -> ready
    (0, 2, True,  False),  # 1 (primary) + 0 healthy < 2                   -> not ready
    (1, 2, True,  True),   # 1 + 1 >= 2                                    -> ready
    (1, 3, True,  False),  # 1 + 1 < 3                                     -> not ready
    (2, 3, True,  True),   # 1 + 2 >= 3                                    -> ready
    (2, 1, False, False),  # primary unavailable                            -> never ready
])
def test_required_ack_readiness_matrix(configured, required_acks, primary_available, expected_ready):
    t, shard = _healthy_transport(["rep-a", "rep-b", "rep-c"][:configured])
    health = t.replication_health("ns_fake", shard, required_acks=required_acks,
                                  primary_available=primary_available)
    assert health.healthy_replicas == configured
    assert health.ready is expected_ready
    # degraded is independent of ready: with zero/healthy replicas there is no known failure
    assert health.degraded is False


# --- Test 10: explicit failure -> recovery sequence observability ------------

@pytest.mark.asyncio
async def test_failure_to_recovery_sequence_observability(tmp_path):
    base = str(tmp_path)
    ns = "ns_seq"
    ingest, _, shard = _env(base, ns, required_acks=2, replicas=("rep-a",))

    # initial successful delivery -> healthy
    await ingest.upsert(ns, [{"id": _uid(130), "vector": _vec(130)}])
    h0 = ingest.replicator.replication_health(ns, shard, required_acks=2)
    r0 = h0.replicas[0]
    assert h0.ready is True and h0.degraded is False
    assert r0.healthy is True
    assert r0.successes == 1 and r0.failures == 0
    assert r0.last_success is not None and r0.last_failure is None and r0.last_error is None
    last_success0 = r0.last_success

    # inject fault -> delivery fails -> unhealthy
    ingest.replicator.fail_node("rep-a")
    await ingest.upsert(ns, [{"id": _uid(131), "vector": _vec(131)}])
    h1 = ingest.replicator.replication_health(ns, shard, required_acks=2)
    r1 = h1.replicas[0]
    assert r1.healthy is False and r1.available is False
    assert r1.failures == 1 and r1.successes == 1      # failure counted only on actual failure
    assert r1.last_failure is not None
    assert r1.last_error is not None and "ReplicaUnavailableError" in r1.last_error
    assert r1.last_success == last_success0            # no fabricated success timestamp
    assert h1.ready is False and h1.degraded is True

    # repeated health inspection must not move counters/timestamps
    for _ in range(3):
        ingest.replicator.replication_health(ns, shard, required_acks=2)
    live = ingest.replicator._health[(ns, shard.id, "rep-a")]
    assert (live.failures, live.successes) == (1, 1)

    # clear fault alone -> STILL unhealthy (no fabricated recovery)
    ingest.replicator.heal_node("rep-a")
    stale = ingest.replicator.replication_health(ns, shard, required_acks=2)
    r_stale = stale.replicas[0]
    assert r_stale.healthy is False
    assert r_stale.available is True                   # fault gone, health not yet
    assert (r_stale.failures, r_stale.successes) == (1, 1)
    assert r_stale.last_success == last_success0

    # next real successful delivery -> healthy restored
    res = await ingest.upsert(ns, [{"id": _uid(132), "vector": _vec(132)}])
    assert res["replication"]["success"] is True
    h2 = ingest.replicator.replication_health(ns, shard, required_acks=2)
    r2 = h2.replicas[0]
    assert r2.healthy is True and r2.available is True
    assert (r2.failures, r2.successes) == (1, 2)       # success counted only on actual success
    assert r2.last_success is not None and r2.last_success > last_success0
    assert r2.last_error is None and r2.last_failure is not None  # error cleared, history kept
    assert h2.ready is True and h2.degraded is False
    _shutdown_all(ingest)


# --- Test 11: metrics follow the same health model ---------------------------

@pytest.mark.asyncio
async def test_metrics_follow_health_model(tmp_path):
    from vsector.infra.metrics import REPLICATION_HEALTHY_REPLICAS as G_HEALTHY
    from vsector.infra.metrics import REPLICATION_READY as G_READY

    base = str(tmp_path)
    ns = "ns_metrics"
    ns2 = "ns_metrics_required"
    ingest, _, shard = _env(base, ns, required_acks=1, replicas=("rep-a",))
    hg = G_HEALTHY.labels(namespace=ns, shard_id=shard.id)
    rg = G_READY.labels(namespace=ns, shard_id=shard.id)

    # all healthy (required_acks=1)
    await ingest.upsert(ns, [{"id": _uid(140), "vector": _vec(140)}])
    ingest.stats()                                     # gauges refresh from replication_health()
    assert hg._value.get() == 1.0
    assert rg._value.get() == 1.0

    # degraded-but-ready: required_acks=1, replica down (primary ack still suffices)
    ingest.replicator.fail_node("rep-a")
    await ingest.upsert(ns, [{"id": _uid(141), "vector": _vec(141)}])
    st = ingest.stats()
    assert hg._value.get() == 0.0
    assert rg._value.get() == 1.0                      # degraded does NOT force not-ready
    assert st[f"{ns}:{shard.id}"]["replication_health"]["ready"] is True

    # not-ready: required_acks=2, replica down -> 1 + 0 < 2
    ingest2, _, shard2 = _env(base, ns2, required_acks=2, replicas=("rep-a",))
    hg2 = G_HEALTHY.labels(namespace=ns2, shard_id=shard2.id)
    rg2 = G_READY.labels(namespace=ns2, shard_id=shard2.id)
    ingest2.replicator.fail_node("rep-a")
    await ingest2.upsert(ns2, [{"id": _uid(142), "vector": _vec(142)}])
    ingest2.stats()
    assert hg2._value.get() == 0.0
    assert rg2._value.get() == 0.0

    # recovery after a real successful delivery
    ingest2.replicator.heal_node("rep-a")
    await ingest2.upsert(ns2, [{"id": _uid(143), "vector": _vec(143)}])
    st2 = ingest2.stats()
    assert hg2._value.get() == 1.0
    assert rg2._value.get() == 1.0

    # stats().replication_health and gauges must derive from the same logical state
    rh = st2[f"{ns2}:{shard2.id}"]["replication_health"]
    assert rh["healthy_replicas"] == int(hg2._value.get())
    assert rh["ready"] == bool(rg2._value.get())
    _shutdown_all(ingest)
    _shutdown_all(ingest2)


# --- Test 12: multi-shard health representation stays deterministic ----------

@pytest.mark.asyncio
async def test_multi_shard_health_response_deterministic(tmp_path):
    base = str(tmp_path)
    ns_name = "ns_multi_shard"
    ingest, ns, shard1 = _env(base, ns_name, required_acks=1, replicas=("rep-a",))
    shard2 = Shard(namespace=ns_name, node_id="primary-node", id="shard-repl-2", replicas=["rep-b"])
    ingest.router.register_shard(shard2)
    ctx1 = ingest._get_or_create_ctx(ns_name, shard1)
    ctx2 = ingest._get_or_create_ctx(ns_name, shard2)

    # one shard healthy, one degraded (rep-b faulted -> unavailable)
    await ingest.upsert(ns_name, [{"id": _uid(150), "vector": _vec(150)}])
    ingest.replicator.fail_node("rep-b")
    await ingest.upsert(ns_name, [{"id": _uid(151), "vector": _vec(151)}])

    hm = ingest._replication_health_map(ns_name, ns, {"a": ctx1, "b": ctx2})
    assert isinstance(hm, dict) and "shards" in hm
    shards = hm["shards"]
    assert len(shards) == 2
    assert set(shards) == {f"{ns_name}:shard-repl-1", f"{ns_name}:shard-repl-2"}

    # per-shard health remains inspectable; each reflects its own replica state
    s1 = shards[f"{ns_name}:shard-repl-1"]
    s2 = shards[f"{ns_name}:shard-repl-2"]
    assert s1["degraded"] is True or s2["degraded"] is True  # at least the faulted shard degraded
    assert s1["ready"] is True and s2["ready"] is True       # required_acks=1 keeps both ready

    # same underlying state, stricter policy -> degraded shard no longer ready
    t = ingest.replicator
    s2_strict = t.replication_health(ns_name, shard2, required_acks=3)
    assert s2_strict.ready is False and s2_strict.degraded is True

    # deterministic across repeated reads
    assert ingest._replication_health_map(ns_name, ns, {"a": ctx1, "b": ctx2}) == hm

    # single-shard map collapses to the object (backward-compatible API shape)
    single = ingest._replication_health_map(ns_name, ns, {"a": ctx1})
    assert "shards" not in single
    assert "replicas" in single and "ready" in single and "degraded" in single
    _shutdown_all(ingest)