"""VS-10: shard lifecycle & ownership semantics tests.

Covers:
- lifecycle transitions (CREATING -> ACTIVE -> DRAINING -> OFFLINE, plus the
  ownership-change path DRAINING -> ACTIVE) and rejection of arbitrary jumps;
- ownership invariants (one primary owner, replicas distinct, stable identity,
  conflicting ownership rejected);
- ownership/lifecycle-aware routing (active-drain-offline behavior, no silent
  forwarding to a non-primary owner);
- truthful restart behavior via the durable shard store (identity / lifecycle /
  ownership survive; nothing is fabricated ACTIVE);
- replication integration (ownership and VS-09 health stay independent);
- stats/API observability (deterministic, JSON-serializable, fresh copies).
"""
from __future__ import annotations

import json
import uuid

import pytest

from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard, ShardState
from vsector.sharding.lifecycle import ShardLifecycleManager
from vsector.sharding.exceptions import (
    InvalidLifecycleTransitionError,
    NoPrimaryOwnerError,
    OwnershipConflictError,
    OwnershipMismatchError,
    ShardCreatingError,
    ShardDrainingError,
    ShardOfflineError,
)
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService

SHARD_ID = "shard-lifecycle-1"
OWNER = "node-0"


def _env(base_dir: str, ns_name: str = "ns10", required_acks: int = 1,
         node_id: str = OWNER, owner: str = OWNER, replicas: tuple[str, ...] = ("rep-a",),
         shard_id: str = SHARD_ID, state: ShardState = ShardState.ACTIVE):
    md = MetadataStore(path=f"{base_dir}/metadata.json")
    etcd = EtcdStore()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir=base_dir, node_id=node_id)
    ns = md.get(ns_name)
    if ns is None:
        ns = Namespace(name=ns_name, dimension=4, index_type=IndexType.FLAT,
                       distance_metric=DistanceMetric.COSINE, required_acks=required_acks)
        md.create(ns)
    shard = Shard(namespace=ns_name, node_id=owner, id=shard_id, replicas=list(replicas), state=state)
    if not router.etcd.list_by_namespace(ns_name):
        router.register_shard(shard)
    return ingest, router, shard, ns, md


def _shutdown(ingest: IngestService) -> None:
    for ctx in ingest._shards.values():
        try:
            ctx.wal.close()
        except Exception:
            pass
    for ep in ingest._replica_endpoints.values():
        try:
            ep.close()
        except Exception:
            pass


def _uid(i: int) -> str:
    return str(uuid.UUID(int=i))


# ===========================================================================
# 1. Lifecycle transitions
# ===========================================================================


def test_lifecycle_valid_sequence(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    mgr = ShardLifecycleManager(EtcdStore())
    mgr.create(shard, owner=OWNER, replicas=["rep-a"])
    assert shard.state is ShardState.CREATING

    mgr.activate(shard)
    assert shard.state is ShardState.ACTIVE

    mgr.begin_drain(shard)
    assert shard.state is ShardState.DRAINING

    mgr.retire(shard)
    assert shard.state is ShardState.OFFLINE


def test_lifecycle_ownership_change_sequence(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    mgr = ShardLifecycleManager(EtcdStore())
    mgr.create(shard, owner=OWNER, replicas=["rep-a"])
    mgr.activate(shard)

    # ACTIVE -> DRAINING -> ownership change -> ACTIVE
    mgr.begin_drain(shard)
    mgr.change_owner(shard, "node-1")
    assert shard.state is ShardState.DRAINING        # change is stateful, no auto-handoff
    assert shard.primary_owner == "node-1"
    mgr.activate(shard)
    assert shard.state is ShardState.ACTIVE
    assert shard.primary_owner == "node-1"


@pytest.mark.parametrize("start,target", [
    (ShardState.CREATING, ShardState.DRAINING),
    (ShardState.CREATING, ShardState.RETIRED),
    (ShardState.ACTIVE, ShardState.CREATING),
    (ShardState.ACTIVE, ShardState.ACTIVE),
    (ShardState.DRAINING, ShardState.CREATING),
    (ShardState.OFFLINE, ShardState.ACTIVE),
    (ShardState.OFFLINE, ShardState.DRAINING),
    (ShardState.OFFLINE, ShardState.CREATING),
])
def test_lifecycle_invalid_transitions_rejected(tmp_path, start, target):
    _, _, shard, _, _ = _env(str(tmp_path), state=start)
    with pytest.raises(InvalidLifecycleTransitionError) as ei:
        shard.transition(target)
    # deterministic message carries namespace, shard id, current and target state
    msg = str(ei.value)
    assert f"{shard.namespace}:{shard.id}" in msg
    assert start.value in msg and target.value in msg
    # no silent jump: state is unchanged after a rejected transition
    assert shard.state is start


def test_lifecycle_abort_paths_are_valid(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    mgr = ShardLifecycleManager(EtcdStore())
    # CREATING -> OFFLINE (abort before activation) is an explicit allowed path
    s1 = shard
    mgr.create(s1, owner=OWNER, replicas=["rep-a"])
    mgr.create  # no-op reference keep
    s2 = Shard(namespace=shard.namespace, node_id=OWNER, id=SHARD_ID + "-2")
    mgr.create(s2, owner=OWNER, replicas=["rep-a"])
    assert s2.transition(ShardState.OFFLINE).state is ShardState.OFFLINE
    # ACTIVE -> OFFLINE (decommission) is an explicit allowed path
    s3 = Shard(namespace=shard.namespace, node_id=OWNER, id=SHARD_ID + "-3")
    mgr.create(s3, owner=OWNER, replicas=["rep-a"])
    mgr.activate(s3)
    assert s3.transition(ShardState.OFFLINE).state is ShardState.OFFLINE
    del shard  # s1 keeps the CREATING state check simple


@pytest.mark.asyncio
async def test_lifecycle_aware_writes_error_deterministically(tmp_path):
    ingest, _, shard, _, _ = _env(str(tmp_path))
    mgr = ShardLifecycleManager(ingest.router.etcd)

    # CREATING: writes refused with a deterministic error
    mgr.create(shard, owner=OWNER, replicas=["rep-a"])
    assert shard.state is ShardState.CREATING
    r = await ingest.upsert("ns10", [{"id": _uid(1), "vector": [1.0, 0.0, 0.0, 0.0]}])
    assert r["upserted"] == 0
    assert any("CREATING" in e["error"] for e in r["errors"])

    # ACTIVE: writes accepted
    mgr.activate(shard)
    r = await ingest.upsert("ns10", [{"id": _uid(2), "vector": [1.0, 0.0, 0.0, 0.0]}])
    assert r["upserted"] == 1

    # DRAINING: new writes rejected with a deterministic error
    mgr.begin_drain(shard)
    r = await ingest.upsert("ns10", [{"id": _uid(3), "vector": [1.0, 0.0, 0.0, 0.0]}])
    assert r["upserted"] == 0
    assert any("shard draining" in e["error"] and "DRAINING" in e["error"] for e in r["errors"])

    # OFFLINE: writes refused and the shard is excluded from query fan-out
    mgr.retire(shard)
    r = await ingest.upsert("ns10", [{"id": _uid(4), "vector": [1.0, 0.0, 0.0, 0.0]}])
    assert r["upserted"] == 0
    assert any("shard offline" in e["error"] for e in r["errors"])
    assert ingest.router.route_for_query("ns10") == []
    _shutdown(ingest)


# ===========================================================================
# 2. Ownership
# ===========================================================================


def test_active_shard_has_exactly_one_primary_owner(tmp_path):
    _, router, shard, _, _ = _env(str(tmp_path))
    # ACTIVE + owned -> routing resolves to the single primary owner
    picked = router.route_write("ns10", "record-1", OWNER)
    assert picked.id == shard.id
    assert picked.primary_owner == OWNER
    own = shard.ownership()
    assert isinstance(own["primary_owner"], str) and own["primary_owner"] == OWNER


def test_replica_membership_distinct_from_primary(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    mgr = ShardLifecycleManager(EtcdStore())
    with pytest.raises(OwnershipConflictError):
        mgr.create(shard, owner=OWNER, replicas=[OWNER])  # primary cannot be its own replica


def test_ownership_identity_stable_and_fresh_copies(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    own1 = shard.ownership()
    own2 = shard.ownership()
    assert own1 == own2                             # deterministic across reads
    # mutation of a returned snapshot never leaks into the shard
    own1["replicas"].append("hacker")
    own1["primary_owner"] = "hacker"
    assert shard.ownership()["replicas"] == ["rep-a"]
    assert shard.ownership()["primary_owner"] == OWNER
    # to_dict also returns a safe, fresh replica list
    d = shard.to_dict()
    d["replicas"].append("hacker")
    assert shard.to_dict()["replicas"] == ["rep-a"]


def test_conflicting_ownership_changes_rejected(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    mgr = ShardLifecycleManager(EtcdStore())
    # ownership cannot change while silently ACTIVE
    with pytest.raises(InvalidLifecycleTransitionError):
        mgr.change_owner(shard, "node-2")
    assert shard.primary_owner == OWNER
    # no empty primary owner
    mgr.begin_drain(shard)
    with pytest.raises(NoPrimaryOwnerError):
        mgr.change_owner(shard, "")
    # new primary cannot sit inside the new replica list
    with pytest.raises(OwnershipConflictError):
        mgr.change_owner(shard, "rep-a")


# ===========================================================================
# 3. Routing
# ===========================================================================


def test_routing_resolves_to_current_primary(tmp_path):
    _, router, shard, _, _ = _env(str(tmp_path))
    shard2 = Shard(namespace="ns10", node_id=OWNER, id="shard-lifecycle-2")
    router.register_shard(shard2)
    for i in range(50):
        # a locally-owned multi-shard namespace resolves every record to its primary
        picked = router.route_write("ns10", f"record-{i}", OWNER)
        assert picked.id in (shard.id, shard2.id)
        assert picked.primary_owner == OWNER           # the explicit current primary owner
        # never silently routed to the replica set
        assert picked.primary_owner not in (picked.replicas)
    # a shard owned by a different node is refused for this node (no fake routing)
    shard3 = Shard(namespace="ns10", node_id="node-2", id="shard-lifecycle-3")
    router.register_shard(shard3)
    # every record that happens to land on shard3 must be refused deterministically
    refused = 0
    for i in range(200):
        try:
            router.route_write("ns10", f"other-{i}", OWNER)
        except OwnershipMismatchError:
            refused += 1
    assert refused > 0
    assert any(s.id == "shard-lifecycle-3" for s in router.route_for_query("ns10"))


def test_routing_draining_is_deterministic(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    router = ShardRouter(EtcdStore())
    router.register_shard(shard)
    mgr = ShardLifecycleManager(router.etcd)
    mgr.begin_drain(shard)
    with pytest.raises(ShardDrainingError) as ei:
        router.route_write("ns10", "record-x", OWNER)
    assert f"{shard.namespace}:{shard.id}" in str(ei.value)


def test_routing_offline_is_unavailable(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path), state=ShardState.OFFLINE)
    router = ShardRouter(EtcdStore())
    router.register_shard(shard)
    with pytest.raises(ShardOfflineError):
        router.route_write("ns10", "record-x", OWNER)
    with pytest.raises(ShardOfflineError):
        router.route("ns10", "record-x")                     # reads do not fake availability
    assert router.route_for_query("ns10") == []              # excluded from query fan-out


def test_routing_creating_is_not_writable(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path), state=ShardState.CREATING)
    router = ShardRouter(EtcdStore())
    router.register_shard(shard)
    with pytest.raises(ShardCreatingError):
        router.route_write("ns10", "record-x", OWNER)


def test_routing_no_primary_owner_is_deterministic(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path), owner="")
    router = ShardRouter(EtcdStore())
    router.register_shard(shard)
    with pytest.raises(NoPrimaryOwnerError):
        router.route_write("ns10", "record-x", OWNER)


@pytest.mark.asyncio
async def test_routing_refuses_non_primary_owner_without_write(tmp_path):
    ingest, router, shard, _, _ = _env(str(tmp_path), node_id="node-0", owner="node-1")
    with pytest.raises(OwnershipMismatchError) as ei:
        router.route_write("ns10", "record-x", "node-0")
    msg = str(ei.value)
    assert "node-1" in msg and "node-0" in msg              # owner + local node, deterministic
    assert "refused" in msg

    # full service path: the write is refused and nothing is persisted anywhere
    r = await ingest.upsert("ns10", [{"id": _uid(1), "vector": [1.0, 0.0, 0.0, 0.0]}])
    assert r["upserted"] == 0
    assert any("ownership mismatch" in e["error"] for e in r["errors"])
    assert shard.vector_count == 0
    _shutdown(ingest)


def test_routing_identity_stable_across_lifecycle(tmp_path):
    _, router, shard, _, _ = _env(str(tmp_path))
    active = router.route_write("ns10", "record-x", OWNER)
    assert active.id == shard.id
    ShardLifecycleManager(router.etcd).begin_drain(shard)
    with pytest.raises(ShardDrainingError) as ei:
        router.route_write("ns10", "record-x", OWNER)
    # the same shard is still the target — the record did not silently move
    assert shard.id in str(ei.value)
    assert active.id == shard.id


# ===========================================================================
# 4. Recovery / restart (durable shard store)
# ===========================================================================


def test_shard_identity_and_lifecycle_survive_restart(tmp_path):
    path = str(tmp_path / "shards.json")
    store1 = EtcdStore(path=path)
    mgr1 = ShardLifecycleManager(store1)
    s1 = Shard(namespace="ns_persist", node_id=OWNER, id="persist-shard-1", replicas=["rep-a"])
    mgr1.create(s1, owner=OWNER, replicas=["rep-a"])
    mgr1.activate(s1)
    mgr1.begin_drain(s1)
    assert s1.state is ShardState.DRAINING

    # restart: fresh store over the same durable file
    store2 = EtcdStore(path=path)
    loaded = store2.get("persist-shard-1")
    assert loaded is not None
    assert loaded.id == "persist-shard-1"               # identity survives (no new uuid)
    assert loaded.namespace == "ns_persist"
    assert loaded.primary_owner == OWNER                # owner survives
    assert loaded.state is ShardState.DRAINING          # nothing fabricated ACTIVE after restart


def test_offline_state_not_falsely_upgraded(tmp_path):
    path = str(tmp_path / "shards.json")
    store1 = EtcdStore(path=path)
    mgr = ShardLifecycleManager(store1)
    offline = Shard(namespace="ns_persist2", node_id=OWNER, id="offline-shard-1")
    mgr.create(offline, owner=OWNER)
    mgr.activate(offline)
    mgr.retire(offline)
    assert offline.state is ShardState.OFFLINE

    store2 = EtcdStore(path=path)
    loaded = store2.get("offline-shard-1")
    assert loaded.state is ShardState.OFFLINE           # OFFLINE stays OFFLINE (no false ACTIVE)


def test_ownership_transfer_survives_restart(tmp_path):
    path = str(tmp_path / "shards.json")
    store1 = EtcdStore(path=path)
    mgr1 = ShardLifecycleManager(store1)
    s = Shard(namespace="ns_persist3", node_id=OWNER, id="owner-shard-1", replicas=["rep-a"])
    mgr1.create(s, owner=OWNER, replicas=["rep-a"])
    mgr1.activate(s)
    mgr1.begin_drain(s)
    mgr1.change_owner(s, "node-1", replicas=["node-2"])

    store2 = EtcdStore(path=path)
    loaded = store2.get("owner-shard-1")
    assert loaded.state is ShardState.DRAINING
    assert loaded.primary_owner == "node-1"
    assert loaded.replicas == ["node-2"]
    # the new owner is still valid (primary not in replicas)
    loaded.validate_ownership()


# ===========================================================================
# 5. Replication interaction (ownership and health stay independent)
# ===========================================================================

@pytest.mark.asyncio
async def test_ownership_and_health_are_separate(tmp_path, monkeypatch):
    ingest, _, shard, _, _ = _env(str(tmp_path), required_acks=2, replicas=("rep-a",))

    # healthy primary: ownership intact, health ready
    await ingest.upsert("ns10", [{"id": _uid(100), "vector": [1.0, 0.0, 0.0, 0.0]}])
    h = ingest.replicator.replication_health("ns10", shard, required_acks=2)
    assert h.ready is True and h.degraded is False
    assert shard.primary_owner == OWNER

    # unhealthy primary (local apply fails): ownership does NOT move; the healthy
    # replica is NOT auto-promoted; VS-09 health semantics unchanged
    pctx = ingest._get_or_create_ctx("ns10", shard)
    monkeypatch.setattr(pctx.segments, "put", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("primary failed")))
    r = await ingest.upsert("ns10", [{"id": _uid(101), "vector": [1.0, 0.0, 0.0, 0.0]}])
    assert r["replication"]["success"] is False
    assert r["replication"]["attempted"] == 0          # no replication from a failed primary
    assert shard.primary_owner == OWNER                # no fabricated ownership transfer
    assert shard.state is ShardState.ACTIVE
    assert shard.ownership()["primary_owner"] == OWNER
    _shutdown(ingest)


@pytest.mark.asyncio
async def test_healthy_replica_does_not_become_primary(tmp_path):
    ingest, _, shard, _, _ = _env(str(tmp_path), required_acks=1, replicas=("rep-a",))
    await ingest.upsert("ns10", [{"id": _uid(110), "vector": [1.0, 0.0, 0.0, 0.0]}])
    h = ingest.replicator.replication_health("ns10", shard, required_acks=1)
    assert h.replicas[0].healthy is True               # replica healthy...
    assert shard.primary_owner == OWNER                # ...but still not the primary
    assert h.primary == OWNER
    # readiness policy unchanged: 1 primary + 1 healthy replica satisfies R=1 and R=2
    assert ingest.replicator.replication_health("ns10", shard, required_acks=2).ready is True
    _shutdown(ingest)


# ===========================================================================
# 6. API / stats observability
# ===========================================================================

@pytest.mark.asyncio
async def test_stats_expose_lifecycle_and_ownership(tmp_path):
    ingest, _, shard, _, _ = _env(str(tmp_path))
    await ingest.upsert("ns10", [{"id": _uid(120), "vector": [1.0, 0.0, 0.0, 0.0]}])
    st = ingest.stats()
    key = f"ns10:{SHARD_ID}"
    assert key in st
    entry = st[key]
    assert entry["shard_state"] == "ACTIVE"
    assert entry["primary_owner"] == OWNER
    assert entry["replicas"] == ["rep-a"]
    # existing compatibility fields are untouched
    assert "state" in entry and "backend_name" in entry
    assert "replication" in entry and "replication_health" in entry
    # deterministic + JSON serializable + fresh copies
    assert st == ingest.stats()
    dumped = json.dumps(st, sort_keys=True)
    assert "object at 0x" not in dumped
    entry["replicas"].append("hacker")
    assert ingest.stats()[key]["replicas"] == ["rep-a"]
    _shutdown(ingest)


def test_namespace_stats_to_dict_is_deterministic_and_safe(tmp_path):
    _, _, shard, _, _ = _env(str(tmp_path))
    d1 = shard.to_dict()
    d2 = shard.to_dict()
    assert d1 == d2
    assert d1["primary_owner"] == OWNER
    assert d1["state"] == "ACTIVE"
    assert json.loads(json.dumps(d1)) == d1
    # mutating one to_dict must not corrupt the shard or a later snapshot
    d1["replicas"].append("x")
    assert shard.to_dict()["replicas"] == ["rep-a"]
    # the API (namespace stats) exposes the ownership/lifecycle fields
    assert d2["primary_owner"] == OWNER
    assert d2["state"] == "ACTIVE"
    assert d2["replicas"] == ["rep-a"]