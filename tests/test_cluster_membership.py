"""VS-11: cluster membership & shard ownership integration tests.

Covers:
- cluster identity (stable, persistent, distinct from node_id and shard_id);
- node membership lifecycle (register/activate/drain/remove; valid + invalid
  transitions; duplicate registration; unknown/removed nodes);
- ownership validated against membership (unknown owner rejected, removed owner
  rejected, known owner accepted, VS-10 lifecycle rules preserved);
- replica membership validation (known members pass, unknown/removed rejected,
  no automatic promotion);
- routing (VS-10 routing preserved; unknown owner deterministic failure; no
  cross-node forwarding);
- lifecycle independence (cluster membership never auto-changes shard state);
- truthful restart (cluster + shard metadata survive consistently; nothing is
  fabricated ACTIVE; no automatic ownership transfer).
"""
from __future__ import annotations

import pytest

from vsector.infra.config import get_settings
from vsector.cluster import (
    NodeMembershipState,
    ClusterNode,
    ClusterMembershipManager,
    bootstrap_local_node,
    UnknownNodeError,
    DuplicateNodeRegistrationError,
    RemovedNodeError,
    InvalidMembershipTransitionError,
    ClusterIdMismatchError,
)
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard, ShardState
from vsector.sharding.lifecycle import ShardLifecycleManager
from vsector.sharding.exceptions import InvalidLifecycleTransitionError, OwnershipMismatchError

CLUSTER_ID = "cluster-a"
LOCAL = "node-0"
OTHER = "node-1"


def _manager(store, cluster_id: str = CLUSTER_ID) -> ClusterMembershipManager:
    return ClusterMembershipManager(store, cluster_id=cluster_id)


def _durable(path) -> EtcdStore:
    return EtcdStore(path=str(path))


def _router(store, membership=None) -> ShardRouter:
    return ShardRouter(store, membership=membership)


def _register_active(membership: ClusterMembershipManager, node_id: str) -> None:
    membership.register(node_id)
    membership.activate(node_id)


# ===========================================================================
# 1. Cluster identity
# ===========================================================================


def test_cluster_id_is_stable_and_persistable(tmp_path):
    store1 = _durable(tmp_path / "shards.json")
    mgr1 = _manager(store1)
    _register_active(mgr1, LOCAL)

    # "restart": a fresh manager over the same durable file keeps the identity
    store2 = _durable(tmp_path / "shards.json")
    mgr2 = _manager(store2, cluster_id="cluster-a")
    assert mgr2.cluster_id == CLUSTER_ID
    assert mgr2.get(LOCAL).cluster_id == CLUSTER_ID  # node identity also stable


def test_default_cluster_id_deterministic():
    s = get_settings()
    assert s.cluster_id == "vsector-cluster-default"  # pinned default, never regenerated


def test_identity_separation():
    """cluster_id, node_id, shard_id are three different identities."""
    s = get_settings()
    assert s.cluster_id != s.node_id
    shard = Shard(namespace="ns", node_id=s.node_id, id="sh-1")
    assert shard.id != s.node_id
    assert shard.id != s.cluster_id


# ===========================================================================
# 2. Node membership lifecycle
# ===========================================================================


def test_register_join_active_drain_sequence(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    n = mgr.register(LOCAL)
    assert n.state is NodeMembershipState.JOINING
    n = mgr.activate(LOCAL)
    assert n.state is NodeMembershipState.ACTIVE
    n = mgr.begin_drain(LOCAL)
    assert n.state is NodeMembershipState.DRAINING
    n = mgr.activate(LOCAL)  # DRAINING -> ACTIVE
    assert n.state is NodeMembershipState.ACTIVE
    n = mgr.begin_drain(LOCAL)
    n = mgr.remove(LOCAL)
    assert n.state is NodeMembershipState.REMOVED
    assert n.version == 6  # register + 5 validated transitions


@pytest.mark.parametrize("start,target", [
    (NodeMembershipState.JOINING, NodeMembershipState.DRAINING),
    (NodeMembershipState.ACTIVE, NodeMembershipState.JOINING),
    (NodeMembershipState.DRAINING, NodeMembershipState.DRAINING),
    (NodeMembershipState.REMOVED, NodeMembershipState.ACTIVE),  # terminal
    (NodeMembershipState.REMOVED, NodeMembershipState.JOINING),
    (NodeMembershipState.ACTIVE, NodeMembershipState.ACTIVE),
])
def test_invalid_membership_transitions(tmp_path, start, target):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    node = ClusterNode(node_id=LOCAL, cluster_id=CLUSTER_ID, state=start)
    mgr._store.put_node(node)
    with pytest.raises(InvalidMembershipTransitionError):
        mgr._transition(LOCAL, target)
    # state is silently unchanged
    assert mgr.get(LOCAL).state is start


def test_duplicate_registration_rejected(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    mgr.register(LOCAL)
    with pytest.raises(DuplicateNodeRegistrationError):
        mgr.register(LOCAL)
    assert mgr.get(LOCAL).state is NodeMembershipState.JOINING  # untouched


def test_unknown_node_operations_fail(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    with pytest.raises(UnknownNodeError):
        mgr.activate("ghost")
    with pytest.raises(UnknownNodeError):
        mgr.begin_drain("ghost")
    with pytest.raises(UnknownNodeError):
        mgr.remove("ghost")
    with pytest.raises(UnknownNodeError):
        mgr.validate_node("ghost")
    assert mgr.get("ghost") is None
    assert mgr.contains("ghost") is False
    assert mgr.is_available("ghost") is False
    with pytest.raises(ValueError):  # all cluster errors subclass ValueError (REST 400 mapping)
        mgr.activate("ghost")


def test_removed_node_validate_rejected(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    _register_active(mgr, LOCAL)
    mgr.remove(LOCAL)
    with pytest.raises(RemovedNodeError):
        mgr.validate_node(LOCAL)
    assert mgr.contains(LOCAL) is True   # still *known*, just removed
    assert mgr.is_available(LOCAL) is False


def test_membership_state_and_availability():
    store = EtcdStore()
    mgr = _manager(store)
    mgr.register(LOCAL)
    assert mgr.membership_state(LOCAL) == "JOINING"
    assert mgr.is_available(LOCAL) is False
    mgr.activate(LOCAL)
    assert mgr.membership_state(LOCAL) == "ACTIVE"
    assert mgr.is_available(LOCAL) is True
    mgr.begin_drain(LOCAL)
    assert mgr.is_available(LOCAL) is True  # DRAINING still serves
    mgr.remove(LOCAL)
    assert mgr.is_available(LOCAL) is False


def test_fresh_copies_no_mutable_state_leak(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    _register_active(mgr, LOCAL)
    a = mgr.get(LOCAL)
    b = mgr.get(LOCAL)
    assert a is not b                     # fresh copies each time
    a.state = NodeMembershipState.REMOVED  # mutating a copy must not affect the store
    assert mgr.get(LOCAL).state is NodeMembershipState.ACTIVE


def test_cluster_id_mismatch_rejected(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr_a = _manager(store, cluster_id="cluster-a")
    _register_active(mgr_a, LOCAL)

    mgr_b = _manager(store, cluster_id="cluster-b")  # same store, different identity
    with pytest.raises(ClusterIdMismatchError):
        mgr_b.validate_cluster_id(LOCAL)
    with pytest.raises(ClusterIdMismatchError):
        mgr_b.activate(LOCAL)
    with pytest.raises(ClusterIdMismatchError):
        mgr_a.register("node-x", cluster_id="wrong-cluster")
    assert mgr_a.get(LOCAL).state is NodeMembershipState.ACTIVE  # unchanged


# ===========================================================================
# 3. Persistence / restart
# ===========================================================================


def test_membership_survives_restart(tmp_path):
    path = tmp_path / "shards.json"
    mgr1 = _manager(_durable(path))
    _register_active(mgr1, LOCAL)
    _register_active(mgr1, OTHER)
    mgr1.begin_drain(OTHER)

    mgr2 = _manager(_durable(path))
    assert mgr2.cluster_id == CLUSTER_ID
    assert mgr2.get(LOCAL).state is NodeMembershipState.ACTIVE
    assert mgr2.get(OTHER).state is NodeMembershipState.DRAINING  # truthful, not ACTIVE
    assert mgr2.get(LOCAL).version == 2
    assert mgr2.get(OTHER).version == 3


def test_bootstrap_local_node_fresh_join_then_active(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    node = bootstrap_local_node(mgr, LOCAL)
    assert node.state is NodeMembershipState.ACTIVE
    assert mgr.membership_state(LOCAL) == "ACTIVE"
    assert mgr.contains(OTHER) is False  # no membership fabricated for other nodes


def test_bootstrap_restores_persisted_active(tmp_path):
    path = tmp_path / "shards.json"
    mgr1 = _manager(_durable(path))
    _register_active(mgr1, LOCAL)
    mgr2 = _manager(_durable(path))
    node = bootstrap_local_node(mgr2, LOCAL)
    assert node.state is NodeMembershipState.ACTIVE
    assert mgr2.list() == [mgr2.get(LOCAL)]  # exactly one record, no re-registration


def test_bootstrap_never_fabricates_active_for_draining(tmp_path):
    path = tmp_path / "shards.json"
    mgr1 = _manager(_durable(path))
    _register_active(mgr1, LOCAL)
    mgr1.begin_drain(LOCAL)
    mgr2 = _manager(_durable(path))
    assert bootstrap_local_node(mgr2, LOCAL).state is NodeMembershipState.DRAINING


def test_bootstrap_never_fabricates_active_for_removed(tmp_path):
    path = tmp_path / "shards.json"
    mgr1 = _manager(_durable(path))
    _register_active(mgr1, LOCAL)
    mgr1.remove(LOCAL)
    mgr2 = _manager(_durable(path))
    assert bootstrap_local_node(mgr2, LOCAL).state is NodeMembershipState.REMOVED


# ===========================================================================
# 4. Ownership validation against membership
# ===========================================================================


def _owned_router(tmp_path, owner: str):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    shard = Shard(namespace="ns_cl", node_id=owner, id="cl-shard-1", replicas=["node-r"])
    router = _router(store, membership=mgr)
    router.register_shard(shard)
    return router, mgr, shard


def test_router_rejects_unknown_owner_deterministic(tmp_path):
    router, mgr, shard = _owned_router(tmp_path, "ghost-owner")
    _register_active(mgr, LOCAL)  # only the local node is a member
    with pytest.raises(UnknownNodeError):
        router.route_write("ns_cl", "record-1", LOCAL)
    with pytest.raises(UnknownNodeError):
        router.route("ns_cl", "record-1")
    assert shard.state is ShardState.ACTIVE  # shard untouched, nothing forwarded


def test_router_rejects_removed_owner(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, "gone")
    mgr.remove("gone")
    shard = Shard(namespace="ns_cl", node_id="gone", id="cl-shard-r")
    router = _router(store, membership=mgr)
    router.register_shard(shard)
    with pytest.raises(RemovedNodeError):
        router.route_write("ns_cl", "record-1", LOCAL)


def test_router_accepts_known_owner(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-ok")
    router = _router(store, membership=mgr)
    router.register_shard(shard)
    picked = router.route_write("ns_cl", "record-1", LOCAL)
    assert picked.primary_owner == LOCAL


def test_router_without_membership_preserved(tmp_path):
    # VS-10 routing semantics are unchanged when no membership is wired
    store = EtcdStore()
    router = _router(store)  # no membership
    shard = Shard(namespace="ns_cl", node_id="node-9", id="cl-shard-legacy")
    router.register_shard(shard)
    picked = router.route_write("ns_cl", "record-1", "node-9")
    assert picked.primary_owner == "node-9"
    # mismatch still deterministic (VS-10)
    with pytest.raises(OwnershipMismatchError):
        router.route_write("ns_cl", "record-1", "node-0")


def test_lifecycle_create_validates_owner_known(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    lc = ShardLifecycleManager(store, membership=mgr)
    ok = Shard(namespace="ns_cl", node_id=LOCAL)
    lc.create(ok, owner=LOCAL, replicas=["node-r"])
    assert ok.state is ShardState.CREATING
    with pytest.raises(UnknownNodeError):
        lc.create(Shard(namespace="ns_cl", node_id="ghost"), owner="ghost", replicas=[])
    assert not store.list_by_namespace("ns_cl") or all(s.node_id != "ghost" for s in store.list_by_namespace("ns_cl"))


def test_change_owner_validates_new_owner_known(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    _register_active(mgr, OTHER)
    lc = ShardLifecycleManager(store, membership=mgr)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-co")
    lc.create(shard, owner=LOCAL, replicas=["node-x"])
    lc.activate(shard)
    lc.begin_drain(shard)
    lc.change_owner(shard, OTHER)  # known member
    assert shard.primary_owner == OTHER
    assert shard.state is ShardState.DRAINING
    with pytest.raises(UnknownNodeError):
        lc.change_owner(shard, "ghost")
    assert shard.primary_owner == OTHER  # unchanged on rejection


def test_change_owner_still_requires_draining(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    _register_active(mgr, OTHER)
    lc = ShardLifecycleManager(store, membership=mgr)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-drain")
    lc.create(shard, owner=LOCAL, replicas=[])
    lc.activate(shard)
    with pytest.raises(InvalidLifecycleTransitionError):  # VS-10 rule preserved
        lc.change_owner(shard, OTHER)  # must be DRAINING first
    assert shard.primary_owner == LOCAL


# ===========================================================================
# 5. Replica membership
# ===========================================================================


def test_validate_replicas_known_members(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    _register_active(mgr, "node-r1")
    _register_active(mgr, "node-r2")
    mgr.validate_replicas(["node-r1", "node-r2"])  # no raise


def test_validate_replicas_rejects_unknown_and_removed(tmp_path):
    mgr = _manager(_durable(tmp_path / "shards.json"))
    _register_active(mgr, "node-r1")
    mgr.remove("node-r1")
    with pytest.raises(RemovedNodeError):
        mgr.validate_replicas(["node-r1"])
    with pytest.raises(UnknownNodeError):
        mgr.validate_replicas(["ghost"])
    assert mgr.validate_replicas([]) is None  # no replicas is trivially valid


def test_no_automatic_promotion(tmp_path):
    # membership draining does NOT promote anyone: ownership stays put
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    _register_active(mgr, OTHER)
    mgr.begin_drain(LOCAL)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-nap")
    router = _router(store, membership=mgr)
    router.register_shard(shard)
    picked = router.route_write("ns_cl", "record-1", LOCAL)  # draining member still owns
    assert picked.primary_owner == LOCAL
    assert mgr.membership_state(OTHER) == "ACTIVE"  # replica/other node untouched


# ===========================================================================
# 6. Lifecycle independence + membership-vs-health separation
# ===========================================================================


def test_node_draining_does_not_auto_drain_shards(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    lc = ShardLifecycleManager(store)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-idp")
    lc.create(shard, owner=LOCAL, replicas=[])
    lc.activate(shard)
    mgr.begin_drain(LOCAL)  # node drains...
    assert shard.state is ShardState.ACTIVE  # ...shard lifecycle is NOT auto-drained


def test_node_removed_does_not_auto_migrate_shards(tmp_path):
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    lc = ShardLifecycleManager(store)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-rm")
    lc.create(shard, owner=LOCAL, replicas=[])
    lc.activate(shard)
    mgr.remove(LOCAL)  # node removed...
    assert shard.state is ShardState.ACTIVE          # ...shard untouched, no migration
    assert shard.primary_owner == LOCAL              # no automatic ownership transfer


def test_membership_is_not_replication_health(tmp_path):
    # A DRAINING member is still a valid owner IF its shard is ACTIVE (membership
    # availability is a different axis from VS-09 delivery health). The writable
    # path keeps serving while the shard lifecycle allows it.
    store = _durable(tmp_path / "shards.json")
    mgr = _manager(store)
    _register_active(mgr, LOCAL)
    mgr.begin_drain(LOCAL)
    shard = Shard(namespace="ns_cl", node_id=LOCAL, id="cl-shard-health")
    router = _router(store, membership=mgr)
    router.register_shard(shard)
    picked = router.route_write("ns_cl", "k", LOCAL)
    assert picked.primary_owner == LOCAL
    assert mgr.is_available(LOCAL) is True          # DRAINING still available
    assert "replication" not in mgr.get(LOCAL).to_dict()  # no health fields forwarded


# ===========================================================================
# 7. Restart consistency (cluster metadata + shard metadata together)
# ===========================================================================


def test_cluster_and_shard_metadata_survive_restart_together(tmp_path):
    path = tmp_path / "shards.json"
    # first "uptime"
    store1 = _durable(path)
    mgr1 = _manager(store1)
    bootstrap_local_node(mgr1, LOCAL)
    lc1 = ShardLifecycleManager(store1, membership=mgr1)
    shard = Shard(namespace="ns_persist", node_id=LOCAL, id="persist-shard-v11", replicas=["node-r"])
    lc1.create(shard, owner=LOCAL, replicas=["node-r"])
    lc1.activate(shard)

    # "restart": fresh store, fresh manager, fresh lifecycle over the same file
    store2 = _durable(path)
    mgr2 = _manager(store2)
    node = bootstrap_local_node(mgr2, LOCAL)
    assert node.state is NodeMembershipState.ACTIVE
    loaded = store2.list_by_namespace("ns_persist")[0]
    assert loaded.id == "persist-shard-v11"        # shard identity stable
    assert loaded.state is ShardState.ACTIVE       # lifecycle stable
    assert loaded.primary_owner == LOCAL           # ownership stable
    assert mgr2.cluster_id == CLUSTER_ID           # cluster identity stable
    # the full path still resolves a write to the known-owner shard
    router2 = _router(store2, membership=mgr2)
    picked = router2.route_write("ns_persist", "k", LOCAL)
    assert picked.primary_owner == LOCAL


def test_restart_no_fabricated_active_or_ownership_transfer(tmp_path):
    path = tmp_path / "shards.json"
    store1 = _durable(path)
    mgr1 = _manager(store1)
    _register_active(mgr1, LOCAL)
    _register_active(mgr1, OTHER)
    mgr1.begin_drain(LOCAL)
    lc1 = ShardLifecycleManager(store1)
    draining = Shard(namespace="ns_persist", node_id=LOCAL, id="drain-shard-v11")
    lc1.create(draining, owner=LOCAL, replicas=[])
    lc1.activate(draining)
    lc1.begin_drain(draining)  # shard draining too

    store2 = _durable(path)
    mgr2 = _manager(store2)
    node = bootstrap_local_node(mgr2, LOCAL)
    assert node.state is NodeMembershipState.DRAINING      # membership NOT fabricated ACTIVE
    assert mgr2.get(OTHER).state is NodeMembershipState.ACTIVE
    loaded = store2.list_by_namespace("ns_persist")[0]
    assert loaded.state is ShardState.DRAINING              # shard NOT fabricated ACTIVE
    assert loaded.primary_owner == LOCAL                    # no automatic ownership transfer


# ===========================================================================
# 8. REST / API compatibility
# ===========================================================================


def test_cluster_endpoints_read_only():
    from fastapi.testclient import TestClient
    from vsector.api.rest import app

    client = TestClient(app)
    r = client.get("/cluster")
    assert r.status_code == 200
    body = r.json()
    assert body["cluster_id"] == get_settings().cluster_id   # stable cluster identity
    assert body["node_id"] == get_settings().node_id          # local node identity
    assert body["membership_state"] == "ACTIVE"               # local node registered at boot
    assert any(n["node_id"] == body["node_id"] for n in body["known_nodes"])
    assert all("cluster_id" in n and "membership_state" in n for n in body["known_nodes"])

    r = client.get("/cluster/nodes")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert any(n["node_id"] == body["node_id"] for n in r.json())

    r = client.get(f"/cluster/nodes/{body['node_id']}")
    assert r.status_code == 200
    assert r.json()["membership_state"] == "ACTIVE"

    r = client.get("/cluster/nodes/ghost-node")
    assert r.status_code == 404