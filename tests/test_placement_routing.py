"""VS-12: deterministic placement, routing & local/remote decisions tests.

Covers:
- deterministic placement (stable selection, distribution, namespace isolation,
  empty/unknown namespace errors, authoritative shard metadata);
- ownership resolution (owner from shard.node_id, validated against VS-11
  membership, replica never promoted);
- local routing (LOCAL decision, existing write path);
- remote routing (REMOTE decision + target, no local write, no fake execution);
- lifecycle-aware decisions (CREATING/ACTIVE/DRAINING/OFFLINE/RETIRED semantics);
- query routing (readable-only, deterministic multi-shard candidate set);
- restart & persistence (decisions stable, derived from persisted metadata);
- purity (repeated routing never mutates shard/ownership/membership/metadata);
- REST placement endpoints.
"""
from __future__ import annotations

import pytest

from vsector.cluster import ClusterMembershipManager
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.placement import RouteType
from vsector.sharding.shard import Shard, ShardState
from vsector.sharding.exceptions import (
    PlacementError,
    OwnershipMismatchError,
)
from vsector.ingest.service import IngestService
from vsector.models.namespace import Namespace, DistanceMetric, IndexType

CLUSTER_ID = "cluster-a"
NS = "ns_route"
LOCAL = "node-0"
REMOTE_NODE = "node-1"


def _store(**kw):
    return EtcdStore(**kw)


def _router(store=None, membership=None, **kw):
    return ShardRouter(store or _store(), membership=membership, **kw)


def _membership(store, cluster_id: str = CLUSTER_ID):
    return ClusterMembershipManager(store, cluster_id=cluster_id)


def _register_known(membership: ClusterMembershipManager, *node_ids: str) -> None:
    for nid in node_ids:
        membership.register(nid)
        membership.activate(nid)


def _shard(node_id: str, shard_id: str, state: ShardState = ShardState.ACTIVE,
           replicas: tuple[str, ...] = ()) -> Shard:
    return Shard(namespace=NS, node_id=node_id, id=shard_id, state=state, replicas=list(replicas))


def _durable_router(path, membership_cluster: str | None = CLUSTER_ID):
    store = _store(path=str(path))
    mgr = _membership(store, cluster_id=membership_cluster) if membership_cluster else None
    return _router(store, membership=mgr), store, mgr


# ===========================================================================
# 1. Placement
# ===========================================================================


def test_same_key_same_shard_deterministic():
    store = _store()
    router = _router(store)
    router.register_shard(_shard(LOCAL, "s-1"))
    router.register_shard(_shard(REMOTE_NODE, "s-2"))
    first = router.place(NS, "record-42", LOCAL)
    for _ in range(25):
        assert router.place(NS, "record-42", LOCAL).shard_id == first.shard_id


def test_different_keys_distribute():
    store = _store()
    router = _router(store)
    for i in range(8):
        router.register_shard(_shard(f"node-{i % 2}", f"s-{i}"))
    seen = {router.place(NS, f"k-{i}", LOCAL).shard_id for i in range(500)}
    assert len(seen) > 1  # deterministic distribution across the shard set


def test_namespace_isolation():
    store = _store()
    router = _router(store)
    router.register_shard(_shard(LOCAL, "s-a"))
    router.register_shard(Shard(namespace="ns_other", node_id=REMOTE_NODE, id="s-b"))
    for i in range(200):
        d = router.place(NS, f"k-{i}", LOCAL)
        assert d.namespace == NS                                        # never routed to another namespace
        assert d.shard_id == "s-a"
    assert router.place("ns_other", "k-1", LOCAL).shard_id == "s-b"


def test_empty_shard_set_deterministic_error():
    router = _router(_store())
    with pytest.raises(PlacementError):
        router.place(NS, "k")
    with pytest.raises(ValueError):  # REST 400 mapping
        router.place(NS, "k")


def test_unknown_namespace_deterministic_error(tmp_path):
    store = _store(path=str(tmp_path / "shards.json"))
    mgr = _membership(store)
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    with pytest.raises(PlacementError):
        router.route_write_decision("no-such-ns", "k", LOCAL)
    with pytest.raises(PlacementError):
        router.route_read_decision("no-such-ns", "k", LOCAL)


def test_placement_uses_authoritative_shard_metadata(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "authoritative-1"))
    d = router.place(NS, "k", LOCAL)
    assert d.shard_id == "authoritative-1"       # selected from the real store, never manufactured


# ===========================================================================
# 2. Ownership resolution
# ===========================================================================


def test_decision_resolves_primary_owner(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "s-owner", replicas=("rep-a",)))
    d = router.place(NS, "k", LOCAL)
    assert d.owner_node_id == REMOTE_NODE
    assert d.target == REMOTE_NODE                    # owner is the resolved target


def test_replicas_never_become_primary(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "s-rep", replicas=("rep-a", "rep-b")))
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.owner_node_id == REMOTE_NODE
    assert d.owner_node_id not in ("rep-a", "rep-b")  # replicas are never the primary
    assert d.target == REMOTE_NODE


def test_owner_validated_against_membership_known(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "s-ok"))
    router.register_shard(_shard(REMOTE_NODE, "s-ok2"))
    assert router.place(NS, "k", LOCAL).route_type in (RouteType.LOCAL, RouteType.REMOTE)
    assert router.place(NS, "k", LOCAL).owner_membership_state == "ACTIVE"


def test_unknown_owner_unavailable(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard("ghost-owner", "s-ghost"))
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.UNAVAILABLE
    assert d.owner_membership_state == "unknown"
    assert "not a known cluster member" in (d.reason or "")
    # the shard owner is untouched - no adoption, no forwarding, no transfer
    stored = store.list_by_namespace(NS)[0]
    assert stored.node_id == "ghost-owner"
    assert stored.state is ShardState.ACTIVE


def test_removed_owner_unavailable_no_transfer(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, REMOTE_NODE)
    mgr.remove(REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "s-gone", replicas=("rep-a",)))
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.UNAVAILABLE
    assert d.owner_membership_state == "REMOVED"
    stored = store.list_by_namespace(NS)[0]
    assert stored.node_id == REMOTE_NODE        # no automatic ownership change
    assert stored.replicas == ["rep-a"]          # no replica promotion


# ===========================================================================
# 3. Local routing
# ===========================================================================


def test_local_owner_local_decision(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "s-local"))
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.LOCAL
    assert d.is_local is True
    assert d.target == LOCAL
    assert d.local_node_id == LOCAL


def test_local_write_uses_existing_path(tmp_path):
    # VS-10 route_write + IngestService write path still work for a local owner
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "s-write"))
    picked = router.route_write(NS, "k", LOCAL)
    assert picked.id == "s-write"

    from vsector.storage.metadata import MetadataStore
    md = MetadataStore(path=str(tmp_path / "metadata.json"))
    ns = Namespace(name=NS, dimension=4, index_type=IndexType.FLAT, distance_metric=DistanceMetric.COSINE)
    md.create(ns)
    ingest = IngestService(md, router, base_dir=str(tmp_path / "data"), node_id=LOCAL)
    try:
        result = __import__("asyncio").run(ingest.upsert(NS, [{"vector": [1, 0, 0, 0]}]))
        assert result["upserted"] == 1
    finally:
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


# ===========================================================================
# 4. Remote routing
# ===========================================================================


def test_remote_owner_remote_decision(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "s-remote"))
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.REMOTE
    assert d.is_remote is True
    assert d.target == REMOTE_NODE
    assert d.local_node_id == LOCAL


def test_remote_write_is_refused_locally(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "s-nolocal"))
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.is_remote is True
    with pytest.raises(OwnershipMismatchError):       # the existing VS-10 gate
        router.route_write(NS, "k", LOCAL)             # NO local write happens


def test_remote_no_fake_execution(tmp_path):
    # a REMOTE decision never writes/creates anything locally
    path = tmp_path / "shards.json"
    store = _store(path=str(path))
    mgr = _membership(store)
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "s-fake"))
    before = path.read_text(encoding="utf-8")
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.is_remote is True
    assert path.read_text(encoding="utf-8") == before      # metadata untouched
    assert len(store.list_by_namespace(NS)) == 1            # no shard created
    assert store.get("s-fake").node_id == REMOTE_NODE      # owner untouched


# ===========================================================================
# 5. Lifecycle-aware decisions
# ===========================================================================


@pytest.mark.parametrize("state", [
    ShardState.CREATING,
    ShardState.DRAINING,
    ShardState.OFFLINE,
    ShardState.RETIRED,
])
def test_non_writable_states_unavailable_for_write(tmp_path, state):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    shard = _shard(LOCAL, f"s-{state.value}", state=state)
    router.register_shard(shard)
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.UNAVAILABLE
    assert d.reason is not None and "writable" in d.reason
    # DRAINING must not silently become writable / state never mutated
    assert store.list_by_namespace(NS)[0].state is state


def test_active_states_writable_and_readable(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "s-active"))
    assert router.route_write_decision(NS, "k", LOCAL).is_local is True
    assert router.route_read_decision(NS, "k", LOCAL).is_local is True


@pytest.mark.parametrize("state", [
    ShardState.CREATING,
    ShardState.OFFLINE,
    ShardState.RETIRED,
    ShardState.SPLITTING,
])
def test_non_readable_states_unavailable_for_read(tmp_path, state):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, f"s-r-{state.value}", state=state))
    d = router.route_read_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.UNAVAILABLE
    assert d.reason is not None and "not readable" in d.reason


def test_draining_readable_but_not_writable(tmp_path):
    # DRAINING stays readable (VS-10), never writable
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "s-drain", state=ShardState.DRAINING))
    assert router.route_read_decision(NS, "k", LOCAL).route_type is RouteType.LOCAL
    d = router.route_write_decision(NS, "k", LOCAL)
    assert d.route_type is RouteType.UNAVAILABLE
    assert "DRAINING" in (d.reason or "")


# ===========================================================================
# 6. Query routing
# ===========================================================================


def test_query_route_readable_only(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "q-act"))
    router.register_shard(_shard(LOCAL, "q-drain", state=ShardState.DRAINING))
    router.register_shard(_shard(LOCAL, "q-off", state=ShardState.OFFLINE))
    ids = [d.shard_id for d in router.query_route(NS, LOCAL)]
    assert sorted(ids) == ["q-act", "q-drain"]
    # matches the existing VS-10 fan-out
    assert sorted(s.id for s in router.route_for_query(NS)) == sorted(ids)


def test_query_route_deterministic_multi_shard(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL)
    router = _router(store, membership=mgr)
    for i in range(5):
        router.register_shard(_shard(LOCAL, f"q-{i}"))
    first = [d.to_dict() for d in router.query_route(NS, LOCAL)]
    for _ in range(10):
        assert [d.to_dict() for d in router.query_route(NS, LOCAL)] == first
    assert len(first) == 5
    assert {d["route_type"] for d in first} == {"LOCAL"}


def test_query_route_removed_owner_unavailable(tmp_path):
    # a readable shard whose owner is removed is UNAVAILABLE in the fan-out
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, REMOTE_NODE)
    mgr.remove(REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(REMOTE_NODE, "q-gone"))
    decisions = router.query_route(NS, LOCAL)
    assert decisions[0].route_type is RouteType.UNAVAILABLE


# ===========================================================================
# 7. Restart & persistence
# ===========================================================================


def test_routing_decision_stable_across_restart(tmp_path):
    path = tmp_path / "shards.json"
    # first "uptime"
    store1 = _store(path=str(path))
    mgr1 = _membership(store1)
    _register_known(mgr1, LOCAL, REMOTE_NODE)
    router1 = _router(store1, membership=mgr1)
    router1.register_shard(_shard(LOCAL, "p-local"))
    router1.register_shard(_shard(REMOTE_NODE, "p-remote"))
    before = {f"k-{i}": router1.route_write_decision(NS, f"k-{i}", LOCAL).to_dict() for i in range(50)}

    # "restart"
    store2 = _store(path=str(path))
    mgr2 = _membership(store2)
    router2 = _router(store2, membership=mgr2)
    for i in range(50):
        assert router2.route_write_decision(NS, f"k-{i}", LOCAL).to_dict() == before[f"k-{i}"]


def test_placement_view_derived_from_persisted_state(tmp_path):
    path = tmp_path / "shards.json"
    store1 = _store(path=str(path))
    mgr1 = _membership(store1)
    _register_known(mgr1, LOCAL, REMOTE_NODE)
    router1 = _router(store1, membership=mgr1)
    router1.register_shard(_shard(LOCAL, "v-l"))
    router1.register_shard(_shard(REMOTE_NODE, "v-r", replicas=("rep-a",)))
    view1 = router1.placement_view(NS, LOCAL)

    store2 = _store(path=str(path))
    mgr2 = _membership(store2)
    router2 = _router(store2, membership=mgr2)
    assert router2.placement_view(NS, LOCAL) == view1


def test_placement_view_fresh_copies(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "f-1", replicas=("rep-a",)))
    view = router.placement_view(NS, LOCAL)
    shard_meta = view["shards"][0]
    assert shard_meta["primary_owner"] == LOCAL
    assert shard_meta["owner_membership_state"] == "ACTIVE"
    assert shard_meta["route_type"] == "LOCAL"
    assert shard_meta["replicas"] == ["rep-a"]
    # mutating the returned view never corrupts the authoritative store
    shard_meta["primary_owner"] = "hacked"
    shard_meta["replicas"].append("hacked")
    fresh = router.placement_view(NS, LOCAL)["shards"][0]
    assert fresh["primary_owner"] == LOCAL
    assert fresh["replicas"] == ["rep-a"]


def test_placement_view_local_remote_relative(tmp_path):
    store, mgr = _store(path=str(tmp_path / "shards.json")), _membership(_store())
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "rel-l"))
    router.register_shard(_shard(REMOTE_NODE, "rel-r"))
    by_id = {s["shard_id"]: s for s in router.placement_view(NS, LOCAL)["shards"]}
    assert by_id["rel-l"]["route_type"] == "LOCAL"
    assert by_id["rel-l"]["target"] == LOCAL
    assert by_id["rel-r"]["route_type"] == "REMOTE"
    assert by_id["rel-r"]["target"] == REMOTE_NODE
    # same placement seen from node-1 reverses LOCAL/REMOTE
    by_id2 = {s["shard_id"]: s for s in router.placement_view(NS, REMOTE_NODE)["shards"]}
    assert by_id2["rel-l"]["route_type"] == "REMOTE"
    assert by_id2["rel-r"]["route_type"] == "LOCAL"


# ===========================================================================
# 8. Purity
# ===========================================================================


def test_routing_is_side_effect_free(tmp_path):
    path = tmp_path / "shards.json"
    store = _store(path=str(path))
    mgr = _membership(store)
    _register_known(mgr, LOCAL, REMOTE_NODE)
    router = _router(store, membership=mgr)
    router.register_shard(_shard(LOCAL, "pure-a", state=ShardState.DRAINING))
    router.register_shard(_shard(REMOTE_NODE, "pure-b"))
    before_bytes = path.read_bytes()
    before_shards = [(s.id, s.state.value, s.node_id, s.version)
                     for s in store.list_by_namespace(NS)]
    before_nodes = [(n.node_id, n.state.value, n.version) for n in store.list_nodes()]

    for i in range(100):
        router.place(NS, f"k-{i}", LOCAL)
        router.route_read_decision(NS, f"k-{i}", LOCAL)
        router.route_write_decision(NS, f"k-{i}", LOCAL)
        router.query_route(NS, LOCAL)
        router.placement_view(NS, LOCAL)
        router.shard_placement(store.list_by_namespace(NS)[0], LOCAL)

    assert path.read_bytes() == before_bytes                    # persisted metadata untouched
    after_shards = [(s.id, s.state.value, s.node_id, s.version)
                    for s in store.list_by_namespace(NS)]
    after_nodes = [(n.node_id, n.state.value, n.version) for n in store.list_nodes()]
    assert after_shards == before_shards                        # shard state/version untouched
    assert after_nodes == before_nodes                          # membership untouched


def test_repeated_routing_same_result():
    store = _store()
    router = _router(store)
    router.register_shard(_shard(LOCAL, "det-1"))
    router.register_shard(_shard(REMOTE_NODE, "det-2"))
    d1 = router.route_write_decision(NS, "same-key", LOCAL).to_dict()
    for _ in range(50):
        assert router.route_write_decision(NS, "same-key", LOCAL).to_dict() == d1


# ===========================================================================
# 9. REST placement endpoints
# ===========================================================================


def test_placement_endpoints_read_only():
    import uuid
    from fastapi.testclient import TestClient
    from vsector.api.rest import app, settings

    client = TestClient(app)
    # arbitrary-owner / unknown-owner shard never manufactured by the API
    r = client.get("/cluster/placement")
    assert r.status_code == 200
    body = r.json()
    assert "cluster_id" in body and "local_node_id" in body and "namespaces" in body
    assert body["local_node_id"] == settings.node_id

    ns = f"pl_ns_{uuid.uuid4().hex[:6]}"
    assert client.post("/v1/namespaces", json={"name": ns, "dimension": 4},
                       headers={"X-API-Key": "test-key"}).status_code == 200

    r = client.get(f"/v1/namespaces/{ns}/placement")
    assert r.status_code == 200
    view = r.json()
    assert view["namespace"] == ns
    assert view["local_node_id"] == settings.node_id
    assert view["shard_count"] == len(view["shards"])
    for s in view["shards"]:
        assert s["primary_owner"] == settings.node_id        # local single-node owner
        assert s["owner_membership_state"] == "ACTIVE"       # local node is a known member
        assert s["route_type"] == "LOCAL"                    # owned by this node
        assert s["target"] == settings.node_id
        assert "state" in s and "replicas" in s

    r = client.get(f"/v1/namespaces/{ns}/shards")
    assert r.status_code == 200
    assert r.json()["namespace"] == ns
    assert r.json()["shards"] == view["shards"]

    r = client.get("/v1/namespaces/definitely-missing-ns/placement")
    assert r.status_code == 404
    r = client.get("/v1/namespaces/definitely-missing-ns/shards")
    assert r.status_code == 404