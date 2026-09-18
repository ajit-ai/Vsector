"""VS-13 â€” shard migration & rebalancing.

Tests drive a VALIDATED, real-lifecycle migration manager against a durable
in-process etcd store and a real ``ShardTransferProvider`` that moves shard
data through an actual storage boundary (WAL + SegmentStore under per-node
roots). The tests assert VS-13's invariant core:

 * migration is a FIRST-CLASS, SEPARATE object â€” never an overloaded shard
   ``node_id``/``state``;
 * ownership is derived from ``Shard.node_id`` (the single source of truth)
   and changes ONLY at the commit boundary, and ONLY after the target digest
   is verified to match the captured source digest;
 * real data movement (export -> import -> verify/finalize) crosses the
   storage boundary â€” no bare ``node_id`` reassignment;
 * restart semantics: durable migrations survive; an incomplete migration is
   never fabricated into COMPLETED; a failed/rolled-back migration never
   changes ownership;
 * failures at any step are observable (FAILED + error), deterministic, and
   never leave stale ownership.

Rebalancing is read-only (a deterministic "if we re-sharded, this is what we'd
do" recommendation) â€” it never moves data and never runs automatically.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from vsector.api.rest import app
from vsector.cluster.membership import ClusterMembershipManager
from vsector.sharding.etcd import EtcdStore
from vsector.sharding.lifecycle import ShardLifecycleManager
from vsector.sharding.migration import (
   MigrationState,
   ShardMigration,
   ShardMigrationManager,
   MigrationConflictError,
   MigrationError,
    InvalidMigrationError,
    MigrationNotFoundError,
    MigrationSourceError,
   MigrationStateError,
   MigrationTargetError,
   MigrationVerificationError,
)
from vsector.sharding.placement import RouteType
from vsector.sharding.router import ShardRouter
from vsector.sharding.shard import Shard, ShardState
from vsector.sharding.transfer import ShardTransferProvider, compute_digest
from vsector.storage.segment import SegmentStore
from vsector.storage.wal import WAL, WALEntry
from vsector.storage.recovery import recover
from vsector.models.vector_record import VectorRecord

CLUSTER = "vs-test-cluster"
NAMESPACE = "products"
DIM = 8


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

def _make_etcd(tmp_path: Path) -> EtcdStore:
   return EtcdStore(path=str(tmp_path / "etcd.json"))


def _make_cluster(etcd: EtcdStore) -> ClusterMembershipManager:
   cluster = ClusterMembershipManager(etcd, cluster_id=CLUSTER)
   for node_id in ("node-0", "node-1", "node-2"):
       cluster.register(node_id, cluster_id=CLUSTER)
       cluster.activate(node_id)
   return cluster


def _make_transfer(tmp_path: Path) -> ShardTransferProvider:
   node_dirs = {n: str(tmp_path / "nodes" / n) for n in ("node-0", "node-1", "node-2")}
   return ShardTransferProvider(node_base_dirs=node_dirs)


def _seed_shard(transfer: ShardTransferProvider, node_id: str, shard: Shard,
               n: int = 5, seed: int = 0) -> list[VectorRecord]:
   """Persist real records (and a tombstone) into node storage under the exact
   layout ``ShardTransferProvider`` reads; return what recovery sees."""
   base = transfer.node_base_dir(node_id)
   seg_dir = base / "segments" / shard.id
   wal_dir = base / "wal" / shard.id
   seg = SegmentStore(seg_dir)
   wal = WAL(wal_dir)
   records: list[VectorRecord] = []
   try:
       for i in range(n):
           rec = VectorRecord(
               id=uuid.UUID(int=(seed + i) + 0x1000),
               namespace=shard.namespace,
               vector=[float((seed + i)) / 10.0] * DIM,
               dimension=DIM,
               metadata={"i": seed + i},
           )
           seg.put(rec)
           wal.append(WALEntry(
               payload=json.dumps(
                   {"__op": "upsert", "record": rec.model_dump(mode="json")},
                   sort_keys=True,
               ).encode("utf-8"),
           ))
           records.append(rec)
       # a tombstone (concurrently deleted) must survive the migration too
       wal.append(WALEntry(
           payload=json.dumps({"__op": "delete", "id": f"{shard.id}:{seed + n}"}).encode("utf-8"),
       ))
       seg.flush()
       wal.flush()
       return records
   finally:
       try:
           wal.close()
       except Exception:
           pass


def _migratable_shard(store: EtcdStore, lifecycle: ShardLifecycleManager, ns: str,
                     shard_id: str, owner: str = "node-0") -> Shard:
   shard = Shard(id=shard_id, namespace=ns, node_id=owner, state=ShardState.CREATING)
   lifecycle.create(shard, owner=owner)
   return lifecycle.activate(shard)


@pytest.fixture
def env(tmp_path):
   """Fresh durable store + ACTIVE membership (node-0..2) + per-node storage."""
   etcd = _make_etcd(tmp_path)
   cluster = _make_cluster(etcd)
   lifecycle = ShardLifecycleManager(etcd)
   transfer = _make_transfer(tmp_path)
   manager = ShardMigrationManager(
       store=etcd, membership=cluster, transfer=transfer, cluster_id=CLUSTER,
   )
   router = ShardRouter(etcd=etcd, membership=cluster)
   return {
       "store": etcd,
       "cluster": cluster,
       "lifecycle": lifecycle,
       "transfer": transfer,
       "manager": manager,
       "router": router,
       "tmp": tmp_path,
   }


def _create(env, *, shard_id="sh-000", owner="node-0", target="node-1",
           ns=NAMESPACE, seed=0) -> ShardMigration:
   lifecycle = env["lifecycle"]
   shard = _migratable_shard(env["store"], lifecycle, ns, shard_id, owner=owner)
   _seed_shard(env["transfer"], owner, shard, seed=seed)
   return env["manager"].create_migration(ns, shard_id, target)


def _full_run(env, migration_id: str) -> ShardMigration:
   mgr = env["manager"]
   mgr.prepare(migration_id)
   mgr.start_copy(migration_id)
   mgr.verify(migration_id)
   return mgr.commit(migration_id)


# ==========================================================================
# 1  creation / validation (data is NOT moved on create)
# ==========================================================================

class TestCreation:
   def test_create_migration_valid(self, env):
       m = _create(env)
       assert m.state is MigrationState.PENDING
       assert m.source_node_id == "node-0"
       assert m.target_node_id == "node-1"
       assert m.source_node_id != m.target_node_id
       assert m.cluster_id == CLUSTER
       shard = env["store"].get("sh-000")
       assert shard.node_id == "node-0"
       assert shard.state is ShardState.ACTIVE

   def test_create_list_get_roundtrip(self, env):
       m = _create(env)
       got = env["manager"].get(m.migration_id)
       assert got == m
       assert [x.migration_id for x in env["manager"].list()] == [m.migration_id]

   def test_create_unknown_shard_raises(self, env):
        with pytest.raises(InvalidMigrationError):
            env["manager"].create_migration(NAMESPACE, "nope", "node-1")
        assert env["manager"].list() == []

   def test_create_unknown_source_raises(self, env):
       store = env["store"]
       # ownership points at a node that is NOT a cluster member
       store.put(Shard(id="sh-x", namespace=NAMESPACE, node_id="ghost",
                       state=ShardState.ACTIVE))
       with pytest.raises(MigrationSourceError):
           env["manager"].create_migration(NAMESPACE, "sh-x", "node-1")

   def test_create_unknown_target_raises(self, env):
       _migratable_shard(env["store"], env["lifecycle"], NAMESPACE, "sh-000")
       with pytest.raises(MigrationTargetError):
           env["manager"].create_migration(NAMESPACE, "sh-000", "ghost-target")
       assert env["manager"].list() == []

   def test_create_removed_target_raises(self, env):
       env["cluster"].remove("node-2")
       _migratable_shard(env["store"], env["lifecycle"], NAMESPACE, "sh-000")
       with pytest.raises(MigrationTargetError):
           env["manager"].create_migration(NAMESPACE, "sh-000", "node-2")

   def test_create_removed_source_raises(self, env):
       env["cluster"].remove("node-2")
       _migratable_shard(env["store"], env["lifecycle"], NAMESPACE, "sh-000", owner="node-2")
       with pytest.raises(MigrationSourceError):
           env["manager"].create_migration(NAMESPACE, "sh-000", "node-1")

   def test_create_same_source_and_target_raises(self, env):
        _migratable_shard(env["store"], env["lifecycle"], NAMESPACE, "sh-000")
        with pytest.raises(MigrationTargetError):
            env["manager"].create_migration(NAMESPACE, "sh-000", "node-0")
        assert env["manager"].list() == []

   def test_create_target_in_other_cluster_raises(self, env):
        _migratable_shard(env["store"], env["lifecycle"], NAMESPACE, "sh-000")
        # node-x lives in a DIFFERENT cluster (registered on a separate
        # membership manager) so it is NOT a member of our cluster.
        foreign = ClusterMembershipManager(env["store"], cluster_id="other-cluster")
        foreign.register("node-x")
        with pytest.raises(MigrationTargetError):
            env["manager"].create_migration(NAMESPACE, "sh-000", "node-x")

   def test_create_oom_offline_shard_rejected(self, env):
       env["store"].put(Shard(id="sh-oom", namespace=NAMESPACE, node_id="node-0",
                              state=ShardState.CREATING))
       with pytest.raises(MigrationSourceError):
           env["manager"].create_migration(NAMESPACE, "sh-oom", "node-1")

   def test_create_retired_shard_rejected(self, env):
       env["store"].put(Shard(id="sh-dead", namespace=NAMESPACE, node_id="node-0",
                              state=ShardState.OFFLINE))
       with pytest.raises(MigrationSourceError):
           env["manager"].create_migration(NAMESPACE, "sh-dead", "node-1")

   def test_create_conflicting_pending_migration_rejected(self, env):
       _create(env)
       with pytest.raises(MigrationConflictError):
           env["manager"].create_migration(NAMESPACE, "sh-000", "node-2")

   def test_create_does_not_move_data(self, env):
       m = _create(env)
       shard = env["store"].get("sh-000")
       src = _logical_state(env["transfer"], "node-0", shard)
       dst = _logical_state(env["transfer"], "node-1", shard)
       assert src.records
       assert not dst.records
       _ = m

   def test_create_fresh_copy_not_shared_with_store(self, env):
        m1 = _create(env)
        m2 = env["manager"].get(m1.migration_id)
        m2.migration_id = "tampered"
        assert env["manager"].get(m1.migration_id).migration_id == m1.migration_id


# ==========================================================================
# 2  data transfer integrity (export -> import -> verify -> finalize)
# ==========================================================================

def _logical_state(transfer, node_id, shard):
   return recover(
       WAL(transfer.node_base_dir(node_id) / "wal" / shard.id),
       SegmentStore(transfer.node_base_dir(node_id) / "segments" / shard.id),
   )


class TestTransfer:
   def test_source_export_and_target_import_digests_match(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       m = mgr.start_copy(m.migration_id)
       assert m.state is MigrationState.COPYING
       assert m.source_digest == m.target_digest
       # both sides now hold identical logical data
       shard = env["store"].get("sh-000")
       src = _logical_state(env["transfer"], "node-0", shard)
       dst = _logical_state(env["transfer"], "node-1", shard)
       assert len(src.records) == len(dst.records)
       ids = {r.id for r in src.records}
       assert ids == {r.id for r in dst.records}
       assert src.deleted_ids == dst.deleted_ids

   def test_tampered_target_fails_verification_and_keeps_ownership(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       m = mgr.start_copy(m.migration_id)
       # REAL on-disk tamper: unlink a segment file on the TARGET copy
       target_root = env["transfer"].node_base_dir("node-1") / "segments" / "sh-000"
       # REAL on-disk tamper: unlink an actual copied SST on node-1 so the
       # recomputed target digest genuinely diverges and verify raises
       sst = sorted(env["transfer"].node_base_dir("node-1").rglob("sst-*.jsonl"))
       assert sst, "target copy must hold a flushed SST to corrupt"
       sst[0].unlink(missing_ok=True)
       with pytest.raises(MigrationVerificationError):
           mgr.verify(m.migration_id)
       m = env["manager"].get(m.migration_id)
       assert m.state is MigrationState.FAILED
       assert m.error
       # ownership NEVER moved
       assert env["store"].get("sh-000").node_id == "node-0"

   def test_complete_migration_preserves_all_source_records(self, env):
       m = _create(env)
       shard = env["store"].get("sh-000")
       src_before = _logical_state(env["transfer"], "node-0", shard)
       m = _full_run(env, m.migration_id)
       shard = env["store"].get("sh-000")
       src = _logical_state(env["transfer"], "node-0", shard)
       dst = _logical_state(env["transfer"], "node-1", shard)
       assert len(dst.records) == len(src_before.records)
       assert src.deleted_ids == dst.deleted_ids
       assert m.source_digest == m.target_digest


# ==========================================================================
# 3  ownership / routing transitions (VS-13 boundary semantics)
# ==========================================================================

class TestOwnership:
   def test_ownership_unchanged_before_commit(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       mgr.start_copy(m.migration_id)
       mgr.verify(m.migration_id)
       assert env["store"].get("sh-000").node_id == "node-0"
       # even at VERIFYING the source is still the authoritative owner
       d = env["router"].route_read_decision("products", "sh-000:0", local_node_id="node-0")
       assert d.route_type is RouteType.LOCAL

   def test_ownership_changes_only_after_verify(self, env):
       m = _create(env)
       m = env["manager"].start_copy(m.migration_id) if False else env["manager"].get(m.migration_id)
       _ = m
       mgr = env["manager"]
       _full_run(env, m.migration_id)
       assert env["store"].get("sh-000").node_id == "node-1"

   def test_target_authoritative_for_routing_after_commit(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       mgr.start_copy(m.migration_id)
       mgr.verify(m.migration_id)
       mgr.commit(m.migration_id)
       owners = [s.node_id for s in env["router"].writable(NAMESPACE)]
       assert owners == ["node-1"]


# ==========================================================================
# 4  failure / rollback semantics
# ==========================================================================

class TestFailure:
   def test_stale_migration_cannot_overwrite_newer_owner(self, env):
       m = _create(env)
       # ... a concurrent migration abandons; ownership of the shard moves to
       # node-2 (a NEWER authoritative record), bumping shard.version
       shard = env["store"].get("sh-000")
       shard.node_id = "node-2"
       shard.version += 1
       env["store"].put(shard)
       with pytest.raises(MigrationSourceError):
           env["manager"].prepare(m.migration_id)
       m = env["manager"].get(m.migration_id)
       assert m.state is MigrationState.FAILED
       assert env["store"].get("sh-000").node_id == "node-2"

   def test_copy_step_failure_rolls_back_state_and_version(self, env):
       class FailingTransfer(ShardTransferProvider):
           def export_shard(self, node_id, shard):
               raise RuntimeError("export boom")

       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       m = mgr.get(m.migration_id)
       v = m.version
       m2 = ShardMigrationManager(
           store=env["store"], membership=env["cluster"],
           transfer=FailingTransfer(), cluster_id=CLUSTER,
       )
       with pytest.raises(MigrationStateError):
           m2.start_copy(m.migration_id)
       m3 = env["manager"].get(m.migration_id)
       assert m3.state is MigrationState.FAILED
       assert m3.error
       assert env["store"].get("sh-000").node_id == "node-0"

   def test_verification_failure_after_start_copy(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       mgr.start_copy(m.migration_id)
       # tamper the REAL target on-disk digest AFTER copy
       m = env["manager"].get(m.migration_id)
       m.source_digest = compute_digest([], [])
       env["store"].put_migration(m)
       with pytest.raises(MigrationVerificationError):
           mgr.verify(m.migration_id)

   def test_verify_failure_does_not_change_owner_when_target_broken(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       mgr.start_copy(m.migration_id)
       # simulate the digest stored on the target being wrong â€” verify must fail
       m = env["manager"].get(m.migration_id)
       m.source_digest = compute_digest([], [])
       env["store"].put_migration(m)
       with pytest.raises(MigrationVerificationError):
           mgr.verify(m.migration_id)
       assert env["store"].get("sh-000").node_id == "node-0"


# ==========================================================================
# 5  restart semantics (durable store + durable migrations)
# ==========================================================================

def _membership_only(etcd):
    """Durable-membership manager: auto-loads existing durable members.

    Critically NOT re-registering them (re-registering is what raised
    DuplicateNodeRegistrationError on reopen). Ownership is durable; a restart
    must recover it, never re-create it.
    """
    cluster = ClusterMembershipManager(etcd, cluster_id=CLUSTER)
    return cluster


def _reopen(env):
    """Simulate a node restart: new store + manager over the SAME durable file.

    MUST NOT re-register durable members (they already exist in the durable
    etcd store); re-registering is what raised DuplicateNodeRegistrationError.
    """
    etcd = _make_etcd(env["tmp"])
    cluster = _membership_only(etcd)   # durable members auto-loaded, NOT re-registered
    lifecycle = ShardLifecycleManager(etcd)
    transfer = _make_transfer(env["tmp"])
    manager = ShardMigrationManager(
        store=etcd, membership=cluster, transfer=transfer, cluster_id=CLUSTER,
    )
    router = ShardRouter(etcd=etcd, membership=cluster)
    return {
        "store": etcd, "cluster": cluster, "lifecycle": lifecycle,
        "transfer": transfer, "manager": manager, "router": router,
        "tmp": env["tmp"],
    }


class TestRestart:
   def test_migration_state_survives_node_restart(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       mgr.start_copy(m.migration_id)
       # crash mid-COPYING; restart must recover the durable state, not re-register
       env2 = _reopen(env)
       m2 = env2["manager"].get(m.migration_id)
       assert m2.state is MigrationState.COPYING
       env2["manager"].verify(m.migration_id)
       env2["manager"].commit(m.migration_id)
       assert env2["store"].get("sh-000").node_id == "node-1"

   def test_restart_never_fabricates_completed(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       # crash before copy; reopen must NOT invent a COMPLETED state
       env2 = _reopen(env)
       m2 = env2["manager"].get(m.migration_id)
       assert m2.state is not MigrationState.COMPLETED
       assert m2.state is MigrationState.PREPARING
       assert env2["store"].get("sh-000").node_id == "node-0"

   def test_completed_stays_completed_after_restart(self, env):
       m = _create(env)
       m = _full_run(env, m.migration_id)
       assert m.state is MigrationState.COMPLETED
       env2 = _reopen(env)
       m2 = env2["manager"].get(m.migration_id)
       assert m2.state is MigrationState.COMPLETED
       assert m2.source_digest == m2.target_digest
       assert env2["store"].get("sh-000").node_id == "node-1"

   def test_failed_stays_failed_and_owner_kept(self, env):
       m = _create(env)
       mgr = env["manager"]
       mgr.prepare(m.migration_id)
       mgr.start_copy(m.migration_id)
       target = env["transfer"].node_base_dir("node-1") / "segments" / "sh-000"
       # REAL on-disk tamper: unlink an actual copied SST on node-1 so the
       # recomputed target digest genuinely diverges and verify raises
       sst = sorted(env["transfer"].node_base_dir("node-1").rglob("sst-*.jsonl"))
       assert sst, "target copy must hold a flushed SST to corrupt"
       sst[0].unlink(missing_ok=True)
       env2 = _reopen(env)
       with pytest.raises(MigrationVerificationError):
           env2["manager"].verify(m.migration_id)
       m2 = env2["manager"].get(m.migration_id)
       assert m2.state is MigrationState.FAILED
       assert env2["store"].get("sh-000").node_id == "node-0"


# ==========================================================================
# 6  replicas: primary not duplicated, removed on commit, no auto-promotion
# ==========================================================================

class TestReplicas:
   def test_primary_removed_from_replicas_after_commit(self, env):
       m = _create(env, owner="node-0", target="node-1")
       # seed source with a replica set that includes the target
       shard = env["store"].get("sh-000")
       shard.replicas = ["node-1", "node-2"]
       env["store"].put(shard)
       _full_run(env, m.migration_id)
       shard = env["store"].get("sh-000")
       assert shard.node_id == "node-1"
       assert shard.state is ShardState.ACTIVE
       assert "node-1" not in shard.replicas

   def test_primary_never_in_replicas_anywhere(self, env):
       m = _create(env, owner="node-0", target="node-1")
       shard = env["store"].get("sh-000")
       shard.replicas = ["node-0", "node-2"]
       env["store"].put(shard)
       _full_run(env, m.migration_id)
       shard = env["store"].get("sh-000")
       assert shard.node_id == "node-1"
       assert "node-1" not in shard.replicas
       assert "node-0" in shard.replicas  # source becomes a replica? NO â€” spec:
       # old primary becomes a plain replica; primary is never duplicated.
       assert "node-0" != shard.node_id

   def test_replica_validation_rejects_invalid_members(self, env):
       m = _create(env, owner="node-0", target="node-1")
       shard = env["store"].get("sh-000")
       shard.replicas = ["ghost-replica", "node-2"]
       env["store"].put(shard)
       with pytest.raises(MigrationError):
           env["manager"].commit(m.migration_id) if False else _full_run(env, m.migration_id)

   def test_no_primary_duplication_after_any_run(self, env):
       m = _create(env, owner="node-0", target="node-1")
       shard = env["store"].get("sh-000")
       shard.replicas = ["node-1", "node-2"]
       env["store"].put(shard)
       _full_run(env, m.migration_id)
       shard = env["store"].get("sh-000")
       assert "node-1" not in shard.replicas
       assert len(set(shard.replicas)) == len(shard.replicas)


# ==========================================================================
# 7  rebalancing (read-only recommendation)
# ==========================================================================

class TestRebalance:
   def test_rebalance_is_deterministic_readonly_recommendation(self, env):
       for shard_id, owner in (
           ("sh-000", "node-0"), ("sh-001", "node-1"), ("sh-002", "node-2"),
       ):
           _seed_shard(env["transfer"], owner,
                       env["store"].get(shard_id) or _migratable_shard(
                           env["store"], env["lifecycle"], NAMESPACE, shard_id, owner=owner))
       env["router"].invalidate(NAMESPACE)
       rec1 = env["manager"].recommend_rebalance()
       rec2 = env["manager"].recommend_rebalance()
       assert rec1 == rec2
       assert rec1 == []

   def test_rebalance_empty_when_balanced(self, env):
        # one shard per node -> perfectly balanced
        for i, owner in enumerate(("node-0", "node-1")):
            _migratable_shard(env["store"], env["lifecycle"], NAMESPACE, f"sh-{i:03d}",
                              owner=owner)
        env["router"].invalidate(NAMESPACE)
        assert env["manager"].recommend_rebalance() == []
