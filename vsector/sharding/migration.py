"""VS-13: explicit, validated shard migration lifecycle.

Migration is the SAFE, explicit primitive for moving primary ownership of one
shard from a source node to a target node. It is a SEPARATE model from the VS-10
shard lifecycle (``ShardState`` is never overloaded with migration semantics) and
from VS-09 replication health.

Core lifecycle:

    PENDING -> PREPARING -> COPYING -> VERIFYING -> COMMITTING -> COMPLETED
    (any non-terminal state except COMMITTING may go -> CANCELLED; any
     non-terminal state may go -> FAILED)

Invariants preserved by this manager:

- ``shard.node_id`` remains the ONLY authoritative primary owner.
  ``migration.source_node_id`` is historical/migration information, and
  ``migration.target_node_id`` is prospective until the ownership commit
  succeeds.
- Ownership changes ONLY at the commit boundary, and ONLY after the target
  data has been copied and verified (digest equality).
- A migration whose source no longer matches the authoritative shard ownership
  (or whose captured shard version changed) is DETECTED deterministically and
  NEVER blindly overwrites newer ownership — it fails with a deterministic
  conflict.
- A failed pre-commit migration never changes authoritative ownership, and a
  failed ownership commit never fabricates success.
- Restart never fabricates a successful migration: a persisted incomplete
  migration remains incomplete (its state and digests survive), and the
  authoritative owner is always re-derived from the shard record.
- Replicas are never automatically promoted. After commit the new primary is
  never present in its own replica list, and remaining replicas are still
  validated against the VS-11 membership (a REMOVED replica makes the commit
  fail deterministically instead of silently introducing an invalid member).

Explicit non-goals (mirrors VS-13 spec): no cross-node RPC/HTTP transport, no
consensus, no leader election, no automatic failover, no automatic rebalancing,
no background migration workers, no replica-promotion protocol. Data movement is
performed through the ``ShardTransferProvider`` storage boundary (see
``vsector/sharding/transfer.py``); in VS-13 that boundary moves real durable
shard data between per-node storage roots within the architecture that exists.
"""
from __future__ import annotations

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import ClassVar

from ..cluster.exceptions import ClusterIdMismatchError, RemovedNodeError, UnknownNodeError
from .shard import Shard, ShardState

logger = logging.getLogger(__name__)

# Shard states that may participate in an explicit migration.
MIGRATABLE_STATES = frozenset({ShardState.ACTIVE, ShardState.DRAINING})

_TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")
_CANCELABLE = ("PENDING", "PREPARING", "COPYING", "VERIFYING")


class MigrationState(str, enum.Enum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    COPYING = "COPYING"
    VERIFYING = "VERIFYING"
    COMMITTING = "COMMITTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ShardMigration:
    """Durable, immutable-from-callers migration record (fresh copies on read)."""

    migration_id: str
    namespace: str
    shard_id: str
    source_node_id: str          # historical/current source, NOT ownership
    target_node_id: str          # prospective until commit succeeds
    state: MigrationState = MigrationState.PENDING
    source_version: int = 1      # shard.version captured at creation (stale guard)
    cluster_id: str = "vsector-cluster-default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    version: int = 1
    source_digest: str | None = None
    target_digest: str | None = None
    finalized: bool = False       # source finalization completed after commit

    TRANSITIONS: ClassVar[dict[MigrationState, frozenset]] = {
        MigrationState.PENDING: frozenset({MigrationState.PREPARING, MigrationState.CANCELLED, MigrationState.FAILED}),
        MigrationState.PREPARING: frozenset({MigrationState.COPYING, MigrationState.CANCELLED, MigrationState.FAILED}),
        MigrationState.COPYING: frozenset({MigrationState.VERIFYING, MigrationState.CANCELLED, MigrationState.FAILED}),
        MigrationState.VERIFYING: frozenset({MigrationState.COMMITTING, MigrationState.CANCELLED, MigrationState.FAILED}),
        MigrationState.COMMITTING: frozenset({MigrationState.COMPLETED, MigrationState.FAILED}),
        MigrationState.COMPLETED: frozenset(),  # terminal; auditable
        MigrationState.FAILED: frozenset(),      # terminal; observable
        MigrationState.CANCELLED: frozenset(),   # terminal
    }

    def transition(self, to: MigrationState) -> "ShardMigration":
        """Validate and apply an explicit migration-state transition (atomic on success)."""
        allowed = self.TRANSITIONS.get(self.state, frozenset())
        if to not in allowed:
            raise MigrationStateError(
                self.migration_id, f"cannot move from {self.state.value} to {to.value}"
            )
        self.state = to
        self.version += 1
        self.updated_at = time.time()
        return self

    def is_terminal(self) -> bool:
        return self.state.value in _TERMINAL

    def to_dict(self) -> dict:
        """Deterministic, JSON-serializable, fresh-copy snapshot."""
        return {
            "migration_id": self.migration_id,
            "namespace": self.namespace,
            "shard_id": self.shard_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "state": self.state.value,
            "source_version": self.source_version,
            "cluster_id": self.cluster_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "version": self.version,
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "finalized": self.finalized,
        }


class MigrationError(ValueError):
    """Base class for VS-13 migration failures (ValueError -> REST 400)."""


class MigrationNotFoundError(MigrationError):
    def __init__(self, migration_id: str):
        super().__init__(f"migration not found: {migration_id!r}")
        self.migration_id = migration_id


class InvalidMigrationError(MigrationError):
    def __init__(self, reason: str):
        super().__init__(f"invalid migration: {reason}")


class MigrationConflictError(MigrationError):
    def __init__(self, namespace: str, shard_id: str, reason: str):
        super().__init__(f"migration conflict: {namespace}:{shard_id}: {reason}")
        self.namespace = namespace
        self.shard_id = shard_id


class MigrationStateError(MigrationError):
    def __init__(self, migration_id: str, reason: str):
        super().__init__(f"migration state error: {migration_id!r}: {reason}")
        self.migration_id = migration_id


class MigrationSourceError(MigrationError):
    def __init__(self, node_id: str, reason: str):
        super().__init__(f"migration source error: source node {node_id!r} {reason}")
        self.node_id = node_id


class MigrationTargetError(MigrationError):
    def __init__(self, node_id: str, reason: str):
        super().__init__(f"migration target error: target node {node_id!r} {reason}")
        self.node_id = node_id


class MigrationVerificationError(MigrationError):
    def __init__(self, namespace: str, shard_id: str, expected: str, actual: str):
        super().__init__(
            f"migration verification failed: {namespace}:{shard_id}: target digest {actual[:12]} "
            f"!= source digest {expected[:12]}"
        )
        self.namespace = namespace
        self.shard_id = shard_id
        self.expected = expected
        self.actual = actual


class MigrationCommitError(MigrationError):
    def __init__(self, namespace: str, shard_id: str, target_node_id: str, reason: str):
        super().__init__(
            f"migration commit failed: {namespace}:{shard_id} -> {target_node_id!r}: {reason}"
        )
        self.namespace = namespace
        self.shard_id = shard_id
        self.target_node_id = target_node_id


class ShardMigrationManager:
    """Validated, durable, side-effect-aware migration operations.

    ``store`` is the same etcd abstraction used for shard/membership metadata
    (migrations persist under a reserved prefix and survive restart). ``membership``
    is the existing VS-11 ``ClusterMembershipManager`` — no second membership model
    is created. ``transfer`` is the VS-13 storage boundary that actually moves and
    verifies shard data.
    """

    def __init__(self, store, membership, transfer, cluster_id: str | None = None):
        self._store = store
        self._membership = membership
        self._transfer = transfer
        self.cluster_id = cluster_id or membership.cluster_id

    # --- reads (immutable-from-callers: fresh copies only) ------------------

    def get(self, migration_id: str) -> ShardMigration | None:
        m = self._store.get_migration(migration_id)
        return None if m is None else replace(m)

    def list(self) -> list[ShardMigration]:
        return sorted(
            (replace(m) for m in self._store.list_migrations()),
            key=lambda m: (m.created_at, m.migration_id),
        )

    def _require(self, migration_id: str) -> ShardMigration:
        m = self._store.get_migration(migration_id)
        if m is None:
            raise MigrationNotFoundError(migration_id)
        return m

    # --- validation helpers --------------------------------------------------

    def _validate_member(self, node_id: str, kind: str) -> None:
        """Require ``node_id`` to be a known, non-removed member of this cluster."""
        try:
            self._membership.validate_node(node_id)
            self._membership.validate_cluster_id(node_id)
        except UnknownNodeError:
            raise _kind_error(kind, node_id, "is not a known cluster member")
        except RemovedNodeError:
            raise _kind_error(kind, node_id, "has been REMOVED from the cluster")
        except ClusterIdMismatchError:
            raise _kind_error(kind, node_id, "belongs to a different cluster")

    def _assert_source_valid(self, m: ShardMigration) -> Shard:
        """Re-check the authoritative source BEFORE any step touches ownership.

        A migration whose source no longer matches the current authoritative shard
        record is a STALE migration: it must not overwrite newer ownership. The
        migration is marked FAILED (observable) and a deterministic conflict is
        raised without changing ownership.
        """
        shard = self._store.get(m.shard_id)
        if shard is None:
            self._fail(m, f"source validation failed: shard {m.shard_id} no longer exists")
            raise MigrationSourceError(m.source_node_id, f"shard {m.shard_id} no longer exists")
        if shard.state not in MIGRATABLE_STATES:
            self._fail(m, f"source validation failed: shard {m.shard_id} is not migratable (state={shard.state.value})")
            raise MigrationSourceError(m.source_node_id, f"shard {m.shard_id} is not migratable (state={shard.state.value})")
        if shard.node_id != m.source_node_id:
            self._fail(m, f"stale migration: shard {m.shard_id} is now owned by {shard.node_id!r}")
            raise MigrationSourceError(
                m.source_node_id,
                f"shard {m.shard_id} is now owned by {shard.node_id!r}; migration is stale and will not overwrite it",
            )
        if shard.version != m.source_version:
            self._fail(m, f"stale migration: shard {m.shard_id} version changed ({m.source_version} -> {shard.version})")
            raise MigrationSourceError(
                m.source_node_id,
                f"shard {m.shard_id} version changed ({m.source_version} -> {shard.version}); migration is stale",
            )
        return shard

    def _require_state(self, m: ShardMigration, expected: MigrationState) -> None:
        if m.state is not expected:
            raise MigrationStateError(
                m.migration_id, f"expected state={expected.value}, actual={m.state.value}"
            )

    def _fail(self, m: ShardMigration, error: str) -> ShardMigration:
        try:
            m.transition(MigrationState.FAILED)
        except MigrationStateError:
            pass
        m.error = error
        self._store.put_migration(m)
        logger.warning(f"migration {m.migration_id} FAILED: {error}")
        return m

    # --- creation -------------------------------------------------------------

    def create_migration(self, namespace: str, shard_id: str, target_node_id: str) -> ShardMigration:
        """Create a PENDING migration with full validation (no data is moved yet)."""
        shard = self._store.get(shard_id)
        if shard is None:
            raise InvalidMigrationError(f"unknown shard: {shard_id!r}")
        if shard.namespace != namespace:
            raise InvalidMigrationError(
                f"shard {shard_id!r} belongs to namespace {shard.namespace!r}, not {namespace!r}"
            )
        if shard.state not in MIGRATABLE_STATES:
            raise MigrationSourceError(
                shard.require_owner() if shard.node_id else "?",
                f"shard {shard_id!r} is not migratable (state={shard.state.value}; "
                "only ACTIVE/DRAINING shards may migrate)",
            )
        source = shard.require_owner()
        self._validate_member(source, "source")
        self._validate_member(target_node_id, "target")
        if target_node_id == source:
            raise MigrationTargetError(target_node_id, "target must differ from the source node")

        # no conflicting (non-terminal) migration already exists for this shard
        for m in self._store.list_migrations():
            if m.shard_id == shard_id and not m.is_terminal():
                raise MigrationConflictError(
                    namespace, shard_id,
                    f"an in-progress migration ({m.migration_id!r}) already exists (state={m.state.value})",
                )

        migration = ShardMigration(
            migration_id=uuid.uuid4().hex,
            namespace=namespace,
            shard_id=shard_id,
            source_node_id=source,
            target_node_id=target_node_id,
            source_version=shard.version,
            cluster_id=self.cluster_id,
        )
        self._store.put_migration(migration)
        logger.info(
            f"migration created {migration.migration_id}: {namespace}:{shard_id} {source} -> {target_node_id}"
        )
        return replace(migration)

    # --- lifecycle steps -------------------------------------------------------

    def prepare(self, migration_id: str) -> ShardMigration:
        """PENDING -> PREPARING. Re-validates that the source is still authoritative."""
        m = self._require(migration_id)
        self._require_state(m, MigrationState.PENDING)
        self._assert_source_valid(m)
        old_state, old_version = m.state, m.version
        try:
            m.transition(MigrationState.PREPARING)
            self._store.put_migration(m)
        except Exception:
            m.state, m.version = old_state, old_version
            raise
        return replace(m)

    def start_copy(self, migration_id: str) -> ShardMigration:
        """PREPARING -> COPYING: export source data, import into target storage.

        Digests are captured on export (source) and immediately re-derived on the
        imported target data (target). Actual data movement happens here through
        the storage boundary — this is never a bare ``node_id`` reassignment.
        """
        m = self._require(migration_id)
        self._require_state(m, MigrationState.PREPARING)
        shard = self._assert_source_valid(m)
        old_state, old_version = m.state, m.version
        try:
            m.transition(MigrationState.COPYING)
            self._store.put_migration(m)
        except Exception:
            m.state, m.version = old_state, old_version
            raise
        try:
            export = self._transfer.export_shard(m.source_node_id, shard)
            self._transfer.import_shard(m.target_node_id, shard, export)
            m.source_digest = export.digest
            m.target_digest = self._transfer.verify_shard(m.target_node_id, shard,
                                                         expected_digest=export.digest)
            self._store.put_migration(m)
        except MigrationStateError:
            self._fail(m, "copy failed: migration lifecycle failure")
            raise
        except MigrationVerificationError as e:
            self._fail(m, f"verify failed: {e}")
            raise
        except Exception as e:
            self._fail(m, f"copy failed: {e}")
            raise MigrationStateError(m.migration_id, f"copy failed: {e}") from e
        return replace(m)

    def verify(self, migration_id: str) -> ShardMigration:
        """COPYING -> VERIFYING: re-verify target data digest equals the source digest."""
        m = self._require(migration_id)
        self._require_state(m, MigrationState.COPYING)
        shard = self._assert_source_valid(m)
        old_state, old_version = m.state, m.version
        try:
            m.transition(MigrationState.VERIFYING)
            self._store.put_migration(m)
        except Exception:
            m.state, m.version = old_state, old_version
            raise
        try:
            actual = self._transfer.verify_shard(m.target_node_id, shard, expected_digest=m.source_digest)
            m.target_digest = actual
            self._store.put_migration(m)
        except MigrationVerificationError as e:
            self._fail(m, f"verification failed: {e}")
            raise
        return replace(m)

    def commit(self, migration_id: str) -> ShardMigration:
        """VERIFYING -> COMMITTING -> COMPLETED: the ONLY ownership boundary.

        Ordering (never reversed):
            1. re-validate source + target against the authoritative shard +
               membership (stale migration detection);
            2. atomic authoritative ownership commit (``shard.node_id = target``,
               version bumped, primary-not-in-replicas enforced);
            3. finalize the source storage (best-effort, never rolls ownership back);
            4. mark the migration COMPLETED (or COMPLETED with ``finalized=False`` +
               error when source cleanup failed, so the incomplete-cleanup case is
               distinguishable and never hidden).
        """
        m = self._require(migration_id)
        self._require_state(m, MigrationState.VERIFYING)
        shard = self._assert_source_valid(m)
        # target must STILL be a valid member at commit time
        self._validate_member(m.target_node_id, "target")
        old_state, old_version = m.state, m.version
        try:
            m.transition(MigrationState.COMMITTING)
            self._store.put_migration(m)
        except Exception:
            m.state, m.version = old_state, old_version
            raise

        old_owner, old_replicas, old_version = shard.node_id, list(shard.replicas), shard.version
        new_replicas = [r for r in shard.replicas if r != m.target_node_id]
        shard.node_id = m.target_node_id
        shard.replicas = new_replicas
        shard.version += 1
        try:
            shard.validate_ownership()
            self._membership.validate_replicas(new_replicas)
            self._store.put(shard)
        except MigrationError:
            raise
        except Exception as e:
            # atomic rollback: authoritative ownership remains the last valid owner
            shard.node_id, shard.replicas, shard.version = old_owner, old_replicas, old_version
            self._fail(m, f"ownership commit failed: {e}")
            raise MigrationCommitError(m.namespace, m.shard_id, m.target_node_id, str(e)) from e

        # finalize the source storage ONLY after the target is authoritative
        try:
            self._transfer.finalize_source(m.source_node_id, shard)
            m.finalized = True
        except Exception as e:
            # ownership already moved to the target; a cleanup failure must NOT roll
            # ownership backward. Record it honestly on the migration instead.
            m.error = f"source finalization failed after ownership commit: {e}"
            m.finalized = False
        m.transition(MigrationState.COMPLETED)
        self._store.put_migration(m)
        logger.info(f"migration completed {m.migration_id}: {m.namespace}:{m.shard_id} {old_owner} -> {m.target_node_id}")
        return replace(m)

    def cancel(self, migration_id: str) -> ShardMigration:
        """Cancel a non-terminal migration (discards any target-side imports)."""
        m = self._require(migration_id)
        if m.state.value not in _CANCELABLE:
            raise MigrationStateError(
                m.migration_id, f"cannot cancel a migration in state={m.state.value}"
            )
        shard = self._store.get(m.shard_id)
        if shard is not None and m.state.value in ("COPYING", "VERIFYING"):
            try:
                self._transfer.discard_target(m.target_node_id, shard)
            except Exception as e:
                logger.warning(f"migration {m.migration_id} cancel: target cleanup failed: {e}")
        m.transition(MigrationState.CANCELLED)
        self._store.put_migration(m)
        return replace(m)

    def fail(self, migration_id: str, error: str) -> ShardMigration:
        """Force-fail a non-terminal migration (operational backstop)."""
        m = self._require(migration_id)
        if m.is_terminal():
            raise MigrationStateError(m.migration_id, f"cannot fail a terminal migration (state={m.state.value})")
        return replace(self._fail(m, error))

    # --- read-only rebalancing recommendation (VS-13, explicitly not applied) ----

    def recommend_rebalance(self) -> list[dict]:
        """Deterministic, read-only placement recommendation.

        For each namespace, compare per-owner shard counts among known,
        non-removed members (only ACTIVE/DRAINING shards are considered). When the
        most-loaded owner holds more shards than the least-loaded eligible member,
        recommend moving the first (by shard id) migratable shard of the most-loaded
        owner to the least-loaded eligible member. Deterministic tie-breaking by
        node id / shard id. This NEVER performs a migration.
        """
        known = sorted(
            (n.node_id for n in self._membership.list()
             if n.state.value not in ("REMOVED",)),
        )
        recommendations: list[dict] = []
        namespaces = sorted({s.namespace for s in self._store.all()})

        for ns in namespaces:
            shards = [s for s in self._store.list_by_namespace(ns) if s.state in MIGRATABLE_STATES]
            if not shards:
                continue
            owner_shards: dict[str, list[Shard]] = {}
            for s in shards:
                if s.node_id in known:
                    owner_shards.setdefault(s.node_id, []).append(s)
            if not owner_shards:
                continue
            source_owner = min(owner_shards, key=lambda o: (-len(owner_shards[o]), o))
            source_count = len(owner_shards[source_owner])
            if source_count <= 1:
                continue
            eligible = [n for n in known if n != source_owner]
            if not eligible:
                continue
            target = min(eligible, key=lambda n: (len(owner_shards.get(n, [])), n))
            if len(owner_shards.get(target, [])) >= source_count:
                continue
            candidate = sorted(owner_shards[source_owner], key=lambda s: s.id)[0]
            recommendations.append({
                "namespace": ns,
                "shard_id": candidate.id,
                "current_owner": source_owner,
                "recommended_target": target,
                "reason": (
                    f"current owner {source_owner!r} holds {source_count} shards; "
                    f"candidate target {target!r} holds {len(owner_shards.get(target, []))} "
                    f"(deterministic read-only recommendation; rebalance is NOT automatic)"
                ),
            })
        return recommendations


def _kind_error(kind: str, node_id: str, reason: str):
    if kind == "source":
        return MigrationSourceError(node_id, reason)
    return MigrationTargetError(node_id, reason)
