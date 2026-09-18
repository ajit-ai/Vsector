"""Shard metadata, split algorithm & explicit lifecycle/ownership state.

VS-10 adds an explicit shard lifecycle (``CREATING -> ACTIVE -> DRAINING ->
OFFLINE`` with a ``DRAINING -> ACTIVE`` ownership-change path) and makes the
existing ``node_id``/``replicas`` fields the canonical primary-owner / replica
membership model. Transitions must go through ``Shard.transition`` (or
``ShardLifecycleManager``) so arbitrary state jumps are rejected with a
deterministic error. Legacy enum members ``SPLITTING``/``RETIRED`` are kept for
backward compatibility with the split simulation.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import ClassVar

from .exceptions import InvalidLifecycleTransitionError, NoPrimaryOwnerError, OwnershipConflictError


class ShardState(str, enum.Enum):
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    OFFLINE = "OFFLINE"
    # Legacy members kept for backward compatibility (split simulation / retired shards).
    SPLITTING = "SPLITTING"
    RETIRED = "RETIRED"


@dataclass
class Shard:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = ""
    node_id: str = ""  # canonical primary owner node
    state: ShardState = ShardState.ACTIVE
    vector_count: int = 0
    max_vectors: int = 50_000_000_000
    replicas: list[str] = field(default_factory=list)  # follower nodes (distinct from primary owner)
    created_at: float = field(default_factory=time.time)
    version: int = 1

    # Explicit, minimal lifecycle. Arbitrary jumps are rejected; certain legacy
    # members are accepted only as sources so persisted legacy shards keep working.
    TRANSITIONS: ClassVar[dict[ShardState, frozenset]] = {
        ShardState.CREATING: frozenset({ShardState.ACTIVE, ShardState.OFFLINE}),
        ShardState.ACTIVE: frozenset({ShardState.DRAINING, ShardState.OFFLINE}),
        ShardState.DRAINING: frozenset({ShardState.ACTIVE, ShardState.OFFLINE}),
        ShardState.OFFLINE: frozenset(),  # terminal in VS-10; no silent re-entry
        ShardState.SPLITTING: frozenset({ShardState.RETIRED, ShardState.ACTIVE}),
        ShardState.RETIRED: frozenset(),
    }

    @property
    def primary_owner(self) -> str:
        """The current primary owner node. Single representation: ``node_id``."""
        return self.node_id

    def should_split(self) -> bool:
        return self.vector_count > self.max_vectors

    def transition(self, to: ShardState) -> "Shard":
        """Validate and apply an explicit lifecycle transition (atomic on success).

        Raises ``InvalidLifecycleTransitionError`` for any jump outside the
        declared transition table and leaves the shard untouched in that case.
        """
        allowed = self.TRANSITIONS.get(self.state, frozenset())
        if to not in allowed:
            raise InvalidLifecycleTransitionError(self.namespace, self.id, self.state.value, to.value)
        self.state = to
        self.version += 1
        return self

    def require_owner(self) -> str:
        """Return the primary owner or raise ``NoPrimaryOwnerError``."""
        if not self.node_id:
            raise NoPrimaryOwnerError(self.namespace, self.id)
        return self.node_id

    def validate_ownership(self, owner: str | None = None, replicas: list[str] | None = None) -> "Shard":
        """Validate ownership invariants: exactly one primary owner, primary not in
        its own replica list (else ``OwnershipConflictError``)."""
        owner = self.node_id if owner is None else owner
        members = list(self.replicas) if replicas is None else replicas
        if not owner:
            raise NoPrimaryOwnerError(self.namespace, self.id)
        if owner in members:
            raise OwnershipConflictError(
                self.namespace, self.id, f"primary owner {owner!r} cannot be a replica of the same shard"
            )
        return self

    def ownership(self) -> dict:
        """Deterministic, JSON-serializable, fresh-copy ownership/lifecycle snapshot."""
        return {
            "shard_id": self.id,
            "namespace": self.namespace,
            "state": self.state.value,
            "primary_owner": self.node_id,
            "replicas": list(self.replicas),
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "node_id": self.node_id,
            "primary_owner": self.node_id,
            "state": self.state.value,
            "vector_count": self.vector_count,
            "max_vectors": self.max_vectors,
            "replicas": list(self.replicas),
            "version": self.version,
        }