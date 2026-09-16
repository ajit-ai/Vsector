"""VS-10: explicit shard lifecycle & ownership transitions.

Lifecycle: ``CREATING -> ACTIVE -> DRAINING -> OFFLINE``, with the ownership-change
sequence ``ACTIVE -> DRAINING -> (owner change) -> ACTIVE``. Every transition is
validated before it is applied and persisted through the shard store, so an
invalid operation leaves the shard untouched and raises a deterministic error.
If persistence fails, the in-memory shard is rolled back so no partially applied
state is ever observed by other service operations (the service runs on a single
asyncio event loop; transitions are logically atomic within it).

Ownership changes are stateful and explicit: primary ownership may only be
reassigned while the shard is ``DRAINING``; the change is observable through
stats/API and the shard must be re-activated with an explicit transition.

Automatic failover, consensus, and live migration are NOT implemented here.
"""
from __future__ import annotations

import logging
from typing import Iterable

from .exceptions import (
    InvalidLifecycleTransitionError,
    NoPrimaryOwnerError,
    OwnershipConflictError,
)
from .shard import Shard, ShardState

logger = logging.getLogger(__name__)


class ShardLifecycleManager:
    """Validated lifecycle/ownership operations over a shard store (etcd)."""

    def __init__(self, store, membership=None):
        self._store = store
        # Optional VS-11 cluster membership: when provided, primary owners (and
        # new owners on ownership change) must be known cluster members. In
        # self-contained unit usage (no membership), VS-10 behavior is unchanged.
        self._membership = membership

    # --- creation -----------------------------------------------------------

    def create(self, shard: Shard, owner: str | None = None, replicas: Iterable[str] | None = None) -> Shard:
        """Register a new shard in ``CREATING`` state with an explicit primary owner."""
        if owner is not None:
            shard.node_id = owner
        if replicas is not None:
            shard.replicas = list(replicas)
        shard.validate_ownership()
        if self._membership is not None:
            self._membership.validate_node(shard.node_id)
        shard.state = ShardState.CREATING
        self._store.put(shard)
        return shard

    # --- validated transitions ----------------------------------------------

    def transition(self, shard: Shard, to: ShardState) -> Shard:
        """Apply a validated lifecycle transition and persist it atomically."""
        old_state, old_version = shard.state, shard.version
        try:
            shard.transition(to)
            self._store.put(shard)
        except Exception:
            shard.state = old_state
            shard.version = old_version
            raise
        return shard

    def activate(self, shard: Shard) -> Shard:
        """CREATING -> ACTIVE (or DRAINING -> ACTIVE after an ownership change)."""
        return self.transition(shard, ShardState.ACTIVE)

    def begin_drain(self, shard: Shard) -> Shard:
        """ACTIVE -> DRAINING: stop new writes; reads remain served."""
        return self.transition(shard, ShardState.DRAINING)

    def retire(self, shard: Shard) -> Shard:
        """DRAINING -> OFFLINE: terminal; normal reads/writes refuse it."""
        return self.transition(shard, ShardState.OFFLINE)

    # --- ownership changes (stateful, explicit) -----------------------------

    def change_owner(self, shard: Shard, new_owner: str, replicas: list[str] | None = None) -> Shard:
        """Reassign primary ownership. Requires the shard to be ``DRAINING``.

        Ownership is never reassigned while the shard is silently ``ACTIVE``; no
        automatic/fabricated handoff is possible. State stays ``DRAINING`` until
        the caller completes the transition with ``activate()``.
        """
        if shard.state is not ShardState.DRAINING:
            raise InvalidLifecycleTransitionError(
                shard.namespace,
                shard.id,
                shard.state.value,
                "ownership change (owner reassignment requires state=DRAINING)",
            )
        if not new_owner:
            raise NoPrimaryOwnerError(shard.namespace, shard.id)
        if self._membership is not None:
            self._membership.validate_node(new_owner)
        new_replicas = list(shard.replicas) if replicas is None else list(replicas)
        if new_owner in new_replicas:
            raise OwnershipConflictError(
                shard.namespace, shard.id, f"primary owner {new_owner!r} cannot be a replica of the same shard"
            )
        old_owner, old_replicas = shard.node_id, list(shard.replicas)
        try:
            shard.node_id = new_owner
            shard.replicas = new_replicas
            shard.version += 1
            self._store.put(shard)
        except Exception:
            shard.node_id = old_owner
            shard.replicas = old_replicas
            raise
        logger.info(f"ownership transfer {shard.namespace}:{shard.id} {old_owner} -> {new_owner} (state=DRAINING)")
        return shard