"""VS-11: ClusterMembershipManager — explicit, durable cluster membership.

Membership is a SEPARATE model from replication health (VS-09) and from shard
lifecycle (VS-10). Membership state never automatically drives replication
health or shard lifecycle; ownership is *validated against* membership: a shard
whose primary owner is not a known cluster member raises a deterministic error
instead of being silently adopted, rerouted, or transferred.

The manager provides deterministic operations over the shared metadata store
(the same etcd abstraction VS-10 uses for shard metadata), persists every
transition, returns fresh copies (never mutable internal state), and rejects
invalid operations without silently changing state.

Automatic failover, leader election, consensus, discovery, replica promotion,
shard migration, and rebalancing are NOT implemented here — they belong to
later phases and this manager must not fabricate any of them.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Iterable

from .exceptions import (
    ClusterIdMismatchError,
    DuplicateNodeRegistrationError,
    RemovedNodeError,
    UnknownNodeError,
)
from .node import ClusterNode, NodeMembershipState

logger = logging.getLogger(__name__)


class ClusterMembershipManager:
    """Validated, durable cluster membership over the shared metadata store."""

    def __init__(self, store, cluster_id: str | None = None):
        self._store = store
        # Deterministic default (stable across restarts, never regenerated);
        # operators pin VSECTOR_CLUSTER_ID for a real deployment.
        self.cluster_id = cluster_id or "vsector-cluster-default"

    # --- reads (fresh copies; no mutable internal state leaks) --------------

    def get(self, node_id: str) -> ClusterNode | None:
        node = self._store.get_node(node_id)
        return None if node is None else replace(node)

    def list(self) -> list[ClusterNode]:
        return [replace(n) for n in self._store.list_nodes()]

    def contains(self, node_id: str) -> bool:
        return self._store.get_node(node_id) is not None

    def membership_state(self, node_id: str) -> str | None:
        node = self._store.get_node(node_id)
        return None if node is None else node.state.value

    def is_available(self, node_id: str) -> bool:
        node = self._store.get_node(node_id)
        return node is not None and node.is_available()

    # --- validation ---------------------------------------------------------

    def validate_node(self, node_id: str) -> None:
        """Require ``node_id`` to be a known, non-removed cluster member.

        Raises ``UnknownNodeError`` for unknown nodes and ``RemovedNodeError``
        for nodes that once were members but are now REMOVED. This is the
        ownership-validation gate: a shard owner must be a known member. A
        DRAINING or JOINING member is still *known* — membership draining never
        automatically revokes ownership (VS-10 lifecycle stays authoritative).
        """
        node = self._store.get_node(node_id)
        if node is None:
            raise UnknownNodeError(node_id)
        if node.state is NodeMembershipState.REMOVED:
            raise RemovedNodeError(node_id)

    def validate_replicas(self, replica_ids: Iterable[str]) -> None:
        """Require every replica to be a known, non-removed member."""
        for rid in replica_ids:
            self.validate_node(rid)

    def validate_cluster_id(self, node_id: str) -> None:
        """Reject operations on a node that belongs to a different cluster."""
        node = self._store.get_node(node_id)
        if node is not None and node.cluster_id != self.cluster_id:
            raise ClusterIdMismatchError(node_id, self.cluster_id, node.cluster_id)

    def validate_cluster_id_for(self, cluster_id: str) -> None:
        """Reject a mismatched cluster identity for a node about to be created."""
        if cluster_id != self.cluster_id:
            raise ClusterIdMismatchError("<new node>", self.cluster_id, cluster_id)

    # --- explicit transitions ----------------------------------------------

    def register(self, node_id: str, cluster_id: str | None = None) -> ClusterNode:
        """Register a JOINING member. Duplicate registration is rejected.

        Membership is never fabricated for arbitrary nodes: every member is
        created here explicitly through this operation (or the startup
        bootstrap helper below).
        """
        if self.contains(node_id):
            raise DuplicateNodeRegistrationError(node_id)
        self.validate_cluster_id_for(cluster_id or self.cluster_id)
        node = ClusterNode(node_id=node_id, cluster_id=self.cluster_id, state=NodeMembershipState.JOINING)
        self._store.put_node(node)
        logger.info(f"cluster member registered: {node_id!r} (cluster={self.cluster_id!r}, state=JOINING)")
        return replace(node)

    def _transition(self, node_id: str, to: NodeMembershipState) -> ClusterNode:
        node = self._store.get_node(node_id)
        if node is None:
            raise UnknownNodeError(node_id)
        self.validate_cluster_id(node_id)
        old_state, old_version = node.state, node.version
        try:
            node.transition(to)
            self._store.put_node(node)
        except Exception:
            node.state, node.version = old_state, old_version
            raise
        return replace(node)

    def activate(self, node_id: str) -> ClusterNode:
        """JOINING -> ACTIVE (or DRAINING -> ACTIVE)."""
        return self._transition(node_id, NodeMembershipState.ACTIVE)

    def begin_drain(self, node_id: str) -> ClusterNode:
        """ACTIVE -> DRAINING: node stops being available for new work."""
        return self._transition(node_id, NodeMembershipState.DRAINING)

    def remove(self, node_id: str) -> ClusterNode:
        """JOINING | ACTIVE | DRAINING -> REMOVED (terminal)."""
        return self._transition(node_id, NodeMembershipState.REMOVED)


def bootstrap_local_node(membership: ClusterMembershipManager, node_id: str) -> ClusterNode:
    """Register/restore the LOCAL node at server startup.

    Deterministic startup semantics:
    - a never-before-seen local node is registered ``JOINING`` then activated
      (``JOINING -> ACTIVE``);
    - a persisted ``ACTIVE`` node is restored ``ACTIVE``;
    - a persisted ``DRAINING`` or ``REMOVED`` node is restored truthfully — it is
      NEVER fabricated back to ``ACTIVE`` by a restart.

    Only the local node is ever registered here; membership for arbitrary nodes
    is never fabricated (there is no distributed discovery in VS-11).
    """
    node = membership.get(node_id)
    if node is None:
        node = membership.register(node_id)
    if node.state is NodeMembershipState.JOINING:
        node = membership.activate(node_id)
    elif node.state is NodeMembershipState.DRAINING:
        logger.info(f"local node {node_id!r} restored as DRAINING (persisted), not fabricated ACTIVE")
    elif node.state is NodeMembershipState.REMOVED:
        logger.info(f"local node {node_id!r} restored as REMOVED (persisted), not fabricated ACTIVE")
    return node