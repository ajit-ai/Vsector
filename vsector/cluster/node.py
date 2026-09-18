"""VS-11: cluster node identity & explicit membership state model.

A ``ClusterNode`` is the minimal durable record for a member of a Vsector
deployment: its stable ``node_id``, the ``cluster_id`` it belongs to, and its
membership-state lifecycle. ``node_id`` is the SAME identity VS-10 already uses
for shard ownership (``shard.node_id``); ``cluster_id`` names the cluster and is
deliberately a different identity from both ``node_id`` and ``shard_id`` — the
three are never conflated.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import ClassVar

from .exceptions import InvalidMembershipTransitionError


class NodeMembershipState(str, enum.Enum):
    JOINING = "JOINING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    REMOVED = "REMOVED"


@dataclass
class ClusterNode:
    node_id: str
    cluster_id: str
    state: NodeMembershipState = NodeMembershipState.JOINING
    version: int = 1

    # Explicit, minimal membership lifecycle. Arbitrary jumps are rejected with a
    # deterministic error; REMOVED is terminal (no silent re-entry).
    TRANSITIONS: ClassVar[dict[NodeMembershipState, frozenset]] = {
        NodeMembershipState.JOINING: frozenset({NodeMembershipState.ACTIVE, NodeMembershipState.REMOVED}),
        NodeMembershipState.ACTIVE: frozenset({NodeMembershipState.DRAINING, NodeMembershipState.REMOVED}),
        NodeMembershipState.DRAINING: frozenset({NodeMembershipState.ACTIVE, NodeMembershipState.REMOVED}),
        NodeMembershipState.REMOVED: frozenset(),  # terminal; no silent re-entry
    }

    def transition(self, to: NodeMembershipState) -> "ClusterNode":
        """Validate and apply an explicit membership transition (atomic on success)."""
        allowed = self.TRANSITIONS.get(self.state, frozenset())
        if to not in allowed:
            raise InvalidMembershipTransitionError(self.node_id, self.state.value, to.value)
        self.state = to
        self.version += 1
        return self

    def is_available(self) -> bool:
        """Membership availability: ACTIVE and DRAINING nodes still serve.

        This is a membership property only — it carries no replication-health
        meaning (VS-09) and never drives shard lifecycle (VS-10).
        """
        return self.state in (NodeMembershipState.ACTIVE, NodeMembershipState.DRAINING)

    def to_dict(self) -> dict:
        """Deterministic, JSON-serializable, fresh-copy membership snapshot."""
        return {
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "membership_state": self.state.value,
            "version": self.version,
        }