"""VS-12: deterministic placement, ownership-aware routing & local/remote decisions.

A ``RoutingDecision`` is the explicit, immutable description of how a logical
request key maps onto Vsector's existing shard + cluster membership models:

- ``place`` answers *which shard* a record belongs to (deterministic HRW over the
  authoritative shard metadata) and *who owns it* (``shard.node_id``).
- the owner is validated against the VS-11 cluster membership when membership is
  wired — an unknown or REMOVED owner yields an explicit ``UNAVAILABLE`` decision,
  never a silent adoption, reroute, or ownership change.
- ``route_type`` explicitly answers *"is this shard owned by me?"* with
  ``LOCAL`` / ``REMOTE`` / ``UNAVAILABLE``: ``LOCAL`` means the owner is the local
  node, ``REMOTE`` means the owner is a known member on another node (a *decision
  only* — VS-12 does NOT execute cross-node transport), ``UNAVAILABLE`` means the
  shard cannot currently be served per lifecycle/membership rules.

Decisions are side-effect-free: building one never mutates shard state, ownership,
replication health, counters, membership, or persisted metadata. Repeated routing
of the same input while metadata is unchanged produces the same decision.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class RouteType(str, enum.Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable snapshot of a placement/routing decision (no mutable internal state)."""

    namespace: str
    shard_id: str | None
    shard_state: str | None
    owner_node_id: str | None
    local_node_id: str | None
    target: str | None
    route_type: RouteType
    owner_membership_state: str | None = None
    reason: str | None = None

    @property
    def is_local(self) -> bool:
        return self.route_type is RouteType.LOCAL

    @property
    def is_remote(self) -> bool:
        return self.route_type is RouteType.REMOTE

    @property
    def is_unavailable(self) -> bool:
        return self.route_type is RouteType.UNAVAILABLE

    def to_dict(self) -> dict:
        """Deterministic, JSON-serializable fresh copy of the decision."""
        return {
            "namespace": self.namespace,
            "shard_id": self.shard_id,
            "shard_state": self.shard_state,
            "owner_node_id": self.owner_node_id,
            "local_node_id": self.local_node_id,
            "target": self.target,
            "route_type": self.route_type.value,
            "owner_membership_state": self.owner_membership_state,
            "reason": self.reason,
        }