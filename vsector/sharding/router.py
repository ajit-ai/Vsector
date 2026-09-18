"""Shard Router: cached in-memory + gossip invalidation + etcd fallback.

VS-10 adds explicit lifecycle/ownership-aware routing:

- ``route`` resolves namespace -> shard -> the current readable primary shard;
  records always map to the same shard via HRW (identity stays stable across
  lifecycle changes); a shard that is not readable yields a deterministic error
  instead of pretending the record exists elsewhere.
- ``route_write`` additionally enforces the writable lifecycle state (ACTIVE)
  and primary ownership: writes are refused with a deterministic error when the
  shard is CREATING/DRAINING/OFFLINE or when the local node is not the primary
  owner. No write is ever silently forwarded to a non-primary node.
- ``route_for_query`` fans out only over readable shards (ACTIVE / DRAINING).

HRW membership is the full shard set so record placement is stable; lifecycle
checks are applied to the selected shard.

VS-11 owner validation: when the router is wired to a ``ClusterMembershipManager``
(``membership=...``), the primary owner of every routed shard must be a known,
non-removed cluster member.

VS-12 adds the explicit, side-effect-free decision layer (``place``,
``route_read_decision``, ``route_write_decision``, ``query_route``,
``placement_view``): every decision clearly answers *which shard*, *who owns it*,
*is the owner a known member*, and *LOCAL / REMOTE / UNAVAILABLE* relative to the
local node. A REMOTE decision is a routing *decision only* — no cross-node
transport is executed.
"""
from __future__ import annotations

import time
import threading
import hashlib
from typing import Dict, List

from .hash_ring import RendezvousHash
from ..cluster.exceptions import RemovedNodeError, UnknownNodeError
from .exceptions import (
    OwnershipMismatchError,
    PlacementError,
    ShardCreatingError,
    ShardDrainingError,
    ShardOfflineError,
)
from .placement import RouteType, RoutingDecision
from .shard import Shard, ShardState

# Abstract etcd - in-memory for single-node; pluggable for distributed
# Real implementation moved to vsector.sharding.etcd (Etcd3Store + make_etcd_store)
from .etcd import EtcdStore  # re-export for backward compat
# (InMemoryEtcd alias)


READABLE_STATES = frozenset({ShardState.ACTIVE, ShardState.DRAINING})


def _state_error(shard: Shard) -> ShardCreatingError | ShardDrainingError | ShardOfflineError:
    """Deterministic per-state error for a non-writable/-readable shard."""
    if shard.state is ShardState.CREATING:
        return ShardCreatingError(shard.namespace, shard.id)
    if shard.state is ShardState.DRAINING:
        return ShardDrainingError(shard.namespace, shard.id)
    return ShardOfflineError(shard.namespace, shard.id, state=shard.state.value)


class ShardRouter:
    def __init__(self, etcd: EtcdStore | None = None, cache_ttl_s: int = 5, membership=None):
        self.etcd = etcd or EtcdStore()
        self.cache_ttl = cache_ttl_s
        # Optional VS-11 cluster membership: when provided, every routed shard's
        # primary owner must be a known (non-removed) cluster member. When absent
        # (self-contained unit tests / host-mode), ownership is not cluster-checked.
        self.membership = membership
        self._cache: Dict[str, tuple[float, List[Shard]]] = {}
        self._cache_lock = threading.RLock()
        self._hrw_cache: Dict[str, RendezvousHash] = {}

    def _shard_key(self, namespace: str, record_id: str) -> str:
        return hashlib.sha256(f"{namespace}:{record_id}".encode()).hexdigest()

    def _get_shards_cached(self, namespace: str) -> List[Shard]:
        now = time.time()
        with self._cache_lock:
            if namespace in self._cache:
                ts, shards = self._cache[namespace]
                if now - ts < self.cache_ttl:
                    return shards
        # cache miss -> etcd fallback
        shards = self.etcd.list_by_namespace(namespace)
        with self._cache_lock:
            self._cache[namespace] = (now, shards)
        return shards

    def invalidate(self, namespace: str) -> None:
        """Gossip invalidation hook."""
        with self._cache_lock:
            self._cache.pop(namespace, None)
            self._hrw_cache.pop(namespace, None)

    def _select(self, namespace: str, record_id: str) -> Shard:
        """HRW selection over the full shard set (stable identity per record)."""
        shards = self._get_shards_cached(namespace)
        if not shards:
            raise KeyError(f"no shards for namespace {namespace!r}")
        active = [s for s in shards if s.id]
        # HRW over shard ids
        hrw = self._hrw_cache.get(namespace)
        if hrw is None:
            hrw = RendezvousHash([s.id for s in active])
            self._hrw_cache[namespace] = hrw
        else:
            # sync membership
            current_ids = set(s.id for s in active)
            hrw_ids = set(hrw.nodes)
            if current_ids != hrw_ids:
                hrw = RendezvousHash([s.id for s in active])
                self._hrw_cache[namespace] = hrw
        key = self._shard_key(namespace, record_id)
        shard_id = hrw.get_node(key)
        for s in active:
            if s.id == shard_id:
                return s
        return active[0]

    def _validate_owner(self, shard: Shard) -> None:
        """VS-11 ownership check: the primary owner must be a known cluster member.

        Raises a deterministic cluster error (``UnknownNodeError`` /
        ``RemovedNodeError``) when the owner is not a known, non-removed member.
        No shard adoption, forwarding, or automatic transfer is ever attempted.
        """
        if self.membership is not None:
            self.membership.validate_node(shard.require_owner())

    def route(self, namespace: str, record_id: str) -> Shard:
        """Resolve a record to its readable primary shard (does not fake availability)."""
        shard = self._select(namespace, record_id)
        if shard.state not in READABLE_STATES:
            raise _state_error(shard)
        self._validate_owner(shard)
        return shard

    def route_write(self, namespace: str, record_id: str, owner_node_id: str | None = None) -> Shard:
        """Resolve a write to its primary-owner shard, enforcing lifecycle + ownership.

        Raises a deterministic error when the shard is CREATING / DRAINING /
        OFFLINE, when its primary owner is not a known cluster member (VS-11),
        or when this node is not the primary owner. Writes are never silently
        rerouted to a different shard or forwarded to a replica.
        """
        shard = self._select(namespace, record_id)
        if shard.state is not ShardState.ACTIVE:
            raise _state_error(shard)
        self._validate_owner(shard)
        owner = shard.require_owner()
        if owner_node_id is not None and owner != owner_node_id:
            raise OwnershipMismatchError(namespace, shard.id, owner, owner_node_id)
        return shard

    def route_for_query(self, namespace: str) -> List[Shard]:
        """Fan-out to readable shards only (ACTIVE / DRAINING)."""
        return [s for s in self._get_shards_cached(namespace) if s.state in READABLE_STATES]

    def writable(self, namespace: str, owner_node_id: str | None = None) -> List[Shard]:
        """All ACTIVE, locally-owned shards for a namespace; refuse otherwise."""
        out: List[Shard] = []
        shards = self._get_shards_cached(namespace)
        if not shards:
            return out
        for shard in shards:
            if shard.state is not ShardState.ACTIVE:
                raise _state_error(shard)
            owner = shard.require_owner()
            if owner_node_id is not None and owner != owner_node_id:
                raise OwnershipMismatchError(namespace, shard.id, owner, owner_node_id)
            out.append(shard)
        return out

    # --- VS-12 deterministic placement & local/remote routing decisions ------

    def _select_or_raise(self, namespace: str, record_id: str) -> Shard:
        """Deterministic HRW selection with a defined ``PlacementError`` contract."""
        try:
            return self._select(namespace, record_id)
        except KeyError as e:
            raise PlacementError(namespace, str(e)) from None

    def _decision_for(self, namespace: str, shard: Shard, local_node_id: str | None,
                      mode: str = "any") -> RoutingDecision:
        """Build a side-effect-free decision for a selected shard.

        ``mode``: "any" (placement only), "read" (lifecycle-readable), or
        "write" (writable/ACTIVE). Lifecycle, ownership, replication health,
        membership, and persisted metadata are never mutated.
        """
        owner = shard.node_id or None
        state_value = shard.state.value
        reason: str | None = None
        owner_state: str | None = None

        if mode == "read":
            if shard.state not in READABLE_STATES:
                reason = f"shard state={state_value} is not readable (requires ACTIVE/DRAINING)"
        elif mode == "write":
            if shard.state is not ShardState.ACTIVE:
                reason = (
                    f"shard state={state_value} is not writable (requires ACTIVE); "
                    "DRAINING/OFFLINE/CREATING never become writable silently"
                )
        if reason is None and owner is None:
            reason = "shard has no primary owner"
        if reason is None:
            if self.membership is not None:
                try:
                    self.membership.validate_node(owner)
                    owner_state = self.membership.membership_state(owner)
                except UnknownNodeError:
                    owner_state = "unknown"
                    reason = f"primary owner {owner!r} is not a known cluster member"
                except RemovedNodeError:
                    owner_state = "REMOVED"
                    reason = (
                        f"primary owner {owner!r} is REMOVED from the cluster; "
                        "ownership is never transferred automatically"
                    )

        if reason is not None:
            route_type = RouteType.UNAVAILABLE
        elif local_node_id is not None and owner == local_node_id:
            route_type = RouteType.LOCAL
        else:
            route_type = RouteType.REMOTE

        return RoutingDecision(
            namespace=namespace,
            shard_id=shard.id,
            shard_state=state_value,
            owner_node_id=owner,
            local_node_id=local_node_id,
            target=owner,
            route_type=route_type,
            owner_membership_state=owner_state,
            reason=reason,
        )

    def place(self, namespace: str, record_id: str, local_node_id: str | None = None) -> RoutingDecision:
        """Deterministic placement decision: which shard + owner handles this record."""
        shard = self._select_or_raise(namespace, record_id)
        return self._decision_for(namespace, shard, local_node_id, mode="any")

    def route_read_decision(self, namespace: str, record_id: str, local_node_id: str | None = None) -> RoutingDecision:
        """Lifecycle-aware read routing decision (readable shard states only)."""
        shard = self._select_or_raise(namespace, record_id)
        return self._decision_for(namespace, shard, local_node_id, mode="read")

    def route_write_decision(self, namespace: str, record_id: str, local_node_id: str | None = None) -> RoutingDecision:
        """Lifecycle- and ownership-aware write routing decision (ACTIVE + owner)."""
        shard = self._select_or_raise(namespace, record_id)
        return self._decision_for(namespace, shard, local_node_id, mode="write")

    def query_route(self, namespace: str, local_node_id: str | None = None) -> list[RoutingDecision]:
        """Deterministic multi-shard read fan-out: one decision per readable shard."""
        shards = self._get_shards_cached(namespace)
        return [self._decision_for(namespace, s, local_node_id, mode="read")
                for s in shards if s.state in READABLE_STATES]

    def shard_placement(self, shard: Shard, local_node_id: str | None = None) -> dict:
        """Placement metadata (fresh copies) for one shard, membership-aware."""
        owner = shard.node_id or None
        owner_state: str | None = None
        reason: str | None = None
        route_type = RouteType.LOCAL if (local_node_id is not None and owner == local_node_id) else RouteType.REMOTE
        if self.membership is not None:
            if owner is None:
                reason = "shard has no primary owner"
                route_type = RouteType.UNAVAILABLE
            else:
                try:
                    self.membership.validate_node(owner)
                    owner_state = self.membership.membership_state(owner)
                except UnknownNodeError:
                    owner_state = "unknown"
                    reason = "primary owner is not a known cluster member"
                    route_type = RouteType.UNAVAILABLE
                except RemovedNodeError:
                    owner_state = "REMOVED"
                    reason = "primary owner is REMOVED from the cluster"
                    route_type = RouteType.UNAVAILABLE
        return {
            "shard_id": shard.id,
            "state": shard.state.value,
            "primary_owner": owner,
            "replicas": list(shard.replicas),
            "owner_membership_state": owner_state,
            "route_type": route_type.value,
            "target": owner,
            "reason": reason,
        }

    def placement_view(self, namespace: str, local_node_id: str | None = None) -> dict:
        """Fresh-copy placement metadata for a namespace (no mutable internals)."""
        return {
            "namespace": namespace,
            "local_node_id": local_node_id,
            "shards": [self.shard_placement(s, local_node_id)
                       for s in self.etcd.list_by_namespace(namespace)],
        }

    # --- Shard Split Algorithm (zero-downtime) ---
    def maybe_split(self, shard: Shard, on_split) -> List[Shard] | None:
        """Detect > threshold, lock, copy halves, update routing atomically."""
        if not shard.should_split():
            return None
        # 1. Detect
        # 2. Lock shard for writes (caller should redirect to WAL only)
        # 3. Copy halves via callback
        # 4. Update routing atomically
        # 5. Drain WAL
        # 6. Retire parent
        # Simplified in-memory simulation:
        shard_a = Shard(namespace=shard.namespace, node_id=shard.node_id + "-a", vector_count=shard.vector_count // 2, max_vectors=shard.max_vectors)
        shard_b = Shard(namespace=shard.namespace, node_id=shard.node_id + "-b", vector_count=shard.vector_count - shard_a.vector_count, max_vectors=shard.max_vectors)
        if on_split:
            on_split(shard, shard_a, shard_b)
        self.etcd.put(shard_a)
        self.etcd.put(shard_b)
        # retire parent after drain
        self.etcd.delete(shard.id)
        self.invalidate(shard.namespace)
        return [shard_a, shard_b]

    def register_shard(self, shard: Shard) -> None:
        self.etcd.put(shard)
        self.invalidate(shard.namespace)