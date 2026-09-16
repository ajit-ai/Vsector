"""VS-10: deterministic shard lifecycle & ownership error semantics.

Every error subclasses ``ValueError`` so the existing REST mapping
(``ValueError`` -> HTTP 400) applies unchanged. Messages are stable,
operator-actionable, and never contain tracebacks or object addresses.
"""
from __future__ import annotations

__all__ = [
    "ShardLifecycleError",
    "ShardNotFoundError",
    "ShardCreatingError",
    "ShardDrainingError",
    "ShardOfflineError",
    "NoPrimaryOwnerError",
    "OwnershipMismatchError",
    "OwnershipConflictError",
    "InvalidLifecycleTransitionError",
]


class ShardLifecycleError(ValueError):
    """Base class for shard lifecycle/ownership failures."""


class ShardNotFoundError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str):
        super().__init__(f"shard not found: {namespace}:{shard_id}")
        self.namespace = namespace
        self.shard_id = shard_id


class ShardCreatingError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str):
        super().__init__(
            f"shard creating: {namespace}:{shard_id} is not ready for writes yet (state=CREATING)"
        )
        self.namespace = namespace
        self.shard_id = shard_id


class ShardDrainingError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str):
        super().__init__(
            f"shard draining: {namespace}:{shard_id} does not accept new writes (state=DRAINING)"
        )
        self.namespace = namespace
        self.shard_id = shard_id


class ShardOfflineError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str, state: str = "OFFLINE"):
        super().__init__(f"shard offline: {namespace}:{shard_id} is unavailable (state={state})")
        self.namespace = namespace
        self.shard_id = shard_id


class NoPrimaryOwnerError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str):
        super().__init__(f"no primary owner: {namespace}:{shard_id} has no primary owner assigned")
        self.namespace = namespace
        self.shard_id = shard_id


class OwnershipMismatchError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str, owner: str, local_node: str):
        super().__init__(
            f"ownership mismatch: {namespace}:{shard_id} primary owner {owner!r} != local node {local_node!r}; "
            "writes are refused rather than routed to a non-primary node"
        )
        self.namespace = namespace
        self.shard_id = shard_id
        self.owner = owner
        self.local_node = local_node


class OwnershipConflictError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str, reason: str):
        super().__init__(f"ownership conflict: {namespace}:{shard_id}: {reason}")
        self.namespace = namespace
        self.shard_id = shard_id


class InvalidLifecycleTransitionError(ShardLifecycleError):
    def __init__(self, namespace: str, shard_id: str, current: str, target: str):
        super().__init__(
            f"invalid lifecycle transition: {namespace}:{shard_id} cannot move from {current} to {target}"
        )
        self.namespace = namespace
        self.shard_id = shard_id
        self.current = current
        self.target = target