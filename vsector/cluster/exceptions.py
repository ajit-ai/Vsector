"""VS-11: deterministic cluster membership error semantics.

Every error subclasses ``ValueError`` so the existing REST mapping
(``ValueError`` -> HTTP 400) applies unchanged. Messages are stable,
operator-actionable, and never contain tracebacks or object addresses.
"""
from __future__ import annotations

__all__ = [
    "ClusterMembershipError",
    "UnknownNodeError",
    "DuplicateNodeRegistrationError",
    "RemovedNodeError",
    "InvalidMembershipTransitionError",
    "ClusterIdMismatchError",
]


class ClusterMembershipError(ValueError):
    """Base class for cluster membership failures."""


class UnknownNodeError(ClusterMembershipError):
    def __init__(self, node_id: str, context: str = ""):
        msg = f"unknown node: {node_id!r} is not a known cluster member"
        if context:
            msg = f"{msg} ({context})"
        super().__init__(msg)
        self.node_id = node_id


class DuplicateNodeRegistrationError(ClusterMembershipError):
    def __init__(self, node_id: str):
        super().__init__(f"duplicate node registration: {node_id!r} is already a cluster member")
        self.node_id = node_id


class RemovedNodeError(ClusterMembershipError):
    def __init__(self, node_id: str, context: str = ""):
        msg = f"removed node: {node_id!r} has been removed from the cluster"
        if context:
            msg = f"{msg} ({context})"
        super().__init__(msg)
        self.node_id = node_id


class InvalidMembershipTransitionError(ClusterMembershipError):
    def __init__(self, node_id: str, current: str, target: str):
        super().__init__(
            f"invalid membership transition: {node_id!r} cannot move from {current} to {target}"
        )
        self.node_id = node_id
        self.current = current
        self.target = target


class ClusterIdMismatchError(ClusterMembershipError):
    def __init__(self, node_id: str, expected: str, actual: str):
        super().__init__(
            f"cluster id mismatch: node {node_id!r} belongs to cluster {actual!r}, expected {expected!r}"
        )
        self.node_id = node_id
        self.expected = expected
        self.actual = actual