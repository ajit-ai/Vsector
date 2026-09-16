"""VS-11: explicit, durable cluster membership for Vsector deployments.

``cluster_id`` names the cluster, ``node_id`` identifies a member (the same
identity VS-10 uses for shard ownership), and membership state is an explicit,
validated lifecycle (JOINING -> ACTIVE -> DRAINING/REMOVED). Membership is a
separate model from replication health (VS-09) and shard lifecycle (VS-10).
"""
from .node import NodeMembershipState, ClusterNode
from .membership import ClusterMembershipManager, bootstrap_local_node
from .exceptions import (
    ClusterMembershipError,
    UnknownNodeError,
    DuplicateNodeRegistrationError,
    RemovedNodeError,
    InvalidMembershipTransitionError,
    ClusterIdMismatchError,
)

__all__ = [
    "NodeMembershipState",
    "ClusterNode",
    "ClusterMembershipManager",
    "bootstrap_local_node",
    "ClusterMembershipError",
    "UnknownNodeError",
    "DuplicateNodeRegistrationError",
    "RemovedNodeError",
    "InvalidMembershipTransitionError",
    "ClusterIdMismatchError",
]