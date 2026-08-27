"""MODULE 2: Distributed Sharding - HRW + Consistent Hash Ring."""
from __future__ import annotations

import hashlib
import mmh3
from typing import Iterable


def _hash_key(key: str) -> int:
    # mmh3 128-bit -> int
    return mmh3.hash128(key, signed=False)


class RendezvousHash:
    """Highest Random Weight (HRW) for shard assignment.

    shard_key = hash(namespace + record_id)
    score(node) = hash(shard_key + node_id)
    pick max score.
    """

    def __init__(self, nodes: Iterable[str] | None = None):
        self.nodes: list[str] = list(nodes) if nodes else []

    def add_node(self, node: str) -> None:
        if node not in self.nodes:
            self.nodes.append(node)

    def remove_node(self, node: str) -> None:
        if node in self.nodes:
            self.nodes.remove(node)

    def get_node(self, key: str) -> str:
        if not self.nodes:
            raise RuntimeError("no nodes in ring")
        best = None
        best_score = -1
        for n in self.nodes:
            score = _hash_key(f"{key}#{n}")
            if score > best_score:
                best_score = score
                best = n
        assert best is not None
        return best

    def get_nodes(self, key: str, n: int = 1) -> list[str]:
        """Top-N nodes ordered by weight (for replication)."""
        scored = [( _hash_key(f"{key}#{node}"), node) for node in self.nodes]
        scored.sort(reverse=True)
        return [node for _, node in scored[:n]]


class ConsistentHashRing:
    """Consistent hashing with virtual nodes (256 per physical node min)."""

    def __init__(self, replicas: int = 256):
        self.replicas = replicas
        self.ring: dict[int, str] = {}
        self.sorted_keys: list[int] = []

    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        for i in range(self.replicas):
            vnode = f"{node}#{i}"
            h = self._hash(vnode)
            self.ring[h] = node
        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node: str) -> None:
        for i in range(self.replicas):
            vnode = f"{node}#{i}"
            h = self._hash(vnode)
            self.ring.pop(h, None)
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: str) -> str:
        if not self.ring:
            raise RuntimeError("ring empty")
        h = self._hash(key)
        # binary search clockwise
        import bisect
        idx = bisect.bisect(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]
