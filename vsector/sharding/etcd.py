"""Etcd backends: InMemory + real etcd3 with gossip invalidation.

VS-10 adds optional JSON durability to the in-memory backend
(``InMemoryEtcd(path=...)``): shard identity, lifecycle state, and ownership
metadata are persisted so they survive a process restart. Shard objects are
reconstructed from persisted dicts with proper ``ShardState`` coercion (never a
raw string in a typed field) and never rely on object memory addresses.

VS-11 persists cluster membership (``ClusterNode`` records) through the SAME
store, under a reserved key prefix, so there is one coherent metadata model:
identity, membership, and shard ownership survive a restart together.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from ..cluster.node import ClusterNode, NodeMembershipState
from .shard import Shard, ShardState

logger = logging.getLogger(__name__)

_DEFAULT_MAX_VECTORS = 50_000_000_000

# Reserved key prefix for cluster membership records inside the shard store.
_CLUSTER_KEY_PREFIX = "cluster::node::"


def _node_from_dict(d: dict) -> ClusterNode:
    try:
        state = NodeMembershipState(str(d.get("membership_state", NodeMembershipState.ACTIVE.value)))
    except ValueError:
        logger.warning(f"cluster node {d.get('node_id', '?')}: unknown persisted state {d.get('membership_state')!r}; defaulting to ACTIVE")
        state = NodeMembershipState.ACTIVE
    return ClusterNode(
        node_id=d.get("node_id", ""),
        cluster_id=d.get("cluster_id", ""),
        state=state,
        version=int(d.get("version", 1)),
    )


def _state_from(value, shard_id: str) -> ShardState:
    if isinstance(value, ShardState):
        return value
    try:
        return ShardState(str(value))
    except ValueError:
        logger.warning(f"shard {shard_id}: unknown persisted state {value!r}; defaulting to ACTIVE")
        return ShardState.ACTIVE


def _shard_from_dict(d: dict) -> Shard:
    return Shard(
        id=d.get("id"),
        namespace=d.get("namespace", ""),
        node_id=d.get("node_id", "") or d.get("primary_owner", ""),
        state=_state_from(d.get("state", ShardState.ACTIVE.value), d.get("id", "")),
        vector_count=d.get("vector_count", 0),
        max_vectors=d.get("max_vectors", _DEFAULT_MAX_VECTORS),
        replicas=list(d.get("replicas", [])),
        created_at=d.get("created_at", time.time()),
        version=d.get("version", 1),
    )


class InMemoryEtcd:
    """Minimal etcd abstraction — thread-safe dict, optionally JSON-durable."""

    def __init__(self, path: str | None = None):
        self._data: dict[str, Shard] = {}
        self._nodes: dict[str, ClusterNode] = {}
        self._lock = threading.RLock()
        self.path = path
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
            for k, v in raw.items():
                if k.startswith(_CLUSTER_KEY_PREFIX):
                    self._nodes[k[len(_CLUSTER_KEY_PREFIX):]] = _node_from_dict(v)
                else:
                    self._data[k] = _shard_from_dict(v)
        except Exception as e:
            logger.warning(f"InMemoryEtcd load failed for {self.path}: {e}")

    def _persist(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {k: v.to_dict() for k, v in self._data.items()}
            data.update({f"{_CLUSTER_KEY_PREFIX}{n.node_id}": n.to_dict() for n in self._nodes.values()})
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"InMemoryEtcd persist failed for {self.path}: {e}")

    def put(self, shard: Shard) -> None:
        with self._lock:
            self._data[shard.id] = shard
        self._persist()

    def get(self, shard_id: str) -> Shard | None:
        with self._lock:
            return self._data.get(shard_id)

    def list_by_namespace(self, namespace: str) -> list[Shard]:
        with self._lock:
            return [s for s in self._data.values() if s.namespace == namespace]

    def delete(self, shard_id: str) -> None:
        with self._lock:
            self._data.pop(shard_id, None)
        self._persist()

    def all(self) -> list[Shard]:
        with self._lock:
            return list(self._data.values())

    # --- cluster membership (VS-11) -----------------------------------------

    def put_node(self, node: ClusterNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node
        self._persist()

    def get_node(self, node_id: str) -> ClusterNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    def list_nodes(self) -> list[ClusterNode]:
        with self._lock:
            return list(self._nodes.values())

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)
        self._persist()

    def watch_prefix(self, prefix: str, callback):
        """No-op for in-mem — caller polls."""
        return None


class Etcd3Store:
    """Real etcd3 via etcd3 / etcd3-py client. Stores shard JSON at /vsector/shards/{id}."""

    def __init__(self, endpoints: str = "localhost:2379", timeout: int = 5):
        self.endpoints = [e.strip() for e in endpoints.split(",") if e.strip()]
        self.timeout = timeout
        self._client = None
        self._lock = threading.RLock()
        self._connect()

    def _connect(self):
        try:
            import etcd3  # type: ignore

            host, port = self.endpoints[0].split(":") if ":" in self.endpoints[0] else (self.endpoints[0], "2379")
            self._client = etcd3.client(host=host, port=int(port), timeout=self.timeout)
            # probe
            self._client.status()
            logger.info(f"Etcd3 connected {self.endpoints[0]}")
        except Exception as e:
            logger.warning(f"etcd3 connect failed, will retry on demand: {e}")
            self._client = None

    def _ensure(self):
        if self._client is None:
            self._connect()
        if self._client is None:
            raise RuntimeError("etcd3 not available")

    def _key(self, shard_id: str) -> str:
        return f"/vsector/shards/{shard_id}"

    def put(self, shard: Shard) -> None:
        self._ensure()
        assert self._client is not None
        self._client.put(self._key(shard.id), json.dumps(shard.to_dict()))  # type: ignore

    def get(self, shard_id: str) -> Shard | None:
        self._ensure()
        assert self._client is not None
        val, _ = self._client.get(self._key(shard_id))  # type: ignore
        if not val:
            return None
        d = json.loads(val.decode())
        return _shard_from_dict(d)

    def _list_dicts(self) -> list[dict]:
        self._ensure()
        assert self._client is not None
        out: list[dict] = []
        for val, _ in self._client.get_prefix("/vsector/shards/"):  # type: ignore
            out.append(json.loads(val.decode()))
        return out

    def list_by_namespace(self, namespace: str) -> list[Shard]:
        return [s for s in (_shard_from_dict(d) for d in self._list_dicts()) if s.namespace == namespace]

    def delete(self, shard_id: str) -> None:
        self._ensure()
        assert self._client is not None
        self._client.delete(self._key(shard_id))  # type: ignore

    def all(self) -> list[Shard]:
        return [_shard_from_dict(d) for d in self._list_dicts()]

    # --- cluster membership (VS-11) -----------------------------------------

    def _node_key(self, node_id: str) -> str:
        return f"/vsector/cluster/nodes/{node_id}"

    def _list_node_dicts(self) -> list[dict]:
        self._ensure()
        assert self._client is not None
        out: list[dict] = []
        for val, _ in self._client.get_prefix("/vsector/cluster/nodes/"):  # type: ignore
            out.append(json.loads(val.decode()))
        return out

    def put_node(self, node: ClusterNode) -> None:
        self._ensure()
        assert self._client is not None
        self._client.put(self._node_key(node.node_id), json.dumps(node.to_dict()))  # type: ignore

    def get_node(self, node_id: str) -> ClusterNode | None:
        self._ensure()
        assert self._client is not None
        val, _ = self._client.get(self._node_key(node_id))  # type: ignore
        if not val:
            return None
        return _node_from_dict(json.loads(val.decode()))

    def list_nodes(self) -> list[ClusterNode]:
        return [_node_from_dict(d) for d in self._list_node_dicts()]

    def delete_node(self, node_id: str) -> None:
        self._ensure()
        assert self._client is not None
        self._client.delete(self._node_key(node_id))  # type: ignore

    def watch_prefix(self, prefix: str, callback):
        self._ensure()
        assert self._client is not None
        try:
            events = self._client.watch_prefix(prefix)  # type: ignore
            # etcd3 watch returns iterator; run in thread
            def _watch():
                for ev in events:
                    try:
                        callback(ev)
                    except Exception as e:
                        logger.warning(f"etcd watch callback error: {e}")

            t = threading.Thread(target=_watch, daemon=True)
            t.start()
            return t
        except Exception as e:
            logger.warning(f"etcd watch failed: {e}")
            return None


def make_etcd_store(endpoints: str | None = None) -> InMemoryEtcd | Etcd3Store:
    """Factory: real etcd3 if VSECTOR_ETCD_ENDPOINTS reachable, else in-mem."""
    import os

    eps = endpoints or os.getenv("VSECTOR_ETCD_ENDPOINTS") or os.getenv("ETCD_ENDPOINTS") or ""
    if eps:
        try:
            # quick probe without importing etcd3 at import time
            store = Etcd3Store(eps)
            if store._client is not None:
                return store
        except Exception:
            pass
    return InMemoryEtcd()


# Backward-compat alias used by router/tests
EtcdStore = InMemoryEtcd

__all__ = ["InMemoryEtcd", "Etcd3Store", "make_etcd_store", "EtcdStore"]