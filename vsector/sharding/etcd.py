"""Etcd backends: InMemory + real etcd3 with gossip invalidation."""

from __future__ import annotations

import json
import logging
import threading
import time

from .shard import Shard

logger = logging.getLogger(__name__)


class InMemoryEtcd:
    """Minimal etcd abstraction — thread-safe dict."""

    def __init__(self):
        self._data: dict[str, Shard] = {}
        self._lock = threading.RLock()

    def put(self, shard: Shard) -> None:
        with self._lock:
            self._data[shard.id] = shard

    def get(self, shard_id: str) -> Shard | None:
        with self._lock:
            return self._data.get(shard_id)

    def list_by_namespace(self, namespace: str) -> list[Shard]:
        with self._lock:
            return [s for s in self._data.values() if s.namespace == namespace]

    def delete(self, shard_id: str) -> None:
        with self._lock:
            self._data.pop(shard_id, None)

    def all(self) -> list[Shard]:
        with self._lock:
            return list(self._data.values())

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
        return Shard(**{k: v for k, v in d.items() if k in Shard.__dataclass_fields__})

    def list_by_namespace(self, namespace: str) -> list[Shard]:
        self._ensure()
        assert self._client is not None
        out: list[Shard] = []
        for val, _ in self._client.get_prefix("/vsector/shards/"):  # type: ignore
            d = json.loads(val.decode())
            if d.get("namespace") == namespace:
                out.append(Shard(id=d["id"], namespace=d["namespace"], node_id=d.get("node_id", ""), vector_count=d.get("vector_count", 0), max_vectors=d.get("max_vectors", 50_000_000_000)))
        return out

    def delete(self, shard_id: str) -> None:
        self._ensure()
        assert self._client is not None
        self._client.delete(self._key(shard_id))  # type: ignore

    def all(self) -> list[Shard]:
        self._ensure()
        assert self._client is not None
        out: list[Shard] = []
        for val, _ in self._client.get_prefix("/vsector/shards/"):  # type: ignore
            d = json.loads(val.decode())
            out.append(Shard(id=d["id"], namespace=d["namespace"], node_id=d.get("node_id", ""), vector_count=d.get("vector_count", 0)))
        return out

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
