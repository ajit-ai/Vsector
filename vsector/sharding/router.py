"""Shard Router: cached in-memory + gossip invalidation + etcd fallback."""
from __future__ import annotations

import time
import threading
import hashlib
from typing import Dict, List

from .hash_ring import RendezvousHash
from .shard import Shard

# Abstract etcd - in-memory for single-node; pluggable for distributed
# Real implementation moved to vsector.sharding.etcd (Etcd3Store + make_etcd_store)
from .etcd import EtcdStore  # re-export for backward compat
# (InMemoryEtcd alias)


class ShardRouter:
    def __init__(self, etcd: EtcdStore | None = None, cache_ttl_s: int = 5):
        self.etcd = etcd or EtcdStore()
        self.cache_ttl = cache_ttl_s
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

    def route(self, namespace: str, record_id: str) -> Shard:
        shards = self._get_shards_cached(namespace)
        if not shards:
            raise KeyError(f"no shards for namespace {namespace!r}")
        active = [s for s in shards if s.id]  # all
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

    def route_for_query(self, namespace: str) -> List[Shard]:
        """Fan-out to all shards for query (or filtered by HRW if needed)."""
        return self._get_shards_cached(namespace)

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
        shard.state = shard.state  # would be SPLITTING -> RETIRED
        # retire parent after drain
        self.etcd.delete(shard.id)
        self.invalidate(shard.namespace)
        return [shard_a, shard_b]

    def register_shard(self, shard: Shard) -> None:
        self.etcd.put(shard)
        self.invalidate(shard.namespace)
