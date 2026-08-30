"""Tiered cache: L1 in-mem + L2 Redis (optional) for query + routing.

Uses VSECTOR_REDIS_URL (redis://...). Falls back to L1 only if Redis unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class TieredCache:
    def __init__(self, redis_url: str | None = None, ttl_s: int = 60, namespace: str = "vsector"):
        self.ttl = ttl_s
        self.namespace = namespace
        self._l1: dict[str, tuple[float, str]] = {}
        self._redis = None
        url = redis_url or os.getenv("VSECTOR_REDIS_URL") or ""
        if url:
            try:
                import redis  # type: ignore

                self._redis = redis.from_url(url, socket_timeout=2, decode_responses=True)
                self._redis.ping()
                logger.info(f"TieredCache Redis connected {url.split('@')[-1]}")
            except Exception as e:
                logger.warning(f"Redis unavailable, L1 only: {e}")
                self._redis = None

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def _hash_query(self, namespace: str, vector: list[float], top_k: int, filters: dict | None) -> str:
        h = hashlib.sha256()
        h.update(namespace.encode())
        h.update(str(top_k).encode())
        h.update(json.dumps(vector[:8], sort_keys=True).encode())  # sample for key
        h.update(json.dumps(filters or {}, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def get_query(self, namespace: str, vector: list[float], top_k: int, filters: dict | None) -> dict | None:
        key = self._hash_query(namespace, vector, top_k, filters)
        # L1
        now = time.time()
        if key in self._l1:
            ts, val = self._l1[key]
            if now - ts < self.ttl:
                return json.loads(val)
            else:
                self._l1.pop(key, None)
        # L2 Redis
        if self._redis:
            try:
                v = self._redis.get(self._key(f"q:{key}"))
                if v:
                    self._l1[key] = (now, v)  # warm L1
                    return json.loads(v)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")
        return None

    def set_query(self, namespace: str, vector: list[float], top_k: int, filters: dict | None, result: dict) -> None:
        key = self._hash_query(namespace, vector, top_k, filters)
        payload = json.dumps(result)
        self._l1[key] = (time.time(), payload)
        if self._redis:
            try:
                self._redis.setex(self._key(f"q:{key}"), self.ttl, payload)
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

    def invalidate_namespace(self, namespace: str) -> None:
        # Clear L1 keys for namespace (brute clear for simplicity)
        self._l1.clear()
        if self._redis:
            try:
                for k in self._redis.scan_iter(match=self._key(f"q:*")):
                    # In prod, namespace-scoped keys would use proper prefix
                    self._redis.delete(k)
            except Exception:
                pass


# Global singleton for QueryEngine
_tiered_cache: TieredCache | None = None


def get_cache() -> TieredCache:
    global _tiered_cache
    if _tiered_cache is None:
        _tiered_cache = TieredCache()
    return _tiered_cache
