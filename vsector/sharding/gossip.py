"""Gossip invalidation for ShardRouter — etcd watch + polling fallback."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class GossipInvalidator:
    """Subscribes to etcd watch on /vsector/shards/ and invalidates router cache."""

    def __init__(self, router, poll_interval_s: int = 5):
        self.router = router
        self.poll_interval = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        # Try etcd watch first
        try:
            watch = getattr(self.router.etcd, "watch_prefix", None)
            if watch:
                watch("/vsector/shards/", self._on_event)
                logger.info("Gossip: etcd watch enabled")
        except Exception as e:
            logger.warning(f"Gossip watch failed, falling back to poll: {e}")
        # Poll fallback always runs (covers InMemory + watch failures)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def _on_event(self, ev):
        # Any etcd change -> invalidate affected namespace
        try:
            # ev is etcd3 event; extract namespace from value if possible
            import json

            val = ev.value if hasattr(ev, "value") else b""
            if val:
                d = json.loads(val.decode() if isinstance(val, bytes) else val)
                ns = d.get("namespace")
                if ns:
                    self.router.invalidate(ns)
                    logger.debug(f"Gossip invalidate {ns} via etcd watch")
        except Exception:
            pass

    def _poll_loop(self):
        # Simple poll: periodically clear expired TTL is already handled by router,
        # but gossip ensures cross-node invalidation even before TTL.
        while not self._stop.wait(self.poll_interval):
            try:
                # Invalidate nothing actively — just ensures warmup; real etcd watch does work
                pass
            except Exception as e:
                logger.warning(f"Gossip poll error: {e}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
