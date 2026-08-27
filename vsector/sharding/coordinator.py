"""Replication Coordinator & Compaction Service stubs."""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class ReplicationCoordinator:
    """Quorum-based replication (async, quorum ack). Primary -> 2 followers."""

    def __init__(self, replication_factor: int = 3, regions: list[str] | None = None):
        self.replication_factor = replication_factor
        self.followers: list[str] = []
        # multi-region extension (opt-in via VSECTOR_REGIONS)
        try:
            from .multiregion import MultiRegionCoordinator

            self.multiregion = MultiRegionCoordinator(regions=regions, replication_factor=replication_factor)
        except Exception:
            self.multiregion = None  # type: ignore

    def replicate_async(self, payload: bytes, followers: list[str] | None = None) -> bool:
        targets = followers or self.followers[: self.replication_factor - 1]
        # simulate async quorum ack (always success in single-node)
        logger.debug(f"Replicating to {targets}")
        time.sleep(0.001)
        # also replicate cross-region async best-effort (non-blocking)
        try:
            if self.multiregion:
                import asyncio

                # namespace unknown here — caller can use multiregion directly for ns-aware ship
                pass
        except Exception:
            pass
        return True  # quorum ack


class CompactionService:
    """Background SSTable -> Index merge + fragmentation cleanup."""

    def __init__(self, interval_s: int = 6 * 3600):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, callback):
        def loop():
            while not self._stop.wait(self.interval_s):
                try:
                    callback()
                except Exception as e:
                    logger.error(f"compaction failed: {e}")
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
