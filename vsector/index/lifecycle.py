"""Index Lifecycle Manager - states BUILDING -> READY -> DEGRADED -> COMPACTING -> READY."""
from __future__ import annotations

import enum
import time
import threading
import logging

logger = logging.getLogger(__name__)


class IndexState(str, enum.Enum):
    BUILDING = "BUILDING"  # initial load or shard migration
    READY = "READY"  # serving queries
    DEGRADED = "DEGRADED"  # replica failure, reduced replication
    COMPACTING = "COMPACTING"  # background optimization (still serving)


class IndexLifecycleManager:
    def __init__(self, index, delete_ratio_threshold: float = 0.10, fragmentation_threshold: float = 0.3, interval_hours: int = 6):
        self.index = index
        self.state = IndexState.BUILDING
        self.delete_count = 0
        self.total_count = 0
        self.fragmentation = 0.0
        self.delete_ratio_threshold = delete_ratio_threshold
        self.fragmentation_threshold = fragmentation_threshold
        self.interval_hours = interval_hours
        self._last_compaction = time.time()
        self._lock = threading.RLock()

    def transition(self, to: IndexState):
        with self._lock:
            logger.info(f"Lifecycle {self.state} -> {to}")
            self.state = to

    def mark_ready(self):
        self.transition(IndexState.READY)

    def mark_degraded(self):
        self.transition(IndexState.DEGRADED)

    def mark_building(self):
        self.transition(IndexState.BUILDING)

    def notify_write(self, n: int = 1):
        with self._lock:
            self.total_count += n

    def notify_delete(self, n: int = 1):
        with self._lock:
            self.delete_count += n
            self.total_count = max(0, self.total_count - n)

    def should_compact(self) -> bool:
        with self._lock:
            if self.total_count == 0:
                return False
            delete_ratio = self.delete_count / max(1, self.total_count + self.delete_count)
            time_trigger = (time.time() - self._last_compaction) > self.interval_hours * 3600
            return delete_ratio > self.delete_ratio_threshold or self.fragmentation > self.fragmentation_threshold or time_trigger

    def maybe_compact(self):
        if self.should_compact() and self.state == IndexState.READY:
            self.transition(IndexState.COMPACTING)
            try:
                if hasattr(self.index, "compact"):
                    self.index.compact()
                else:
                    logger.info("Compact simulated")
                self.delete_count = 0
                self.fragmentation = 0.0
                self._last_compaction = time.time()
            finally:
                self.transition(IndexState.READY)
