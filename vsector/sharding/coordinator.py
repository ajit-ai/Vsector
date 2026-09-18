"""Replication Coordination & Compaction Service.

VS-09.2 removes the simulated ``time.sleep(...) ; return True`` acknowledgement.
The real application path lives in ``vsector/sharding/replication.py`` and is
driven directly by the ingest service.  ``ReplicationCoordinator`` is kept as a
thin backward-compatible shell that delegates to the real transport when one is
bound, and never fabricates acknowledgements.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ReplicationCoordinator:
    """Thin coordinator over a real replica transport (no fake acknowledgements).

    The active write path uses ``InProcessReplicaTransport`` directly.  This class
    exists for backward compatibility: it only accepts a transport and forwards
    operations to it.  Without a transport it refuses to claim success.
    """

    def __init__(self, replication_factor: int = 3, regions: list[str] | None = None, transport=None):
        self.replication_factor = replication_factor
        self.regions = regions or []
        self.transport = transport
        self.multiregion = None  # type: ignore

    def _require_transport(self):
        if self.transport is None:
            raise RuntimeError(
                "ReplicationCoordinator has no transport; use IngestService.replicator "
                "(InProcessReplicaTransport) for real replication semantics"
            )
        return self.transport

    def replicate_upsert(self, namespace: str, shard, payload: bytes, required_acks: int = 1):
        return self._require_transport().replicate_upsert(namespace, shard, payload, required_acks=required_acks)

    def replicate_delete(self, namespace: str, shard, payload: bytes, required_acks: int = 1):
        return self._require_transport().replicate_delete(namespace, shard, payload, required_acks=required_acks)


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