"""WAL: Append-only binary log, segmented at 256MB, GROUP_COMMIT.

Entry: [length:4B][checksum:4B][timestamp:8B][payload:NB]
Retention 7 days.
"""
from __future__ import annotations

import os
import struct
import time
import zlib
import threading
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

HEADER_FMT = ">IIQ"  # length, checksum, timestamp
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass
class WALEntry:
    payload: bytes
    timestamp: int = 0  # ns

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time_ns()


class WAL:
    def __init__(self, dir_path: str | Path, segment_bytes: int = 256 * 1024 * 1024, group_commit_batch: int = 1000, group_commit_ms: int = 5, retention_days: int = 7):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.segment_bytes = segment_bytes
        self.group_commit_batch = group_commit_batch
        self.group_commit_ms = group_commit_ms
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._buffer: list[WALEntry] = []
        self._cond = threading.Condition(self._lock)
        self._segment_id = 0
        self._segment_file = None
        self._segment_written = 0
        self._closed = False
        self._flusher = threading.Thread(target=self._group_commit_loop, daemon=True)
        self._flusher.start()
        self._open_segment()

    def _open_segment(self):
        if self._segment_file:
            self._segment_file.close()
        path = self.dir / f"wal-{self._segment_id:06d}.log"
        # O_DIRECT best-effort on Linux (flag 0x4000), fallback to buffered on Windows/macOS/BSD
        flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
        try:
            flags |= os.O_DIRECT  # type: ignore  # Linux only
            fd = os.open(str(path), flags, 0o644)
            self._segment_file = os.fdopen(fd, "ab", buffering=0)  # unbuffered for O_DIRECT
            logger.debug("WAL opened with O_DIRECT")
        except Exception as e:
            logger.debug(f"WAL O_DIRECT not available, using buffered: {e}")
            self._segment_file = open(path, "ab")
        self._segment_written = 0 if not path.exists() else path.stat().st_size
        # S3 replication hook (async best-effort) — set VSECTOR_S3_BUCKET to enable
        self._s3_replicate(path)

    def _rotate_if_needed(self, entry_size: int):
        if self._segment_written + entry_size > self.segment_bytes:
            self._segment_id += 1
            self._open_segment()

    def append(self, entry: WALEntry) -> None:
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.group_commit_batch:
                self._cond.notify()

    def _group_commit_loop(self):
        while not self._closed:
            with self._lock:
                self._cond.wait(timeout=self.group_commit_ms / 1000.0)
                if not self._buffer:
                    continue
                batch = self._buffer[:]
                self._buffer.clear()
            self._flush_batch(batch)

    def _flush_batch(self, batch: list[WALEntry]):
        for e in batch:
            payload = e.payload
            checksum = zlib.crc32(payload) & 0xFFFFFFFF
            header = struct.pack(HEADER_FMT, len(payload), checksum, e.timestamp)
            data = header + payload
            self._rotate_if_needed(len(data))
            assert self._segment_file is not None
            self._segment_file.write(data)
            self._segment_written += len(data)
        assert self._segment_file is not None
        self._segment_file.flush()
        try:
            os.fsync(self._segment_file.fileno())
        except Exception:
            pass
        logger.debug(f"WAL flushed {len(batch)} entries")

    def flush(self):
        with self._lock:
            batch = self._buffer[:]
            self._buffer.clear()
        if batch:
            self._flush_batch(batch)

    def read_all(self) -> list[WALEntry]:
        self.flush()
        entries: list[WALEntry] = []
        for p in sorted(self.dir.glob("wal-*.log")):
            with open(p, "rb") as f:
                data = f.read()
            off = 0
            while off + HEADER_SIZE <= len(data):
                length, checksum, ts = struct.unpack_from(HEADER_FMT, data, off)
                off += HEADER_SIZE
                if off + length > len(data):
                    break
                payload = data[off: off + length]
                off += length
                if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
                    logger.warning("WAL checksum mismatch, skipping")
                    continue
                entries.append(WALEntry(payload=payload, timestamp=ts))
        return entries

    def _s3_replicate(self, path: Path):
        """Best-effort S3 replication of closed segments — no-op if bucket not set."""
        try:
            import os

            if os.getenv("VSECTOR_S3_BUCKET"):
                from .s3 import S3Tier

                tier = S3Tier()
                if tier.enabled:
                    # replicate in background thread to not block WAL
                    threading.Thread(target=tier.upload, args=(path,), daemon=True).start()
        except Exception:
            pass

    def gc(self):
        """Retention 7 days."""
        cutoff = time.time() - self.retention_days * 86400
        for p in self.dir.glob("wal-*.log"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)

    def close(self):
        self._closed = True
        self.flush()
        if self._segment_file:
            self._segment_file.close()
