"""Segment Store: MemTable -> SSTable (tiered)."""
from __future__ import annotations

import os
import json
import threading
import time
from pathlib import Path
from typing import Dict, List

from ..models.vector_record import VectorRecord


class MemTable:
    """In-memory write buffer."""

    def __init__(self, flush_threshold: int = 10000):
        self.flush_threshold = flush_threshold
        self._data: Dict[str, VectorRecord] = {}
        self._lock = threading.RLock()

    def put(self, record: VectorRecord) -> None:
        with self._lock:
            self._data[str(record.id)] = record

    def get(self, id: str) -> VectorRecord | None:
        with self._lock:
            return self._data.get(id)

    def delete(self, id: str) -> None:
        with self._lock:
            self._data.pop(id, None)

    def scan(self) -> List[VectorRecord]:
        with self._lock:
            return list(self._data.values())

    def count(self) -> int:
        with self._lock:
            return len(self._data)

    def should_flush(self) -> bool:
        return self.count() >= self.flush_threshold

    def snapshot_and_clear(self) -> List[VectorRecord]:
        with self._lock:
            snap = list(self._data.values())
            self._data.clear()
            return snap


class SSTable:
    """Immutable on-disk segment (JSONL for simplicity, tiered to S3 in prod)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def flush(cls, records: List[VectorRecord], dir_path: Path) -> "SSTable":
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = dir_path / f"sst-{ts}-{os.getpid()}.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
        return cls(path)

    def scan(self) -> List[VectorRecord]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    out.append(VectorRecord.model_validate_json(line))
        return out

    def count(self) -> int:
        if not self.path.exists():
            return 0
        c = 0
        with open(self.path, "r") as f:
            for _ in f:
                c += 1
        return c


class SegmentStore:
    """Tiered: MemTable -> SSTable (hot) -> S3 cold tier (fallback to local cold_dir)."""

    def __init__(self, base_dir: str | Path, memtable_threshold: int = 10000, s3_bucket: str | None = None):
        self.base_dir = Path(base_dir)
        self.hot_dir = self.base_dir / "hot"
        self.cold_dir = self.base_dir / "cold"
        self.memtable = MemTable(flush_threshold=memtable_threshold)
        self._sstables: List[SSTable] = []
        self._lock = threading.Lock()
        # S3 cold tier (optional)
        try:
            from .s3 import S3Tier

            self.s3 = S3Tier(bucket=s3_bucket)
        except Exception:
            self.s3 = None  # type: ignore
        # load existing
        for d in [self.hot_dir, self.cold_dir]:
            if d.exists():
                for p in sorted(d.glob("sst-*.jsonl")):
                    self._sstables.append(SSTable(p))

    def put(self, record: VectorRecord) -> None:
        self.memtable.put(record)
        if self.memtable.should_flush():
            self.flush()

    def flush(self) -> SSTable | None:
        if self.memtable.count() == 0:
            return None
        snap = self.memtable.snapshot_and_clear()
        sst = SSTable.flush(snap, self.hot_dir)
        with self._lock:
            self._sstables.append(sst)
        # S3 cold offload (async best-effort) — upload hot SSTable to S3
        try:
            if getattr(self, "s3", None) and getattr(self.s3, "enabled", False):
                self.s3.upload(sst.path)  # type: ignore
        except Exception:
            pass
        return sst

    def tier_to_cold(self, sst: SSTable) -> None:
        """Move SSTable from hot to cold (local or S3)."""
        try:
            dest = self.cold_dir / sst.path.name
            sst.path.rename(dest)
            sst.path = dest
            if getattr(self, "s3", None) and getattr(self.s3, "enabled", False):
                self.s3.upload(dest)  # type: ignore
        except Exception:
            pass

    def get(self, id: str) -> VectorRecord | None:
        v = self.memtable.get(id)
        if v:
            return v
        # scan sstables newest first
        with self._lock:
            for sst in reversed(self._sstables):
                # naive linear scan per sstable; use bloom filter in prod
                for r in sst.scan():
                    if str(r.id) == id:
                        return r
        return None

    def scan_all(self) -> List[VectorRecord]:
        out = self.memtable.scan()
        with self._lock:
            for sst in self._sstables:
                out.extend(sst.scan())
        return out

    def count(self) -> int:
        c = self.memtable.count()
        with self._lock:
            for sst in self._sstables:
                c += sst.count()
        return c
