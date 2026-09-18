"""Startup recovery: resolve durable SegmentStore state + WAL replay into one logical state and rebuild the serving index.

Model:
    segment state (SSTables)        -- durably flushed snapshots (MemTable is empty on a fresh process)
    + WAL records (chronological)   -- every write: upserts + delete tombstones
    -> logical current state        -- per-id latest version, deletes applied
    -> serving index rebuild        -- every final record added to the index exactly once

WAL ordering is authoritative for recency: entries are read back in append order
(segment files sorted, in-file order preserved), so the LAST entry for an id wins.
This makes update-after-flush and delete-after-flush resolve correctly without
having to guess whether an individual WAL record was already flushed to an SSTable.

Recovery is idempotent by construction: it is a pure read of the durable on-disk
state (SSTables + WAL segment files) and only repopulates the in-memory MemTable
for records that are not yet durably represented on disk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..models.vector_record import VectorRecord
from .wal import WAL
from .segment import SegmentStore

logger = logging.getLogger(__name__)

DELETE = "delete"
UPSERT = "upsert"


@dataclass
class RecoveryResult:
    records: list[VectorRecord] = field(default_factory=list)  # final logical records (deleted excluded)
    deleted_ids: list[str] = field(default_factory=list)  # ids removed by tombstones
    recovered_from_wal: int = 0  # final records that existed only in WAL (not durably in any SSTable)
    wal_entries_read: int = 0


def _parse_payload(payload: bytes):
    """Return (UPSERT, VectorRecord) or (DELETE, id), or None for unparseable entries.

    Supports the current envelope format ({"__op": "delete", "id": ...}) as well as
    legacy raw VectorRecord JSON payloads written by earlier releases, so persisted
    WAL files remain readable.
    """
    try:
        data = json.loads(payload)
    except Exception:
        logger.warning("recovery: corrupt WAL payload (not JSON), skipping")
        return None
    if not isinstance(data, dict):
        return None
    if data.get("__op") == DELETE:
        _id = data.get("id")
        if isinstance(_id, str) and _id:
            return (DELETE, _id)
        return None
    if data.get("__op") == UPSERT:
        rec_data = data.get("record")
        if not isinstance(rec_data, dict):
            return None
        data = rec_data
    try:
        rec = VectorRecord.model_validate(data)
    except Exception:
        logger.warning("recovery: invalid VectorRecord in WAL payload, skipping")
        return None
    return (UPSERT, rec)


def recover(wal: WAL, segments: SegmentStore) -> RecoveryResult:
    """Resolve durable on-disk state into the current logical state.

    1. Start from the SSTable snapshot (durably flushed records).
    2. Apply WAL entries in chronological order: upserts overwrite, deletes tombstone.
    3. Records that are not durably represented in any SSTable are re-added to the
       MemTable so they are served and can be flushed on the next cycle.
    """
    sstable = segments.durable_records_by_id()

    records: dict[str, VectorRecord] = dict(sstable)
    tombstones: set[str] = set()

    wal_entries = wal.read_all()
    for entry in wal_entries:
        parsed = _parse_payload(entry.payload)
        if parsed is None:
            continue
        op, obj = parsed
        if op == DELETE:
            _id = obj  # type: ignore[assignment]
            tombstones.add(_id)
            records.pop(_id, None)
        else:
            rec = obj  # type: ignore[assignment]
            tombstones.discard(str(rec.id))  # a later upsert resurrects the record
            records[str(rec.id)] = rec

    # Collect records that are NOT yet durably represented (absent from SSTables or
    # updated to a newer version since the last flush). These are the writes that
    # existed only in the WAL and must be re-established in the MemTable.
    recovered: list[VectorRecord] = []
    for _id, rec in records.items():
        cached = sstable.get(_id)
        if cached is None or (cached.version, cached.checksum) != (rec.version, rec.checksum):
            recovered.append(rec)

    for rec in recovered:
        segments.memtable.put(rec)

    result = RecoveryResult(
        records=list(records.values()),
        deleted_ids=sorted(tombstones),
        recovered_from_wal=len(recovered),
        wal_entries_read=len(wal_entries),
    )
    if wal_entries:
        logger.info(
            "recovery: %d final records, %d deleted, %d recovered from WAL, %d WAL entries",
            len(result.records),
            len(result.deleted_ids),
            result.recovered_from_wal,
            len(wal_entries),
        )
    return result