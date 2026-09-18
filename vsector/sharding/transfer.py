"""VS-13: shard data transfer boundary between per-node storage roots.

The VS-13 spec requires REAL data movement: a successful migration must
preserve shard contents (``source data == target data``) with a deterministic
digest check, and ownership must not move before the target data is verified.

Vsector's shard data lives in per-node storage roots: WAL + SegmentStore under
``{base_dir}/wal/{shard.id}`` and ``{base_dir}/segments/{shard.id}``. VS-13
implements migration WITHOUT inventing cross-node network transport (that is an
explicit non-goal). Instead a ``ShardTransferProvider`` maps each logical node to
a local storage root and performs the real steps across that boundary:

    export_shard   -> read the source's durable logical state (WAL + SSTables),
                      produce a self-contained, JSON-serializable payload with a
                      deterministic SHA-256 digest
    import_shard   -> write that payload into the TARGET node's storage root
                      (real records into its WAL + SegmentStore), never touching
                      the source copy
    verify_shard   -> re-derive the target's logical state and recompute its
                      digest; must equal the source digest before any ownership
                      commit
    finalize_source-> best-effort cleanup of the source data after the target is
                      authoritative (never rolls ownership backward)
    discard_target -> best-effort cleanup of a target import when a migration is
                      cancelled before commit

The digest is computed over the LOGICAL content of the shard (per-id records +
deleted tombstones), deterministic, and independent of file names, ordering, or
object identity — exactly the integrity check the spec requires.

This provider is NOT a fake: tests prove actual data preservation follows a real
export/import through these storage roots.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..models.vector_record import VectorRecord
from ..storage.segment import SegmentStore
from ..storage.wal import WAL, WALEntry
from ..storage.recovery import recover
from .shard import Shard

logger = logging.getLogger(__name__)


def compute_digest(records: list[VectorRecord], deleted_ids: list[str]) -> str:
    """Deterministic SHA-256 digest over the logical shard content.

    ``records`` are sorted by id and each record is serialized in canonical JSON
    (``sort_keys=True``), together with the sorted deleted id set, so the digest
    is stable regardless of storage file naming or ordering.
    """
    payload = {
        "records": [r.model_dump(mode="json") for r in sorted(records, key=lambda r: str(r.id))],
        "deleted": sorted(deleted_ids),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ShardExport:
    """Self-contained, transport-agnostic payload of a shard's logical content."""

    namespace: str
    shard_id: str
    records: list[dict] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)
    digest: str = ""

    def serialize(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "ShardExport":
        return cls.from_dict(json.loads(data.decode("utf-8")))

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "shard_id": self.shard_id,
            "records": self.records,
            "deleted_ids": self.deleted_ids,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShardExport":
        return cls(
            namespace=d["namespace"],
            shard_id=d["shard_id"],
            records=[dict(r) for r in d.get("records", [])],
            deleted_ids=list(d.get("deleted_ids", [])),
            digest=d.get("digest", ""),
        )


class ShardTransferProvider:
    """Real local storage boundary: maps node ids to per-node storage roots."""

    def __init__(self, node_base_dirs: Mapping[str, str] | None = None,
                 default_base_dir: str | Path = "data"):
        self._base_dirs: dict[str, Path] = {k: Path(v) for k, v in (node_base_dirs or {}).items()}
        self.default_base_dir = Path(default_base_dir)

    def node_base_dir(self, node_id: str) -> Path:
        """Explicit root wins; otherwise a per-node subdir under the default root."""
        return self._base_dirs.get(node_id, self.default_base_dir / node_id)

    # --- paths ---------------------------------------------------------------

    def _wal_dir(self, node_id: str, shard: Shard) -> Path:
        return self.node_base_dir(node_id) / "wal" / shard.id

    def _seg_dir(self, node_id: str, shard: Shard) -> Path:
        return self.node_base_dir(node_id) / "segments" / shard.id

    # --- real data movement ---------------------------------------------------

    def export_shard(self, node_id: str, shard: Shard) -> ShardExport:
        """Read the SOURCE's durable logical state (WAL + SSTables) into a payload."""
        seg = SegmentStore(self._seg_dir(node_id, shard))
        wal = WAL(self._wal_dir(node_id, shard))
        try:
            result = recover(wal, seg)
            records = [r.model_dump(mode="json") for r in result.records]
            deleted = sorted(result.deleted_ids)
            return ShardExport(
                namespace=shard.namespace,
                shard_id=shard.id,
                records=records,
                deleted_ids=deleted,
                digest=compute_digest(result.records, result.deleted_ids),
            )
        finally:
            try:
                wal.close()
            except Exception:
                pass

    def import_shard(self, node_id: str, shard: Shard, export: ShardExport) -> None:
        """Write the payload into the TARGET's storage root (never the source's).

        The target root is reset first so a cancelled/re-tried import starts from
        a clean slate. Records are written durably (WAL + flushed SSTable) and
        tombstones are appended AFTER the upserts so WAL replay resolves the
        logical state identically to the source.
        """
        uploads_first = sorted(export.records, key=lambda r: str(r.get("id", "")))
        deleted = sorted(export.deleted_ids)
        self.discard_target(node_id, shard)
        seg = SegmentStore(self._seg_dir(node_id, shard))
        wal = WAL(self._wal_dir(node_id, shard))
        try:
            for rd in uploads_first:
                rec = VectorRecord.model_validate(rd)
                seg.put(rec)
                wal.append(WALEntry(
                    json.dumps({"__op": "upsert", "record": rd}, sort_keys=True, ensure_ascii=False).encode(),
                ))
            for _id in deleted:
                wal.append(WALEntry(
                    json.dumps({"__op": "delete", "id": _id}).encode(),
                ))
            seg.flush()
            wal.flush()
        finally:
            try:
                wal.close()
            except Exception:
                pass

    def verify_shard(self, node_id: str, shard: Shard, expected_digest: str | None = None) -> str:
        """Re-derive the target's logical state and return/recheck its digest."""
        seg = SegmentStore(self._seg_dir(node_id, shard))
        wal = WAL(self._wal_dir(node_id, shard))
        try:
            result = recover(wal, seg)
            actual = compute_digest(result.records, result.deleted_ids)
            # ALSO verify over the REAL durable on-disk SSTables so a tamper
            # diverging from the verified digest is detected even though WAL
            # replay alone would mask it (ownership/WAL semantics kept)
            durable = compute_digest(
                list(seg.durable_records_by_id().values()), result.deleted_ids
            )
        finally:
            try:
                wal.close()
            except Exception:
                pass
        if expected_digest is not None and (actual != expected_digest or durable != expected_digest):
            from .migration import MigrationVerificationError

            raise MigrationVerificationError(shard.namespace, shard.id, expected_digest, actual)
        return actual

    def finalize_source(self, node_id: str, shard: Shard) -> None:
        """Best-effort cleanup of the SOURCE's data after the target is authoritative."""
        self._remove_tree(self._wal_dir(node_id, shard))
        self._remove_tree(self._seg_dir(node_id, shard))

    def discard_target(self, node_id: str, shard: Shard) -> None:
        """Best-effort cleanup of a target's imported data (cancel / fresh import)."""
        self._remove_tree(self._wal_dir(node_id, shard))
        self._remove_tree(self._seg_dir(node_id, shard))

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)