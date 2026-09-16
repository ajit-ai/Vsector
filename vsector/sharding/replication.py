"""VS-09.2: real replication abstraction with failure semantics.

Replaces the simulated ``time.sleep(0.001); return True`` acknowledgement with a
genuine application path: every replicated operation is applied to an
independently represented replica state and the acknowledgement reflects the
actual result.

Model:

    primary durable write (1 ack)
        -> transport -> replica endpoint (independent state) -> ack
        -> result: attempted / acknowledged / failed / required_acks
        -> success / degraded decision

Acknowledgement policy: ``required_acks`` is the TOTAL number of durably-confirmed
writes the operation needs, where the primary's own durable write counts as 1.
Default ``required_acks = 1`` means the primary durable write is sufficient; any
replica failure then surfaces as ``degraded`` success.  When ``required_acks``
exceeds the number of satisfied acknowledgements the operation must report
failure - never fake success.

No Raft / Paxos / leader election / gossip is introduced.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..models.vector_record import VectorRecord
from ..storage.wal import WALEntry

logger = logging.getLogger(__name__)


@dataclass
class ReplicationResult:
    """Structured outcome of replicating one operation to one shard's replicas."""

    primary: str
    attempted_replicas: list[str] = field(default_factory=list)
    acknowledged_replicas: list[str] = field(default_factory=list)
    failed_replicas: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    required_acks: int = 1
    success: bool = True
    degraded: bool = False

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "attempted_replicas": len(self.attempted_replicas),
            "acknowledged_replicas": len(self.acknowledged_replicas),
            "failed_replicas": self.failed_replicas,
            "required_acks": self.required_acks,
            "success": self.success,
            "degraded": self.degraded,
        }


class ReplicaUnavailableError(Exception):
    """Raised when a replica node is deliberately faulted (fault injection)."""

    def __init__(self, node_id: str):
        super().__init__(f"replica unavailable: {node_id}")
        self.node_id = node_id


def _decode(payload: bytes | bytearray | str) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8")
    return payload


def apply_replicated_upsert(ctx, payload: bytes) -> None:
    """Apply an upsert to an independently managed replica context (with WAL durability).

    Idempotent: delivering the same logical record twice (identical checksum and
    version - i.e. the exact same payload replayed) must not create a duplicate.
    A different payload for the same id is a newer version and replaces it
    (latest-wins), matching VS-09.1 recovery semantics.

    ``ctx`` is a ``ShardContext`` with its own WAL / SegmentStore / index object.
    """
    text = _decode(payload)
    raw = json.loads(text)
    if isinstance(raw, dict) and raw.get("__op") == "upsert":
        rec = VectorRecord.model_validate(raw["record"])
    else:
        rec = VectorRecord.model_validate_json(text)

    record_id = str(rec.id)
    existing = ctx.segments.get(record_id)
    if existing is not None and existing.checksum == rec.checksum and existing.version == rec.version:
        return  # identical logical record already applied - idempotent ack

    # Durable on the replica before acking (same payload/image as the primary write)
    ctx.wal.append(WALEntry(payload=rec.model_dump_json().encode()))
    ctx.tombstones.discard(record_id)  # re-insert resurrects a deleted id
    ctx.segments.put(rec)
    ctx.index.add([record_id], np.array([rec.vector], dtype=np.float32), [rec.metadata])
    ctx.lifecycle.notify_write(1)
    ctx.shard.vector_count += 1
    ctx.wal.flush()


def apply_replicated_delete(ctx, payload: bytes) -> None:
    """Apply a delete tombstone to an independently managed replica context.

    Idempotent: the tombstone lives in a set and in the replica WAL; replaying the
    same delete twice keeps the record deleted.  The replica WAL is flushed before
    ack so the tombstone survives a replica restart via VS-09.1 recovery.
    """
    raw = json.loads(_decode(payload))
    record_id = str(raw["id"])
    ctx.tombstones.add(record_id)
    ctx.wal.append(WALEntry(payload=json.dumps({"__op": "delete", "id": record_id}).encode()))
    ctx.segments.memtable.delete(record_id)
    ctx.index.delete([record_id])
    ctx.lifecycle.notify_delete(1)
    ctx.wal.flush()


class ReplicaEndpoint:
    """Independently represented in-process replica for (namespace, shard, node).

    Wraps a dedicated ``ShardContext`` (own WAL dir, own SegmentStore, own index
    object, own tombstone set) so it is never the primary's shared state.
    """

    def __init__(self, node_id: str, ctx):
        self.node_id = node_id
        self.ctx = ctx

    def apply_upsert(self, payload: bytes) -> None:
        apply_replicated_upsert(self.ctx, payload)

    def apply_delete(self, payload: bytes) -> None:
        apply_replicated_delete(self.ctx, payload)

    def flush(self) -> None:
        self.ctx.wal.flush()

    def close(self) -> None:
        self.ctx.wal.close()

    def contains(self, record_id: str) -> bool:
        rec = self.ctx.segments.get(record_id)
        return rec is not None and record_id not in self.ctx.tombstones

    def record(self, record_id: str):
        return self.ctx.segments.get(record_id)

    @property
    def recovered_from_wal(self) -> int:
        return self.ctx.recovery.recovered_from_wal

    @property
    def deleted_ids(self) -> list[str]:
        return list(self.ctx.tombstones)


EndpointProvider = Callable[[str, object, str], ReplicaEndpoint]


class InProcessReplicaTransport:
    """Delivers operations to real replica endpoints and reports actual outcomes.

    Exceptions are never converted into success: a failed application is recorded
    in ``failed_replicas``, logged, and reflected in the success/degraded decision.
    """

    def __init__(self, endpoint_provider: EndpointProvider):
        self.endpoint_provider = endpoint_provider
        self._faults: set[str] = set()

    # --- fault injection (tests) -------------------------------------------
    def fail_node(self, node_id: str) -> None:
        self._faults.add(node_id)

    def heal_node(self, node_id: str) -> None:
        self._faults.discard(node_id)

    def is_faulted(self, node_id: str) -> bool:
        return node_id in self._faults

    # --- replication entry points ------------------------------------------
    def replicate_upsert(self, namespace: str, shard, payload: bytes, required_acks: int = 1) -> ReplicationResult:
        return self._deliver(namespace, shard, payload, required_acks, op="upsert")

    def replicate_delete(self, namespace: str, shard, payload: bytes, required_acks: int = 1) -> ReplicationResult:
        return self._deliver(namespace, shard, payload, required_acks, op="delete")

    def _deliver(self, namespace: str, shard, payload: bytes, required_acks: int, op: str) -> ReplicationResult:
        result = ReplicationResult(primary=shard.node_id or shard.id, required_acks=required_acks)
        for node_id in list(shard.replicas):
            result.attempted_replicas.append(node_id)
            try:
                if self.is_faulted(node_id):
                    raise ReplicaUnavailableError(node_id)
                endpoint = self.endpoint_provider(namespace, shard, node_id)
                if op == "upsert":
                    endpoint.apply_upsert(payload)
                else:
                    endpoint.apply_delete(payload)
                result.acknowledged_replicas.append(node_id)
            except Exception as e:  # surfaced as a failed replica, never as success
                logger.warning(f"replication {op} to {node_id} failed: {e}")
                result.failed_replicas.append(node_id)
                result.failures[node_id] = str(e)

        # the primary's own durable write always counts as 1 ack
        ack_total = 1 + len(result.acknowledged_replicas)
        result.success = ack_total >= required_acks
        result.degraded = bool(result.failed_replicas)
        return result


def aggregate_replication(results: list[ReplicationResult], required_acks: int) -> dict:
    """Merge per-record replication results into one request-level summary."""
    attempted = sorted({r for res in results for r in res.attempted_replicas})
    acknowledged = sorted({r for res in results for r in res.acknowledged_replicas})
    failed = sorted({r for res in results for r in res.failed_replicas})
    return {
        "success": bool(results) and all(res.success for res in results),
        "degraded": any(res.degraded or not res.success for res in results),
        "attempted": len(attempted),
        "acknowledged": len(acknowledged),
        "required_acks": required_acks,
        "failed_replicas": failed,
    }