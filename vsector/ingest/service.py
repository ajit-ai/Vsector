"""MODULE 4: Write Path - Ingest Pipeline

Flow: Client -> API Gateway -> Ingest Service (validate) -> WAL Writer (durable, fsync)
      -> Shard Router -> Primary Shard -> MemTable -> Replicate (async quorum) -> ACK
      Background: MemTable -> SSTable flush -> Index merge
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import numpy as np

from ..models.vector_record import VectorRecord
from ..models.namespace import Namespace
from ..storage.wal import WAL, WALEntry
from ..storage.segment import SegmentStore
from ..storage.recovery import recover
from ..storage.metadata import MetadataStore
from ..sharding.router import ShardRouter, EtcdStore
from ..sharding.shard import Shard
from ..sharding.coordinator import ReplicationCoordinator
from ..index.factory import create_index
from ..index.lifecycle import IndexLifecycleManager
from ..infra.metrics import INGEST_COUNTER

logger = logging.getLogger(__name__)


class ShardContext:
    """Per-shard state: segment, index, lifecycle, wal."""

    def __init__(self, shard: Shard, namespace: Namespace, base_dir: str):
        self.shard = shard
        self.namespace = namespace
        wal_dir = f"{base_dir}/wal/{shard.id}"
        seg_dir = f"{base_dir}/segments/{shard.id}"
        self.wal = WAL(wal_dir)
        self.segments = SegmentStore(seg_dir)
        self.index = create_index(namespace.index_type.value, dimension=namespace.dimension, metric=namespace.distance_metric.value)
        self.lifecycle = IndexLifecycleManager(self.index)
        # Single authoritative recovery path: resolve durable SSTable + WAL state into
        # the current logical state and rebuild the index from it exactly once.
        self.recovery = recover(self.wal, self.segments)
        self.tombstones: set[str] = set(self.recovery.deleted_ids)
        for rec in self.recovery.records:
            self.index.add([str(rec.id)], np.array([rec.vector], dtype=np.float32), [rec.metadata])
            self.lifecycle.notify_write(1)
            self.shard.vector_count += 1
        self.lifecycle.mark_ready()


class IngestService:
    def __init__(self, metadata: MetadataStore, router: ShardRouter, base_dir: str = "./data"):
        self.metadata = metadata
        self.router = router
        self.base_dir = base_dir
        self.replicator = ReplicationCoordinator()
        self._shards: dict[str, ShardContext] = {}
        self._idempotency: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def _get_or_create_ctx(self, namespace: str, shard: Shard) -> ShardContext:
        key = f"{namespace}:{shard.id}"
        if key not in self._shards:
            ns = self.metadata.get(namespace)
            if not ns:
                raise ValueError(f"namespace {namespace!r} not found")
            self._shards[key] = ShardContext(shard, ns, self.base_dir)
        return self._shards[key]

    async def upsert(self, namespace: str, records: list[dict[str, Any]], idempotency_key: str | None = None) -> dict[str, Any]:
        t0 = time.time()
        # idempotency
        if idempotency_key and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]

        ns = self.metadata.get(namespace)
        if not ns:
            raise ValueError(f"namespace {namespace!r} not found")
        if len(records) > 10000:
            raise ValueError("Max batch size 10,000")

        upserted: list[str] = []
        errors: list[dict] = []
        touched: dict[str, ShardContext] = {}

        for raw in records:
            try:
                # validate schema
                vec = raw.get("vector")
                if vec is None:
                    raise ValueError("vector missing")
                if len(vec) != ns.dimension:
                    raise ValueError(f"dimension mismatch expected {ns.dimension} got {len(vec)}")
                rec = VectorRecord(
                    id=uuid.UUID(raw["id"]) if "id" in raw else uuid.uuid4(),
                    namespace=namespace,
                    vector=[float(x) for x in vec],
                    dimension=ns.dimension,
                    metadata=raw.get("metadata", {}),
                    tags=raw.get("tags", []),
                    source_system=raw.get("source_system", "unknown"),
                )
                # shard routing (HRW)
                shard = self.router.route(namespace, str(rec.id))
                ctx = self._get_or_create_ctx(namespace, shard)
                # WAL durable write: append to group-commit buffer, batch-flush once per shard below
                # Legacy-compatible payload: keep raw VectorRecord JSON for upserts so existing
                # persisted WAL files and the historical parser both work.
                payload = rec.model_dump_json().encode()
                ctx.wal.append(WALEntry(payload=payload))
                touched[f"{namespace}:{shard.id}"] = ctx
                ctx.tombstones.discard(str(rec.id))  # re-insert resurrects a previously deleted id
                # Shard Router -> Primary Shard Node -> MemTable
                ctx.segments.put(rec)
                # Replicate to 2 followers (async quorum ack)
                self.replicator.replicate_async(payload, shard.replicas)
                # Index merge (in-memory buffer -> index; background SSTable flush handled inside segments)
                # Incremental build: insert directly into live index
                ctx.index.add([str(rec.id)], np.array([rec.vector], dtype=np.float32), [rec.metadata])
                ctx.lifecycle.notify_write(1)
                ctx.shard.vector_count += 1
                shard.vector_count = ctx.shard.vector_count
                # maybe split
                self.router.maybe_split(shard, on_split=lambda p,a,b: logger.info(f"Shard split {p.id} -> {a.id},{b.id}"))
                upserted.append(str(rec.id))
                INGEST_COUNTER.labels(namespace=namespace, status="success").inc()
            except Exception as e:
                logger.warning(f"upsert failed {e}")
                errors.append({"record": raw.get("id"), "error": str(e)})
                INGEST_COUNTER.labels(namespace=namespace, status="error").inc()

        # Batch WAL fsync: one flush per touched shard for the whole request (group commit)
        for ctx in touched.values():
            ctx.wal.flush()

        # Background tasks (simulate)
        for ctx in touched.values():
            if ctx.segments.memtable.should_flush():
                ctx.segments.flush()
                ctx.lifecycle.maybe_compact()

        result = {"upserted": len(upserted), "ids": upserted, "errors": errors, "took_ms": int((time.time()-t0)*1000)}
        if idempotency_key:
            self._idempotency[idempotency_key] = result
        return result

    async def delete(self, namespace: str, ids: list[str] | None = None, filter: dict | None = None) -> dict:
        ns = self.metadata.get(namespace)
        if not ns:
            raise ValueError("namespace not found")
        deleted = 0
        shards = self.router.route_for_query(namespace)
        touched: dict[str, ShardContext] = {}
        for shard in shards:
            ctx = self._get_or_create_ctx(namespace, shard)
            if ids:
                for _id in ids:
                    # remove from segments (scan) and index
                    rec = ctx.segments.get(_id)
                    if rec:
                        ctx.segments.memtable.delete(_id)
                    ctx.index.delete([_id])
                    ctx.lifecycle.notify_delete(1)
                    ctx.tombstones.add(_id)  # hide any stale SSTable copy during this process lifetime
                    # durable tombstone so WAL replay does not resurrect the record
                    ctx.wal.append(WALEntry(payload=json.dumps({"__op": "delete", "id": _id}).encode()))
                    touched[f"{namespace}:{shard.id}"] = ctx
                    deleted += 1
            elif filter:
                # naive filter scan
                for rec in ctx.segments.scan_all():
                    if self._matches_filter(rec.metadata, filter):
                        ctx.index.delete([str(rec.id)])
                        ctx.segments.memtable.delete(str(rec.id))
                        ctx.tombstones.add(str(rec.id))
                        ctx.wal.append(WALEntry(payload=json.dumps({"__op": "delete", "id": str(rec.id)}).encode()))
                        touched[f"{namespace}:{shard.id}"] = ctx
                        deleted += 1
        # Batch WAL fsync for tombstones
        for ctx in touched.values():
            ctx.wal.flush()
        return {"deleted": deleted}

    def _matches_filter(self, md: dict, f: dict) -> bool:
        for k, cond in f.items():
            if isinstance(cond, dict):
                if "$in" in cond:
                    if md.get(k) not in cond["$in"] and k not in str(md.get("tags", [])):
                        # also check tags
                        if k == "tags":
                            if not any(t in cond["$in"] for t in md.get("tags", [])):
                                return False
                        else:
                            return False
                elif "$gte" in cond:
                    if md.get(k, "") < cond["$gte"]:
                        return False
                elif "$eq" in cond:
                    if md.get(k) != cond["$eq"]:
                        return False
            else:
                if md.get(k) != cond:
                    return False
        return True

    async def fetch(self, namespace: str, ids: list[str]) -> list[dict]:
        out=[]
        for _id in ids:
            # route to shard
            shard = self.router.route(namespace, _id)
            ctx = self._get_or_create_ctx(namespace, shard)
            rec = ctx.segments.get(_id)
            if rec and _id not in ctx.tombstones:
                out.append(rec.to_payload(include_vector=True))
        return out

    async def get_one(self, namespace: str, id: str) -> dict | None:
        r = await self.fetch(namespace, [id])
        return r[0] if r else None

    def stats(self) -> dict:
        out = {}
        for k, v in self._shards.items():
            out[k] = {
                "vectors": v.shard.vector_count,
                "state": v.lifecycle.state.value,
                "backend_name": getattr(v.index, "backend_name", "unknown"),
                "is_native_backend": getattr(v.index, "is_native_backend", False),
                "degraded": getattr(v.index, "degraded", False),
                "recovered_from_wal": v.recovery.recovered_from_wal,
                "wal_entries_read": v.recovery.wal_entries_read,
            }
        return out
