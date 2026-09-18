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
from ..sharding.router import ShardRouter
from ..sharding.shard import Shard
from ..sharding.lifecycle import ShardLifecycleManager
from ..sharding.replication import InProcessReplicaTransport, ReplicaEndpoint, aggregate_replication
from ..index.factory import create_index
from ..index.lifecycle import IndexLifecycleManager
from ..infra.metrics import INGEST_COUNTER, REPLICATION_ATTEMPTS, REPLICATION_ACKS, REPLICATION_FAILURES, REPLICATION_DEGRADED, REPLICATION_HEALTHY_REPLICAS, REPLICATION_READY

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
    def __init__(self, metadata: MetadataStore, router: ShardRouter, base_dir: str = "./data", node_id: str | None = None):
        self.metadata = metadata
        self.router = router
        self.base_dir = base_dir
        # Local node identity for VS-10 ownership-aware routing. ``None`` means the
        # service hosts shards as their primary without asserting a fixed identity
        # (single-process deployment); configure an explicit ``node_id`` so writes are
        # refused on shards owned by a different node (no fake forwarding).
        self.node_id = node_id
        self._lifecycle = ShardLifecycleManager(self.router.etcd)
        # Real replication transport: applies writes to independently represented in-process
        # replica contexts and reports actual success/degraded/failure (VS-09.2).
        self.replicator = InProcessReplicaTransport(self._get_or_create_replica_endpoint)
        self._replica_endpoints: dict[str, ReplicaEndpoint] = {}
        self._replica_base_dir = f"{base_dir}/replicas"
        self._replication_stats: dict[str, dict[str, int]] = {}
        self._shards: dict[str, ShardContext] = {}
        self._idempotency: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def _get_or_create_replica_endpoint(self, namespace: str, shard: Shard, node_id: str) -> ReplicaEndpoint:
        key = f"{namespace}:{shard.id}:{node_id}"
        endpoint = self._replica_endpoints.get(key)
        if endpoint is None:
            ns = self.metadata.get(namespace)
            if not ns:
                raise ValueError(f"namespace {namespace!r} not found")
            # Independent replica state: dedicated shard id + dedicated storage dirs so the
            # replica's WAL / SegmentStore / index / tombstones are never the primary's objects.
            replica_shard = Shard(id=f"{shard.id}#replica-{node_id}", namespace=namespace, node_id=node_id)
            ctx = ShardContext(replica_shard, ns, self._replica_base_dir)
            endpoint = ReplicaEndpoint(node_id, ctx)
            self._replica_endpoints[key] = endpoint
        return endpoint

    def _record_replication_stats(self, namespace: str, shard: Shard, repl) -> None:
        key = f"{namespace}:{shard.id}"
        st = self._replication_stats.setdefault(key, {"attempts": 0, "acks": 0, "failures": 0, "degraded": 0})
        st["attempts"] += len(repl.attempted_replicas)
        st["acks"] += len(repl.acknowledged_replicas)
        st["failures"] += len(repl.failed_replicas)
        if repl.degraded:
            st["degraded"] += 1
        REPLICATION_ATTEMPTS.labels(namespace=namespace).inc(len(repl.attempted_replicas))
        REPLICATION_ACKS.labels(namespace=namespace).inc(len(repl.acknowledged_replicas))
        for node_id in repl.failed_replicas:
            REPLICATION_FAILURES.labels(namespace=namespace, replica_id=node_id).inc()
        if repl.degraded:
            REPLICATION_DEGRADED.labels(namespace=namespace).inc()

    def _replication_health_map(self, namespace: str, ns: Namespace, touched: dict[str, ShardContext]) -> dict:
        """Additive per-shard health snapshot for a write request (single shard -> object)."""
        health_map: dict[str, dict] = {}
        for ctx in touched.values():
            key = f"{ctx.shard.namespace}:{ctx.shard.id}"
            health_map[key] = self.replicator.replication_health(namespace, ctx.shard, ns.required_acks).to_dict()
        if len(health_map) == 1:
            return list(health_map.values())[0]
        return {"shards": health_map}

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
        replication_results = []

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
                # shard routing (HRW): lifecycle- and ownership-aware (VS-10)
                shard = self.router.route_write(namespace, str(rec.id), self.node_id)
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
                # Index merge: insert directly into live index (primary local apply)
                ctx.index.add([str(rec.id)], np.array([rec.vector], dtype=np.float32), [rec.metadata])
                ctx.lifecycle.notify_write(1)
                ctx.shard.vector_count += 1
                shard.vector_count = ctx.shard.vector_count
                # Replicate to replicas through the real application path (primary is durable/local).
                # Outcome is structured and reflects actual replica application results.
                repl = self.replicator.replicate_upsert(namespace, shard, payload, required_acks=ns.required_acks)
                replication_results.append(repl)
                self._record_replication_stats(namespace, shard, repl)
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
        # Replication outcome (additive, backward compatible). If any record failed on the
        # primary, the request cannot claim fully replicated success.
        repl_summary = aggregate_replication(replication_results, ns.required_acks)
        if errors:
            repl_summary["success"] = False
        repl_summary["health"] = self._replication_health_map(namespace, ns, touched)
        result["replication"] = repl_summary
        if idempotency_key:
            self._idempotency[idempotency_key] = result
        return result

    async def delete(self, namespace: str, ids: list[str] | None = None, filter: dict | None = None) -> dict:
        ns = self.metadata.get(namespace)
        if not ns:
            raise ValueError("namespace not found")
        deleted = 0
        shards = self.router.writable(namespace, self.node_id)
        touched: dict[str, ShardContext] = {}
        replication_results = []
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
                    payload = json.dumps({"__op": "delete", "id": _id}).encode()
                    ctx.wal.append(WALEntry(payload=payload))
                    touched[f"{namespace}:{shard.id}"] = ctx
                    # replicate the tombstone to replicas through the real application path
                    repl = self.replicator.replicate_delete(namespace, shard, payload, required_acks=ns.required_acks)
                    replication_results.append(repl)
                    self._record_replication_stats(namespace, shard, repl)
                    deleted += 1
            elif filter:
                # naive filter scan
                for rec in ctx.segments.scan_all():
                    if self._matches_filter(rec.metadata, filter):
                        ctx.index.delete([str(rec.id)])
                        ctx.segments.memtable.delete(str(rec.id))
                        ctx.tombstones.add(str(rec.id))
                        payload = json.dumps({"__op": "delete", "id": str(rec.id)}).encode()
                        ctx.wal.append(WALEntry(payload=payload))
                        touched[f"{namespace}:{shard.id}"] = ctx
                        repl = self.replicator.replicate_delete(namespace, shard, payload, required_acks=ns.required_acks)
                        replication_results.append(repl)
                        self._record_replication_stats(namespace, shard, repl)
                        deleted += 1
        # Batch WAL fsync for tombstones
        for ctx in touched.values():
            ctx.wal.flush()
        repl_summary = aggregate_replication(replication_results, ns.required_acks)
        repl_summary["health"] = self._replication_health_map(namespace, ns, touched)
        return {"deleted": deleted, "replication": repl_summary}

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
            ns = v.shard.namespace
            health = self.replicator.replication_health(ns, v.shard, v.namespace.required_acks)
            REPLICATION_HEALTHY_REPLICAS.labels(namespace=ns, shard_id=v.shard.id).set(health.healthy_replicas)
            REPLICATION_READY.labels(namespace=ns, shard_id=v.shard.id).set(1 if health.ready else 0)
            out[k] = {
                "vectors": v.shard.vector_count,
                "state": v.lifecycle.state.value,
                "shard_state": v.shard.state.value,
                "primary_owner": v.shard.node_id,
                "replicas": list(v.shard.replicas),
                "backend_name": getattr(v.index, "backend_name", "unknown"),
                "is_native_backend": getattr(v.index, "is_native_backend", False),
                "degraded": getattr(v.index, "degraded", False),
                "recovered_from_wal": v.recovery.recovered_from_wal,
                "wal_entries_read": v.recovery.wal_entries_read,
                "replication": self._replication_stats.get(k, {"attempts": 0, "acks": 0, "failures": 0, "degraded": 0}),
                "replication_health": health.to_dict(),
            }
        return out
