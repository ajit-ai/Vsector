"""REST API - OpenAPI 3.1.

Endpoints per spec:
  POST /namespaces, DELETE /namespaces/{name}, GET /namespaces/{name}/stats
  POST /vectors/upsert, POST /vectors/query, POST /vectors/fetch, POST /vectors/delete, PATCH /vectors/{id}/metadata, GET /vectors/{id}
  GET /health, GET /ready, GET /metrics
"""
from __future__ import annotations

import os
import uuid
import asyncio
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..infra.config import get_settings
from ..storage.metadata import MetadataStore
from ..sharding.router import ShardRouter, EtcdStore
from ..sharding.shard import Shard
from ..models.namespace import Namespace, DistanceMetric, IndexType, CompressionType
from ..ingest.service import IngestService
from ..query.engine import QueryEngine
from ..gateway.auth import verify_api_key, rate_limiter
from .schemas import NamespaceCreate, UpsertRequest, QueryRequest, FetchRequest, DeleteRequest

settings = get_settings()

# singletons for single-node deployment
metadata_store = MetadataStore(path=f"{settings.data_dir}/metadata.json")
etcd = EtcdStore()
router = ShardRouter(etcd=etcd, cache_ttl_s=settings.routing_cache_ttl_s)
ingest = IngestService(metadata=metadata_store, router=router, base_dir=settings.data_dir)
query_engine = QueryEngine(metadata=metadata_store, router=router, ingest_service=ingest)

app = FastAPI(
    title="Vsector Vector Database",
    version=settings.version,
    description="Trillion-Scale Distributed Vector Database - REST API (OpenAPI 3.1)",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# --- health ---
@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "version": settings.version}

@app.get("/ready", tags=["ops"])
async def ready():
    return {"ready": True, "namespaces": len(metadata_store.list())}

@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- namespaces ---
@app.post(f"{settings.api_prefix}/namespaces", tags=["namespaces"], dependencies=[Depends(rate_limiter)])
async def create_namespace(body: NamespaceCreate, _auth=Depends(verify_api_key)):
    try:
        ns = Namespace(
            name=body.name,
            dimension=body.dimension,
            distance_metric=DistanceMetric(body.distance_metric),
            index_type=IndexType(body.index_type),
            replication_factor=body.replication_factor,
            shard_count=body.shard_count,
            compression=CompressionType(body.compression),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        metadata_store.create(ns)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    # create shards (HRW ring seeds)
    for i in range(ns.shard_count):
        shard = Shard(namespace=ns.name, node_id=f"node-{i % 3}", replicas=[f"node-{(i+1)%3}", f"node-{(i+2)%3}"])
        # vary shard count distribution: use consistent hash
        router.register_shard(shard)
    return ns.model_dump(mode="json")

@app.delete(f"{settings.api_prefix}/namespaces/{{name}}", tags=["namespaces"])
async def delete_namespace(name: str, _auth=Depends(verify_api_key)):
    ns = metadata_store.get(name)
    if not ns:
        raise HTTPException(status_code=404, detail="namespace not found")
    # clean shards
    for s in router.etcd.list_by_namespace(name):
        router.etcd.delete(s.id)
    router.invalidate(name)
    metadata_store.delete(name)
    return {"deleted": name}

@app.get(f"{settings.api_prefix}/namespaces/{{name}}/stats", tags=["namespaces"])
async def namespace_stats(name: str):
    ns = metadata_store.get(name)
    if not ns:
        raise HTTPException(status_code=404, detail="not found")
    shards = router.etcd.list_by_namespace(name)
    total = sum(s.vector_count for s in shards)
    return {"namespace": name, "dimension": ns.dimension, "index_type": ns.index_type.value, "shard_count": len(shards), "total_vectors": total, "replication_factor": ns.replication_factor, "shards": [s.to_dict() for s in shards]}

@app.get(f"{settings.api_prefix}/namespaces", tags=["namespaces"])
async def list_namespaces():
    return [n.model_dump(mode="json") for n in metadata_store.list()]

# --- vectors ---
@app.post(f"{settings.api_prefix}/vectors/upsert", tags=["vectors"], dependencies=[Depends(rate_limiter)])
async def upsert_vectors(body: UpsertRequest, _auth=Depends(verify_api_key)):
    try:
        result = await ingest.upsert(body.namespace, body.vectors, idempotency_key=body.idempotency_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

@app.post(f"{settings.api_prefix}/vectors/query", tags=["vectors"], dependencies=[Depends(rate_limiter)])
async def query_vectors(body: QueryRequest, _auth=Depends(verify_api_key)):
    try:
        res = await query_engine.query(
            namespace=body.namespace,
            vector=body.vector,
            top_k=body.top_k,
            ef_search=body.ef_search,
            filters=body.filters,
            include_metadata=body.include_metadata,
            include_vector=body.include_vector,
            consistency=body.consistency,
            timeout_ms=body.timeout_ms,
            nprobe=body.nprobe,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res

@app.post(f"{settings.api_prefix}/vectors/fetch", tags=["vectors"])
async def fetch_vectors(body: FetchRequest, _auth=Depends(verify_api_key)):
    try:
        res = await ingest.fetch(body.namespace, body.ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"vectors": res}

@app.post(f"{settings.api_prefix}/vectors/delete", tags=["vectors"])
async def delete_vectors(body: DeleteRequest, _auth=Depends(verify_api_key)):
    try:
        res = await ingest.delete(body.namespace, ids=body.ids, filter=body.filter)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res

@app.patch(f"{settings.api_prefix}/vectors/{{vid}}/metadata", tags=["vectors"])
async def patch_metadata(vid: str, body: dict[str, Any], namespace: str = "default", _auth=Depends(verify_api_key)):
    # fetch, merge, re-upsert
    rec = await ingest.get_one(namespace, vid)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    # naive: update metadata via re-upsert
    new_rec = {"id": vid, "vector": rec.get("vector") or [0]*metadata_store.get(namespace).dimension, "metadata": {**rec.get("metadata", {}), **body}}
    # if vector missing (not stored), fetch from segments directly
    if not new_rec["vector"] or all(v==0 for v in new_rec["vector"]):
        # try segment
        from uuid import UUID
        shard = router.route(namespace, vid)
        ctx = ingest._shards.get(f"{namespace}:{shard.id}")
        if ctx:
            r = ctx.segments.get(vid)
            if r:
                new_rec["vector"] = r.vector
    await ingest.upsert(namespace, [new_rec])
    return {"updated": vid}

@app.get(f"{settings.api_prefix}/vectors/{{vid}}", tags=["vectors"])
async def get_vector(vid: str, namespace: str = "default", _auth=Depends(verify_api_key)):
    rec = await ingest.get_one(namespace, vid)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    return rec
