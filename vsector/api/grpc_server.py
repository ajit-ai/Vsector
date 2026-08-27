"""gRPC Service - Protocol Buffers, client streaming, server streaming, deadline propagation."""
from __future__ import annotations

import asyncio
import logging
import time
import grpc  # type: ignore

logger = logging.getLogger(__name__)

# Note: Generates stubs via: python -m grpc_tools.protoc -I proto --python_out=vsector/api --grpc_python_out=vsector/api proto/vsector.proto
# Fallback wrapper that delegates to REST Ingest/Query services without requiring generated stubs at runtime.

class VectorDBGrpcService:
    """Implements VectorDBService rpcs using IngestService/QueryEngine."""

    def __init__(self, ingest, query_engine):
        self.ingest = ingest
        self.query_engine = query_engine

    async def Upsert(self, request_iterator, context):
        vectors = []
        namespace = None
        idempotency_key = None
        async for req in request_iterator:
            namespace = req.namespace or namespace
            idempotency_key = getattr(req, "idempotency_key", None) or idempotency_key
            for v in req.vectors:
                vectors.append({"id": v.id, "vector": list(v.values), "metadata": dict(v.metadata)})
        if not namespace:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("namespace required")
            return None
        res = await self.ingest.upsert(namespace, vectors, idempotency_key=idempotency_key)
        # Build response type dynamically if stubs available
        try:
            from . import vsector_pb2  # type: ignore
            return vsector_pb2.UpsertResponse(upserted=res["upserted"], ids=res["ids"])
        except Exception:
            return res

    async def Query(self, request, context):
        # deadline propagation
        deadline = context.time_remaining()
        timeout_ms = int(deadline * 1000) if deadline else 200
        res = await self.query_engine.query(request.namespace, list(request.vector), top_k=request.top_k or 10, timeout_ms=timeout_ms)
        try:
            from . import vsector_pb2  # type: ignore
            results = [vsector_pb2.QueryResult(id=r["id"], score=r["score"], metadata={k: str(v) for k,v in r["metadata"].items()}) for r in res["results"]]
            return vsector_pb2.QueryResponse(results=results, took_ms=res["took_ms"], shard_count=res["shard_count"])
        except Exception:
            return res

    async def BatchQuery(self, request_iterator, context):
        async for req in request_iterator:
            resp = await self.Query(req, context)
            yield resp

    async def Delete(self, request, context):
        res = await self.ingest.delete(request.namespace, ids=list(request.ids))
        try:
            from . import vsector_pb2  # type: ignore
            return vsector_pb2.DeleteResponse(deleted=res["deleted"])
        except Exception:
            return res

    async def Fetch(self, request, context):
        res = await self.ingest.fetch(request.namespace, list(request.ids))
        try:
            from . import vsector_pb2  # type: ignore
            vecs = [vsector_pb2.Vector(id=r["id"], namespace=r["namespace"], values=r.get("vector") or [], metadata={k: str(v) for k,v in r.get("metadata", {}).items()}) for r in res]
            return vsector_pb2.FetchResponse(vectors=vecs)
        except Exception:
            return {"vectors": res}

    async def WatchNamespace(self, request, context):
        """Server streaming - watch for namespace events (ingest/delete)."""
        # Simple polling loop; in prod use etcd watch + Kafka
        last_count = 0
        try:
            from . import vsector_pb2  # type: ignore
            has_pb2 = True
        except Exception:
            has_pb2 = False
        while not context.done():
            shards = self.ingest.router.etcd.list_by_namespace(request.namespace)
            total = sum(s.vector_count for s in shards)
            if total != last_count:
                last_count = total
                payload = f'{{"total_vectors": {total}, "shard_count": {len(shards)}}}'
                if has_pb2:
                    yield vsector_pb2.NamespaceEvent(type="UPSERT", namespace=request.namespace, payload=payload)
                else:
                    yield {"type": "UPSERT", "namespace": request.namespace, "payload": payload}
            await asyncio.sleep(1)


def create_grpc_server(ingest, query_engine, port: int = 50051):
    """Factory for grpc.aio.Server with reflection."""
    server = grpc.aio.server()
    service = VectorDBGrpcService(ingest, query_engine)
    # If generated stubs exist, register properly:
    try:
        from . import vsector_pb2_grpc  # type: ignore
        vsector_pb2_grpc.add_VectorDBServiceServicer_to_server(service, server)  # type: ignore
        # reflection
        try:
            from grpc_reflection.v1alpha import reflection  # type: ignore
            SERVICE_NAMES = ("vsector.VectorDBService", reflection.SERVICE_NAME)
            reflection.enable_server_reflection(SERVICE_NAMES, server)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"gRPC stubs not generated, using fallback: {e}")
    server.add_insecure_port(f"[::]:{port}")
    return server
