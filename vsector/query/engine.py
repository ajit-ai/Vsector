"""MODULE 5: Read Path - Query Engine

Flow: Parse & validate query vector -> Apply pre-filter (Bloom + inverted index) -> Fan-out to shards (parallel async)
      -> Each shard ANN search ef_search -> post-filter re-check -> top-K local -> merge global top-K -> re-rank -> return
"""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Any

import numpy as np

from ..storage.metadata import MetadataStore
from ..sharding.router import ShardRouter
from ..infra.metrics import QUERY_COUNTER, QUERY_LATENCY

logger = logging.getLogger(__name__)


def _parse_filter(metadata: dict, filters: dict) -> bool:
    """Simple filter eval supporting $in, $gte, $eq for metadata and tags."""
    if not filters:
        return True
    for key, cond in filters.items():
        val = metadata.get(key)
        # tags special: metadata tags stored inside metadata dict?
        # In index, metadatas contain raw metadata plus tags if present
        if isinstance(cond, dict):
            if "$in" in cond:
                needle = cond["$in"]
                if key == "tags":
                    tags = metadata.get("tags") or metadata.get("_tags") or []
                    if isinstance(val, list):
                        tags = val
                    if not any(t in needle for t in (tags if isinstance(tags, list) else [tags])):
                        return False
                else:
                    if val not in needle:
                        return False
            elif "$gte" in cond:
                if val is None or str(val) < str(cond["$gte"]):
                    return False
            elif "$eq" in cond:
                if val != cond["$eq"]:
                    return False
            elif "$in_tags" in cond:
                tags = metadata.get("tags", [])
                if not any(t in cond["$in_tags"] for t in tags):
                    return False
            else:
                # unknown op
                if val != cond:
                    return False
        else:
            if val != cond:
                return False
    return True


class QueryEngine:
    def __init__(self, metadata: MetadataStore, router: ShardRouter, ingest_service):
        self.metadata = metadata
        self.router = router
        self.ingest = ingest_service  # to access shard contexts / indexes

    async def query(self, namespace: str, vector: list[float], top_k: int = 10, ef_search: int | None = None, filters: dict | None = None, include_metadata: bool = True, include_vector: bool = False, consistency: str = "EVENTUAL", timeout_ms: int = 200, nprobe: int | None = None) -> dict[str, Any]:
        t0 = time.time()
        ns = self.metadata.get(namespace)
        if not ns:
            raise ValueError(f"namespace {namespace!r} not found")
        if len(vector) != ns.dimension:
            raise ValueError(f"dimension mismatch expected {ns.dimension} got {len(vector)}")
        filters = filters or {}

        # Bloom filter + inverted index pre-filter simulation: we just build filter_fn
        def filter_fn(md: dict) -> bool:
            return _parse_filter(md, filters)

        shards = self.router.route_for_query(namespace)
        if not shards:
            return {"results": [], "took_ms": 0, "shard_count": 0, "total_candidates_scanned": 0}

        # Fan-out parallel async
        query_vec = np.array(vector, dtype=np.float32)

        async def search_shard(shard):
            key = f"{namespace}:{shard.id}"
            ctx = self.ingest._shards.get(key)
            if not ctx:
                return [], 0
            # ANN search
            results = ctx.index.search(query_vec, top_k=top_k, ef_search=ef_search or ns.index_type, nprobe=nprobe, filter_fn=filter_fn if filters else None)
            # post-filter re-check (already)
            # return local top-K
            candidates = len(results)
            # attach record metadata fetch for include flags
            enriched=[]
            for r in results:
                md = r.metadata or {}
                # fetch full record if needed for vector
                vec_payload = None
                if include_vector:
                    rec = ctx.segments.get(r.id)
                    if rec:
                        vec_payload = rec.vector
                        md = rec.metadata
                enriched.append({
                    "id": r.id,
                    "score": r.score,
                    "metadata": md if include_metadata else {},
                    "vector": vec_payload if include_vector else None,
                })
            return enriched, candidates

        # run with timeout
        try:
            shard_results = await asyncio.wait_for(asyncio.gather(*[search_shard(s) for s in shards]), timeout=timeout_ms/1000.0)
        except asyncio.TimeoutError:
            # partial results
            logger.warning("query timeout")
            shard_results = []

        all_cands = []
        total_scanned = 0
        for local, cnt in shard_results:
            all_cands.extend(local)
            total_scanned += cnt

        # Merge global top-K
        all_cands.sort(key=lambda x: x["score"], reverse=True)
        top = all_cands[:top_k]

        # optional re-ranking: cross-encoder/MMR placeholder (simple score boost for recency)
        # here we keep sorted order

        took_ms = int((time.time() - t0) * 1000)
        QUERY_COUNTER.labels(namespace=namespace).inc()
        QUERY_LATENCY.labels(namespace=namespace).observe(took_ms)

        # consistency: STRONG would read from WAL / primary only - simulated as same

        return {
            "results": top,
            "took_ms": took_ms,
            "shard_count": len(shards),
            "total_candidates_scanned": total_scanned,
        }
