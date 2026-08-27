"""Bulk loader — 10M vec/s sustained target (parallel, batched, backpressure).

Uses asyncio + httpx or direct IngestService for max throughput.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


async def bulk_upsert_direct(ingest_service, namespace: str, vectors: list[list[float]], batch_size: int = 10000, concurrency: int = 8, metadata_fn=None) -> dict:
    """Direct in-process bulk load (bypasses HTTP) — for 10M vec/s target on NVMe.

    Splits vectors into batch_size chunks, fans out via asyncio.Semaphore(concurrency).
    """
    total = len(vectors)
    t0 = time.time()
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def _batch(start: int):
        async with sem:
            chunk = vectors[start : start + batch_size]
            payload = []
            for i, v in enumerate(chunk):
                md = metadata_fn(start + i) if metadata_fn else {}
                payload.append({"vector": v, "metadata": md})
            r = await ingest_service.upsert(namespace, payload)
            return r

    tasks = [asyncio.create_task(_batch(i)) for i in range(0, total, batch_size)]
    for t in asyncio.as_completed(tasks):
        results.append(await t)

    elapsed = time.time() - t0
    qps = total / elapsed if elapsed else 0
    logger.info(f"bulk_upsert {total} vectors in {elapsed:.2f}s qps={qps:.0f} batches={len(results)}")
    return {"total": total, "elapsed_s": elapsed, "qps": qps, "batches": len(results)}


# HTTP bulk loader for remote clusters (LangChain-style)
async def bulk_upsert_http(base_url: str, api_key: str, namespace: str, vectors: list[list[float]], batch_size: int = 1000) -> dict:
    import httpx

    total = len(vectors)
    t0 = time.time()
    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(0, total, batch_size):
            chunk = [{"vector": v} for v in vectors[i : i + batch_size]]
            r = await client.post(
                f"{base_url}/v1/vectors/upsert",
                headers={"X-API-Key": api_key},
                json={"namespace": namespace, "vectors": chunk},
            )
            r.raise_for_status()
    elapsed = time.time() - t0
    return {"total": total, "elapsed_s": elapsed, "qps": total / elapsed if elapsed else 0}
