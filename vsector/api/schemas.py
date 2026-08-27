"""OpenAPI 3.1 schemas for REST."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class NamespaceCreate(BaseModel):
    name: str
    dimension: int
    distance_metric: str = "COSINE"
    index_type: str = "HNSW"
    replication_factor: int = 3
    shard_count: int = 8
    compression: str = "NONE"


class UpsertRequest(BaseModel):
    namespace: str
    vectors: list[dict[str, Any]] = Field(..., description="Each: {id?, vector: float[], metadata?, tags?, source_system?}")
    idempotency_key: str | None = None


class QueryRequest(BaseModel):
    namespace: str
    vector: list[float]
    top_k: int = 10
    ef_search: int | None = None
    nprobe: int | None = None
    filters: dict[str, Any] | None = None
    include_metadata: bool = True
    include_vector: bool = False
    consistency: str = "EVENTUAL"
    timeout_ms: int = 200


class FetchRequest(BaseModel):
    namespace: str
    ids: list[str]


class DeleteRequest(BaseModel):
    namespace: str
    ids: list[str] | None = None
    filter: dict[str, Any] | None = None
