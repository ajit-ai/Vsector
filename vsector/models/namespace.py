"""Namespace config - MODULE 1."""
from __future__ import annotations

import uuid
from enum import Enum
from pydantic import BaseModel, Field


class DistanceMetric(str, Enum):
    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN"
    DOT_PRODUCT = "DOT_PRODUCT"
    MANHATTAN = "MANHATTAN"


class IndexType(str, Enum):
    HNSW = "HNSW"
    IVF_PQ = "IVF_PQ"
    FLAT = "FLAT"
    SCANN = "SCANN"


class CompressionType(str, Enum):
    NONE = "NONE"
    PQ = "PQ"
    SQ8 = "SQ8"
    BF16 = "BF16"


class Namespace(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_\-\.]{1,64}$", description="Logical partition (tenant/collection)")
    dimension: int = Field(..., ge=1, le=65536, description="Fixed per namespace e.g. 768, 1536, 3072")
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    index_type: IndexType = IndexType.HNSW
    replication_factor: int = Field(default=3, ge=1, le=7)
    shard_count: int = Field(default=8, ge=1, le=1024, description="Auto-calculated based on expected record count")
    compression: CompressionType = CompressionType.NONE
    tenant_id: uuid.UUID = Field(default_factory=uuid.uuid4)

    def validate_dimension(self, dim: int) -> None:
        if dim != self.dimension:
            raise ValueError(f"dimension mismatch: expected {self.dimension}, got {dim}")

    model_config = {"frozen": False}
