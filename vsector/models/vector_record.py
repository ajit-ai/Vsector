"""VectorRecord entity - MODULE 1."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, computed_field


def _sha256_vector(vector: list[float]) -> str:
    h = hashlib.sha256()
    for v in vector:
        h.update(str(v).encode())
    return h.hexdigest()


class VectorRecord(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Globally unique record identifier")
    namespace: str = Field(..., description="Logical partition (tenant/collection)")
    vector: list[float] = Field(..., description="High-dimensional embedding (e.g. 1536-dim)")
    dimension: int = Field(..., ge=1, description="Validated against namespace config")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary filterable key-value payload")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = Field(default=1, ge=1, description="Optimistic concurrency control")
    ttl: Optional[datetime] = Field(default=None, description="Optional expiry for auto-deletion")
    tags: list[str] = Field(default_factory=list, description="For label-based filtering")
    source_system: str = Field(default="unknown", description="Origin system identifier")
    checksum: str = Field(default="", description="SHA-256 of vector bytes for integrity")

    @field_validator("vector")
    @classmethod
    def _validate_vector(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("vector must not be empty")
        for x in v:
            if not isinstance(x, (int, float)):
                raise ValueError("vector elements must be numeric")
        return [float(x) for x in v]

    @field_validator("dimension", mode="before")
    @classmethod
    def _coerce_dimension(cls, v: Any, info) -> int:  # type: ignore
        # if dimension not provided, infer from vector
        if v is None and "vector" in info.data:
            return len(info.data["vector"])
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.dimension != len(self.vector):
            raise ValueError(f"dimension {self.dimension} != len(vector) {len(self.vector)}")
        if not self.checksum:
            object.__setattr__(self, "checksum", _sha256_vector(self.vector))

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return datetime.now(timezone.utc) >= self.ttl

    def bump_version(self) -> None:
        object.__setattr__(self, "version", self.version + 1)
        object.__setattr__(self, "updated_at", datetime.now(timezone.utc))
        object.__setattr__(self, "checksum", _sha256_vector(self.vector))

    def to_payload(self, include_vector: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": str(self.id),
            "namespace": self.namespace,
            "metadata": self.metadata,
            "tags": self.tags,
            "source_system": self.source_system,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "checksum": self.checksum,
        }
        if include_vector:
            d["vector"] = self.vector
            d["dimension"] = self.dimension
        return d
