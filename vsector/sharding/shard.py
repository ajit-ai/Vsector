"""Shard metadata & split algorithm."""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field


class ShardState(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SPLITTING = "SPLITTING"
    RETIRED = "RETIRED"


@dataclass
class Shard:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = ""
    node_id: str = ""
    state: ShardState = ShardState.ACTIVE
    vector_count: int = 0
    max_vectors: int = 50_000_000_000
    replicas: list[str] = field(default_factory=list)  # follower nodes
    created_at: float = field(default_factory=time.time)
    version: int = 1

    def should_split(self) -> bool:
        return self.vector_count > self.max_vectors

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "node_id": self.node_id,
            "state": self.state.value,
            "vector_count": self.vector_count,
            "max_vectors": self.max_vectors,
            "replicas": self.replicas,
            "version": self.version,
        }
