"""Central configuration - 12-factor via env vars."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Vsector"
    version: str = "0.1.0"
    env: str = "development"  # development | staging | production

    # API
    host: str = "0.0.0.0"
    port: int = 8080
    api_prefix: str = "/v1"
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Rate limiting
    rate_limit_rps: int = 10000
    rate_limit_burst: int = 20000

    # Sharding
    virtual_nodes: int = 256
    shard_max_vectors: int = 50_000_000_000  # 50B auto-split threshold
    shard_default_count: int = 8
    routing_cache_ttl_s: int = 5

    # Index
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    ivf_nlist: int = 4096
    ivf_nprobe: int = 64
    ivf_pq_subspace: int = 8  # dimension // subspaces

    # Storage
    wal_segment_bytes: int = 256 * 1024 * 1024  # 256MB
    wal_group_commit_batch: int = 1000
    wal_group_commit_ms: int = 5
    wal_retention_days: int = 7
    data_dir: str = "./data"
    s3_bucket: str = ""
    s3_prefix: str = "vsector/"

    # Compaction
    compaction_delete_ratio: float = 0.10
    compaction_fragmentation: float = 0.3
    compaction_interval_hours: int = 6

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "vsector."

    # Etcd (abstracted - can be in-memory for single-node)
    etcd_endpoints: str = "localhost:2379"

    # Postgres
    database_url: str = ""  # VSECTOR_DATABASE_URL

    # S3
    s3_endpoint_url: str = ""  # for MinIO

    # Re-ranking
    rerank: str = ""  # cross_encoder | mmr
    mmr_lambda: float = 0.5

    # Observability
    otel_endpoint: str = ""
    log_level: str = "INFO"
    log_json: bool = False

    model_config = {"env_prefix": "VSECTOR_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
