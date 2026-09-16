# Vsector — Trillion-Scale Distributed Vector Database

> Production-grade, distributed vector DB capable of storing & querying **trillions of high-dimensional vectors** with **low-latency ANN** at massive scale. Built for AI coding tools, RAG, and semantic search.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

---

## Objective

- Storing & querying trillions of high-dimensional vector records
- Serving low-latency **ANN (HNSW / IVF-PQ / ScaNN)** searches
- Communicating with external systems via **REST, gRPC, GraphQL, WebSocket, Kafka**
- Supporting **multi-tenancy, RBAC, observability, disaster recovery**
- SOLID, 12-factor, horizontally scalable, cloud-native

## Architecture (Layered Microservices)

```
Client / External Systems  →  REST | gRPC | GraphQL | WebSocket | Kafka Consumer
         ↓
API Gateway Layer          →  Rate Limiting | Auth (JWT/OAuth2) | LB | Routing | Circuit Breaker | Versioning
         ↓
Core Service Layer         →  Ingest Service | Query Service | Index Manager | Namespace Manager | Replication Coordinator | Compaction Service
         ↓
Storage Engine Layer       →  HNSW/IVF-PQ/ScaNN Shards (10-50B vectors, 3x repl) | WAL | Segment Store (Tiered) | Metadata Store (PostgreSQL → JSON)
         ↓
Infrastructure & Platform  →  Kubernetes | S3 | CDN/DNS | Prometheus/Grafana/Jaeger/ELK
```

See `deploy/k8s/deployment.yaml` for K8s + HPA.

## Quickstart

```bash
pip install -e .          # or pip install -e .[dev,ann]
vsector serve --reload     # REST http://localhost:8080  gRPC :50051
# another shell
python examples/quickstart.py
```

### REST example

```bash
curl -X POST http://localhost:8080/v1/namespaces -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" -d '{"name":"products","dimension":1536,"index_type":"HNSW"}'

curl -X POST http://localhost:8080/v1/vectors/upsert -H "X-API-Key: test-key" \
  -d '{"namespace":"products","vectors":[{"vector":[0.1,0.2,...],"metadata":{"category":"books"}}]}'

curl -X POST http://localhost:8080/v1/vectors/query -H "X-API-Key: test-key" \
  -d '{"namespace":"products","vector":[0.1,0.2,...],"top_k":10,"filters":{"category":{"$eq":"books"}}}'
```

### Python SDK

```python
from vsector.api.sdk import VsectorClient
client = VsectorClient(api_key="test-key")
client.create_namespace("demo", dimension=1536)
client.upsert("demo", [{"vector": [...], "metadata": {"k": "v"}}])
res = client.query("demo", vector=[...], top_k=10)
```

## Module-by-Module Spec

### Module 1: Data Model & Schema — `vsector/models/`
```python
class VectorRecord:
    id: UUID; namespace: str; vector: List[float]; dimension: int
    metadata: Dict[str, Any]; created_at, updated_at, version:int
    ttl: Optional[datetime]; tags: List[str]; source_system:str; checksum:str

class Namespace:
    name, dimension, distance_metric{COSINE,EUCLIDEAN,DOT_PRODUCT,MANHATTAN}
    index_type{HNSW,IVF_PQ,FLAT,SCANN}, replication_factor(default 3), shard_count, compression{NONE,PQ,SQ8,BF16}
```

### Module 2: Distributed Sharding — `vsector/sharding/`
- **Rendezvous Hashing (HRW)** for `shard_key = hash(namespace+record_id)`, 256 vnodes/node
- Hot-shard detection, auto-split at 50B vectors
- Shard metadata in etcd (`EtcdStore` abstraction), routing table cached 5s + gossip invalidation + fallback
- Zero-downtime split: lock WAL → copy halves → atomic routing update in etcd → drain WAL → retire parent

### Module 3: Vector Index Engine — `vsector/index/`
- **HNSW**: M 16-64, efConstruction 200-500, efSearch 50-200, maxLevel auto `log(n)/log(M)`, flat int32 adjacency, mmap files, level-0 NVMe, incremental build
- **IVF-PQ**: nlist 4096-65536, nprobe 64-256, subspaces `dim/8`, 256 codes (8-bit), train on 1M samples, shared-mem codebook, 24h retrain. Uses FAISS if available else flat simulation.
- **Lifecycle**: `BUILDING → READY → DEGRADED → COMPACTING → READY` with triggers delete_ratio>10%, fragmentation>0.3, every 6h

### Module 4: Write Path — `vsector/ingest/`
`Client → Gateway → Ingest (schema+auth) → WAL (GROUP_COMMIT 1000/5ms, fsync) → Shard Router (HRW) → Primary Shard → MemTable → async replicate 2 followers (quorum ack) → ACK → Background MemTable→SSTable flush → SSTable→Index merge`. WAL binary `[len:4B][checksum:4B][ts:8B][payload:NB]`, segmented 256MB, 7-day retention. Batch 10k, async job_id+webhook, idempotency_key.

**Replication health & readiness** — writes are applied to independently represented in-process replica state (WAL + SegmentStore + index); ack reflects the actual result, never a simulation. A shard is **ready** when the primary is available AND `1 + healthy_replicas >= required_acks` can currently be satisfied; **degraded** means at least one configured replica is unhealthy/unavailable (independent of ready — `required_acks = 1` with a down replica is degraded-but-ready). A replica is **healthy** when it has no currently known replication failure; health only clears after a real subsequent success (`heal` alone is not a fake recovery). Health is exposed additively in write responses (`replication.health`), per shard in `GET /metrics` (gauges `vsector_replication_healthy_replicas`, `vsector_replication_ready`) and via `replication_health()` on the transport; inspection never mutates WAL/replica state and never performs writes.

### Module 5: Read Path — `vsector/query/`
`Query Service → validate dim → pre-filter (Bloom+inverted) → fan-out parallel async to shards → per-shard ANN ef_search → post-filter → top-K local → merge global top-K → re-rank (cross-encoder/MMR stub) → return`. Schema supports `filters, ef_search, include_metadata/vector, consistency EVENTUAL|STRONG, timeout_ms 200`. Target P99 <50ms @1M, <200ms @1T.

### Module 6: External System Layer — `vsector/api/` + `vsector/gateway/`
- **6A REST** `Base /v1` OpenAPI 3.1: `POST/DELETE/GET /namespaces*`, `POST /vectors/upsert|query|fetch|delete`, `PATCH /vectors/{id}/metadata`, `GET /vectors/{id}`, `GET /health|/ready|/metrics` — Auth API-Key/JWT/mTLS, rate limit 10k rps, headers `X-RateLimit-*`
- **6B gRPC** `proto/vsector.proto` → `VectorDBService` with Upsert (client stream), Query, BatchQuery (bidi), Delete, Fetch, WatchNamespace (server stream), gRPC-Web, reflection, deadline propagation
- **6C Kafka** `vsector.api.events.EventBus` with `AIOKafkaProducer/Consumer`, topics `vsector.<ns>.<event>`, group consumer → IngestService

## Project Layout

```
vsector/
  models/       # VectorRecord, Namespace
  sharding/     # HRW, ConsistentHashRing, Shard, Router, Coordinator
  index/        # BaseIndex, Flat, HNSW, IVF-PQ, lifecycle, factory
  storage/      # WAL, SegmentStore (MemTable/SSTable tiered), MetadataStore
  ingest/       # IngestService (write path)
  query/        # QueryEngine (read path)
  api/          # rest.py (FastAPI), grpc_server.py, events.py (Kafka), sdk.py, websocket.py, schemas.py
  gateway/      # auth.py (JWT/API-Key/rate limit/circuit breaker)
  infra/        # config.py (12-factor), logging.py, metrics.py
deploy/k8s/     # deployment.yaml (HPA 3-20), configmap.yaml
examples/       # quickstart.py, client_example.py
tests/          # test_*.py
proto/vsector.proto
```

## Configuration (12-factor via env `VSECTOR_*`)

See `.env.example` and `vsector/infra/config.py`. Key vars: `VSECTOR_DATA_DIR`, `VSECTOR_KAFKA_BOOTSTRAP_SERVERS`, `VSECTOR_ETCD_ENDPOINTS`, `VSECTOR_RATE_LIMIT_RPS`, `VSECTOR_SHARD_MAX_VECTORS`, `VSECTOR_WAL_SEGMENT_BYTES`, etc.

## Observability

- Prometheus `/metrics` (ingest_total, query_total/latency, index_vectors, wal_bytes)
- Grafana dashboards (`deploy/grafana/`)
- Jaeger tracing stub (`OTEL_ENDPOINT`), ELK logging

## Deploy

```bash
docker build -t vsector:0.1.0 .
docker-compose up --build   # vsector + kafka + etcd + prometheus + grafana
kubectl apply -f deploy/k8s/
```

## Development

```bash
make install   # pip install -e .[dev]
make proto     # regenerate gRPC stubs
make test      # pytest
make lint      # ruff check
make run       # vsector serve --reload
```

Tests: `tests/test_models.py` `test_sharding.py` `test_index.py` `test_ingest_query.py` `test_api.py`

## Roadmap / Production Hardening

- Replace `MetadataStore` JSON with PostgreSQL + migrations
- Replace `EtcdStore` in-memory with real etcd + gossip
- S3 tier for cold SSTables, CDC + disaster recovery
- RBAC per-namespace, multi-tenancy isolation, mTLS for inter-service
- ScaNN index, BF16/SQ8 quantization
- Full re-ranking with cross-encoder

## License

MIT — see [LICENSE](LICENSE)
