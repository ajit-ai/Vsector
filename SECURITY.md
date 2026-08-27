# Security & Disaster Recovery

## Authentication
- API-Key via `X-API-Key` (service-to-service), JWT Bearer for user-facing, mTLS for inter-service (see `vsector/gateway/auth.py`).
- In production set `VSECTOR_SECRET_KEY` and `VSECTOR_ENV=production` to enforce auth.

## RBAC & Multi-tenancy
- `Namespace.tenant_id` isolates tenants. Enforce per-namespace quota via `rate_limiter` (`X-RateLimit-Remaining/Reset`).
- Future: integrate OPA/Keto for fine-grained RBAC.

## Data Integrity
- `VectorRecord.checksum` is SHA-256 of vector bytes, validated on ingest.
- WAL checksum per entry (`zlib.crc32`) and `fsync` per `GROUP_COMMIT`.

## Disaster Recovery
- Replication factor 3 (primary + 2 followers, quorum ack).
- WAL retention 7 days for catch-up.
- S3 tier for cold SSTables (`VSECTOR_S3_BUCKET`). Use `deploy/k8s` PVC for HA.
- Etcd backup: snapshot `etcdctl snapshot save`.

## Observability
- Prometheus `/metrics`, Grafana `deploy/grafana/dashboard.json`, Jaeger `OTEL_ENDPOINT`, ELK via stdout JSON.

## Reporting
- Report vulnerabilities to security@vsector.io (stub).
