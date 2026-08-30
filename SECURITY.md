# Security & Disaster Recovery — SOC2 + GDPR

## Authentication (SOC2 CC6.1)
- API-Key `X-API-Key` (service), JWT Bearer (user, `vsector/gateway/auth.py:41`), mTLS inter-service (`gateway/mtls.py:1`, `VSECTOR_MTLS_CA_PATH`). Enforce `VSECTOR_ENV=production` + `VSECTOR_SECRET_KEY`.

## RBAC & Multi-tenancy (SOC2 CC6.2)
- `Namespace.tenant_id` isolates tenants (`models/namespace.py:1`), OPA `policy/rbac.rego:1` (admin/writer/reader), per-namespace quota `X-RateLimit-Remaining/Reset` (`gateway/auth.py:26`), `gateway/rbac.py:1` `check_permission` + `VSECTOR_OPA_URL`.

## Data Integrity (SOC2 CC6.7)
- `VectorRecord.checksum` SHA-256 (`models/vector_record.py:1`), WAL `zlib.crc32` + `O_DIRECT` + `fsync` per `GROUP_COMMIT` (`storage/wal.py:1`).

## Disaster Recovery (SOC2 A1.2)
- Repl 3 quorum (`sharding/coordinator.py:1`), WAL 7d, S3 cold (`storage/s3.py:1`, `VSECTOR_S3_BUCKET`), `deploy/k8s` PVC+Etcd `etcdctl snapshot save`, multi-region `sharding/multiregion.py:1` (`VSECTOR_REGIONS`).

## GDPR Hard-Delete Audit
- `POST /v1/vectors/delete` (`api/rest.py:1`) hard-deletes from MemTable+SSTable+Index+WAL, then `gateway/rbac.py:1` audit log. Purge propagation: `infra/cache.py:1` + `infra/cdn.py:1` + S3. Verified via `tests/test_rbac.py:1` + `benchmarks/jepsen.py:1`.

## Observability (SOC2 CC7.2)
- Prometheus `/metrics` + `prometheus-alerts.yml:1` (P99, error budget), Grafana `deploy/grafana/dashboard.json:1`, ELK JSON `infra/logging.py:1` (`VSECTOR_LOG_JSON=1`), Jaeger `OTEL_ENDPOINT`.

## mTLS CA Rotation
- `gateway/mtls.py:1` `MTLSManager.should_rotate` 1h, `certs/README.md:1` openssl CA, rotation via `SIGHUP` reload, `X-Client-Cert` fingerprint → tenant.

## Reporting
- `security@vsector.io` (stub), OPA bundle `policy/` versioned.
