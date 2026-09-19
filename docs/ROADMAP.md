# Vsector Roadmap — VS-15 → GA 1.0

> Status anchor: **VS-14 complete** (Sep 2026). Last delivered: durable shard-migration
> CRUD on the shared etcd store (merge `52031f7`). This roadmap covers everything up to
> and including **VSector GA 1.0**, then lists a brief post-GA vision.

## Where we are

Implemented through **VS-14**: explicit shard lifecycle + ownership (VS-10), cluster
membership + node identity (VS-11), placement / routing decisions (VS-12), digest-verified
shard migration over real durable per-node storage roots (VS-13), durable migration store
on the shared metadata backend (VS-14). Replication health/readiness, WAL recovery, gateway
auth/RBAC/mTLS, index engines (HNSW/IVF-PQ + ScaNN simulation), benchmarks (ann, Jepsen,
soak, chaos), helm/k8s/terraform, and a GitHub Pages docs site also exist.

The gap to a *distributed* GA is now precisely one axis: **the network**. Decisions exist;
execution across nodes does not.

## Definition of Done — GA 1.0

A release is **Vsector 1.0.0** only when all of these hold:

1. **Executable routing.** A `REMOTE` routing decision (VS-12) executes over the data
   plane — a remote write is ACKed only after real durable apply, never fabricated.
2. **Real replication + failover.** Replication runs over the network with quorum acks;
   an unavailable primary can be failed over to a caught-up replica through an auditable,
   deterministic path. Restart never fabricates state.
3. **Durable control plane.** Real etcd (3+ nodes) is the default metadata spine with
   liveness leases; migration/membership/ownership commits are linearized (Raft where
   needed). No JSON-file backend in production deployments.
4. **Operational migration.** Automatic rebalancing runs via the validated VS-13 migration
   lifecycle with throttle, pause/resume, and safety limits — never blind ownership moves.
5. **Distributed query.** Cross-node fan-out + deterministic top-K merge with
   `EVENTUAL|STRONG` consistency selection.
6. **Scale claim is measured.** p99 read latency `<50ms @1M vectors` and `<200ms @1T
   vectors`, recall above the documented baseline on ann-benchmarks, verified by soak,
   chaos, and Jepsen as part of the GA gate. Version string is aligned repo-wide.
7. **Durability & recovery.** PostgreSQL metadata, S3 cold tier + CDC, backup/restore/
   PITR documented and exercised.
8. **Security posture.** Per-namespace RBAC scoping, mTLS enforced on all inter-node
   traffic, audit/SOC2/GDPR artifacts aligned with the documented claims.

## Milestones to GA 1.0

### Phase 1 — Control-plane spine (VS-15 → VS-17)

| VS | Milestone | Scope | Exit gate |
|---|---|---|---|
| **VS-15** | Real-etcd metadata spine | Real `Etcd3Store` as default; etcd watch + gossip as primary invalidation; JSON-file only for single-node dev; node liveness lease registration | 3-node etcd, shard/membership/migration metadata survives restarts, watch-driven cache invalidation verified cross-node |
| **VS-16** | Data-plane RPC transport | gRPC transport serving remote write/read/migrate ops; deadlines, retries, idempotency; preserves VS-12 invariants (no ack without apply); mTLS-ready | `REMOTE` decision actually executes against the true owner; no fabricated success paths |
| **VS-17** | Control-plane linearization | Raft log over the metadata spine for membership votes, ownership commits, migration commits; leader-election protocol for the control plane; lease failover for membership | Concurrent ownership/migration commits serialize; a restarted control-plane leader never forks metadata |

### Phase 2 — Data-plane autonomy (VS-18 → VS-21)

| VS | Milestone | Scope | Exit gate |
|---|---|---|---|
| **VS-18** | Replica promotion & failover | Failover decision on health + lease expiry; promotion via the existing `DRAINING`/ownership boundary; never auto-promotes an uncaught-up replica | Kill-primary chaos test: failover converges, no lost acked writes, audit trail complete |
| **VS-19** | Active health probing | `ProbeManager`: live probe/readiness; feeds VS-09 health; gates routing; expires leases on missed beats | Probe-derived health equals runtime health in tests; routing refuses unprobed/expired owners |
| **VS-20** | Rebalance controller | Operationalizes `recommend_rebalance()`: scheduled, throttled migrations with pause/resume, per-namespace policy, safety limits | Rebalance runs unattended, respects limits, uses only the validated migration lifecycle |
| **VS-21** | Distributed query | Cross-node query decision execution, `QueryFanout` + deterministic `MergeTopK`, consistency selection, deadline propagation | Multi-node query returns identical top-K to single-node on same data |

### Phase 3 — Durability & operations (VS-22 → VS-23)

| VS | Milestone | Scope | Exit gate |
|---|---|---|---|
| **VS-22** | PostgreSQL metadata | Replace JSON `MetadataStore` with PostgreSQL + alembic migrations (schema exists); namespaces, quotas, audit, metering | Migrations apply cleanly; metadata survives pod loss; no vector data in SQL |
| **VS-23** | S3 tiering + CDC + DR | Cold SSTables to S3 with lifecycle, CDC capture for DR, backup/restore + PITR runbook, cross-region replay | Restore drill validated; deleted data removed per retention (GDPR) |

### Phase 4 — Tenancy, security, query quality (VS-24 → VS-27)

| VS | Milestone | Scope | Exit gate |
|---|---|---|---|
| **VS-24** | Multi-tenancy | Per-tenant quotas, rate limits, isolation, metering, per-tenant keys; tenant-aware routing | Tenant A cannot exceed quota or read tenant B; metering reconciles |
| **VS-25** | Security to GA | Per-namespace RBAC (OPA scoping), mTLS on every inter-node op, secret rotation, audit alignment | Policy tests + security review pass; no plaintext secrets in transit |
| **VS-26** | Index completeness | Real ScaNN backend or equivalent GPU path; BF16/SQ8 quantization in the index layer; filter-aware ANN variants | Recall/cost numbers published vs competitors on ann-benchmarks |
| **VS-27** | Re-ranking & hybrid | Full cross-encoder + MMR, hybrid BM25/vector, late interaction | Rerank SLOs defined and met; hybrid improves recall over ANN alone |

### Phase 5 — Scale-out validation & GA (VS-28 → VS-29)

| VS | Milestone | Scope | Exit gate |
|---|---|---|---|
| **VS-28** | Scale-out validation | 1M → 1B → 100B → 1T vectors; soak, chaos, Jepsen live; autoscaling + HPA tuning; benchmark report published | p99 `<50ms @1M`, `<200ms @1T`; Jepsen failures zero; soak clean across upgrades |
| **VS-29** | GA hardening & certification | Repo-wide version alignment, SDK feature parity (py/ts/rust/java/go), Helm charts GA, SLOs/alerts/runbooks, security + docs review, release trains | All DoD items above green; tagged `v1.0.0`; docs site current |

## Timeline (indicative — assumes a small team, ~2 FTEs)

| Phase | Milestones | Window | Cumulative |
|---|---|---|---|
| Phase 1 | VS-15…VS-17 | Q4 2026 | Q4 2026 |
| Phase 2 | VS-18…VS-21 | Q1 2027 | Q1 2027 |
| Phase 3 | VS-22…VS-23 | Q2 2027 | Q2 2027 |
| Phase 4 | VS-24…VS-27 | Q3 2027 | Q3 2027 |
| Phase 5 | VS-28…VS-29 | Q4 2027 | **GA 1.0 ~ Q4 2027** |

Dates are anchors, not promises: each milestone is gated by its exit criteria, and a
milestone that cannot prove its gate is not "done." VS-15/VS-16 (real etcd + RPC) and
VS-28 (the measured scale claim) are the two highest-risk items; they should be started
earliest (VS-15) and continuously de-risked (bench perf work can begin immediately).

## Execution principles for the remaining work

1. **Milestones are shippable atoms.** Each VS lands with tests, migration of existing
   tests, docs section, and observability additions.
2. **Invariants precede optimization.** The eight invariants in `docs/ARCHITECTURE.md`
   gate every merge. No "quick" path that bypasses a validated transition.
3. **Transport is one abstraction.** VS-16's RPC is reused for replication, migration,
   and query fan-out; there is exactly one network boundary to secure (VS-25).
4. **Truthfulness is measured.** Every automated failover, migration, and promotion
   leaves an observable record, and restart tests assert nothing is re-invented.
5. **No scale claims without benchmarks.** The Phase 5 gate is a measurement gate.

## Non-goals of GA 1.0 (post-GA vision)

- Multi-writer / multi-primary writes within a shard
- Proprietary GPU-resident index beyond a single TensorFlow-ScaNN path
- Online schema change / in-place re-embedding pipelines
- Fully synchronous cross-region writes (async CDC at GA; synchronous at a future release)
- External data-source connectors beyond the documented SDKs/system integration layer

## See also

- `docs/ARCHITECTURE.md` — present vs target architecture and the capability coverage matrix.
- `README.md` — module spec, config, and the implemented-vs-not tables (kept in sync each milestone).