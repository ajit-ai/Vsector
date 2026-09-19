Vsector Architecture — Present State & Target (GA 1.0)
=======================================================

Executive summary
-----------------

Vsector is a distributed vector database whose design centers on one idea:

   **Nothing is fabricated.** A write that was not durably applied is never ACKed,
   a node that is not a member is never routed to, a replica that has not caught up
   is never promoted, and a restart never invents health, ownership, or success.

Everything in VS-01…VS-14 exists to make that statement *provable*: validated state
machines (shards, membership, migrations), deterministic digest checks, truthful
restart behavior, and routing decisions that answer *which node, why, and whether*
this node may act. GA 1.0 closes the gap between "the cluster can prove what it
should do" and "the cluster can actually do it across a network."

This document describes (1) the architecture that exists today, (2) the target
GA architecture, and (3) the scope coverage matrix that shows, capability by
capability, what changes between now and GA 1.0.

Design invariants (carried from VS-10…VS-14)
--------------------------------------------

These invariants gate every future milestone. A milestone that violates one is not done.

.. list-table::
   :header-rows: 1

   * - Invariant
     - Meaning
     - Established by
   * - Single authoritative ownership
     - ``shard.node_id`` is the only primary owner; ownership changes only at an explicit boundary
     - VS-10, VS-13
   * - Validated transitions
     - No arbitrary state jumps: shard / membership / migration transitions raise deterministic errors
     - VS-10, VS-11, VS-13
   * - No fabricated success
     - A local write to a ``REMOTE``/unsafe shard never returns success; ack reflects real apply
     - VS-09, VS-10, VS-12
   * - Restart truthfulness
     - Persisted state is restored; nothing is invented (no healthy replicas, no auto-failover)
     - VS-10, VS-11, VS-13
   * - Side-effect-free routing
     - Decisions never mutate shard state, ownership, membership, or files
     - VS-12
   * - Deterministic operations
     - Read-only decisions/recommendations are deterministic, tie-broken by id
     - VS-12, VS-13
   * - One metadata model
     - Shards, cluster nodes, and migrations share one store (reserved prefixes)
     - VS-11, VS-13/14
   * - Membership ≠ health ≠ ownership
     - Three independent axes; one never auto-changes another
     - VS-09, VS-11

Two-plane model
---------------

GA splits the system into a **control plane** (slow, small, consistent — *decides*) and
a **data plane** (fast, large, replicated — *acts*). Today the codebase is one process
with both planes colocated; the split is architectural naming for where routing
intelligence lives versus where records/indexes live, not a deployment topology.

.. code-block:: text
   :caption: Control plane (decide) vs. data plane (act)

                   ┌────────────────────────────────────────────────┐
      Clients ────▶│  Gateway / API layer (REST, gRPC, Kafka, WS)  │
                   └───────┬──────────────────────────────┬────────┘
                           │                              │
              ┌────────────▼────────────┐    ┌────────────▼────────────┐
              │ CONTROL PLANE          │    │ DATA PLANE              │
              │ decides                │    │ acts                    │
              │                        │    │                        │
              │ Metadata spine         │    │ IngestService (write)  │
              │  - etcd(+Raft)(VS-15/17)│   │ QueryEngine (read)     │
              │  - Shard lifecycle      │    │ Replication transport  │
              │  - Membership (VS-11)   │    │  (VS-16)               │
              │  - Placement (VS-12)    │    │ Migration transport    │
              │  - Migration (VS-13/14) │    │  (VS-13 local → VS-20  │
              │  - Gossip (VS-15)       │    │   auto)                │
              │  - Rebalance scheduler  │    │ Index (HNSW/IVF-PQ/    │
              │    (VS-20)              │    │  ScaNN)(VS-26)         │
              │                        │    │ WAL + Segments + S3    │
              └────────────▲────────────┘    │  (VS-23)              │
                           │                 └────────────┬──────────┘
                           └──────────── duplicates ──────┘  (replication)

**Control plane** owns *truth*: what shards exist, who owns them, who is a member,
which migrations are running. **Data plane** owns *mass*: vectors, indexes, WAL,
cache. The only overlap is that each node hosts both a copy of the control-plane
metadata reader and its slice of the data plane.

Today: what is implemented (VS-01 → VS-14)
------------------------------------------

.. list-table::
   :header-rows: 1

   * - Layer
     - Capability
     - Status
   * - Client/API
     - REST ``/v1``, gRPC, Kafka, WebSocket, SDKs (py/ts/rs/java/go)
     - Implemented
   * - Gateway
     - API-Key / JWT, rate limit, circuit breaker, OPA RBAC, mTLS material
     - Implemented (mTLS is cert layer, not yet wired to inter-node transport)
   * - Data model
     - ``VectorRecord``, ``Namespace`` (metrics, compression, ttl, tags)
     - Implemented
   * - Sharding
     - HRW placement, ``RendezvousHash``, lifecycle-aware router, gossip invalidation stub
     - Implemented
   * - Ownership
     - ``CREATING→ACTIVE→DRAINING→OFFLINE``, primary owner + replicas
     - Implemented (VS-10)
   * - Cluster
     - ``JOINING→ACTIVE→DRAINING→REMOVED``, cluster/node identity, validation vs shard ownership
     - Implemented (VS-11)
   * - Placement
     - ``place()/route_read/write/query_*``, ``LOCAL/REMOTE/UNAVAILABLE`` decisions, read-only endpoints
     - Implemented (VS-12)
   * - Migration
     - Explicit ``PENDING→…→COMPLETED`` lifecycle, digest-verified real data movement over per-node storage roots
     - Implemented (VS-13)
   * - Durable migration store
     - Migrations persisted under reserved prefix, restart-truthful, ``recommend_rebalance()`` (read-only)
     - Implemented (VS-14)
   * - Index
     - Flat, HNSW (hnswlib), IVF-PQ, ScaNN **simulation**, quantization stubs (SQ8/BF16)
     - Partially real
   * - Storage
     - WAL (segmented, checksummed), SegmentStore (MemTable/SSTable), recovery, JSON metadata store
     - Implemented at single-node scale
   * - Replication
     - In-process replica state, quorum-ack via a **transport abstraction**; health/readiness, heal-on-real-delivery
     - Partially real — no network
   * - Consensus, failover, discovery, auto-rebalance, distributed query
     - —
     - **Not implemented**
   * - Metadata durable backbone
     - ``InMemoryEtcd(path=…)`` JSON file + optional real ``Etcd3Store``
     - Partial — real etcd is optional/opt-in

What that means
~~~~~~~~~~~~~~~

The current build is a **single-process, multi-node-semantic** system: it models a
cluster correctly (membership, ownership, placement, migration) and proves the
model over real durable local state, but executing a ``REMOTE`` decision, driving
replication over a socket, failing a primary over, or merging results from
several nodes requires the network that GA 1.0 adds. Making the ``REMOTE`` decision
executable is exactly VS-16 → VS-21.

Target GA architecture (scope coverage)
---------------------------------------

The tables below define the GA target per subsystem. Each row is a capability;
"Today" is the VS-14 state; "GA 1.0" is the acceptance target.

Control plane (decide)
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Capability
     - Today
     - GA 1.0
   * - Metadata spine
     - JSON-file ``InMemoryEtcd`` default; real etcd3 optional
     - Real etcd3 (3+ nodes) is the default deployment; JSON file only for single-node dev
   * - Cache invalidation
     - Gossip watch + poll timer
     - etcd watch as primary; gossip + poll fallback (VS-15)
   * - Metadata linearization
     - Single-process lock; blind put for concurrent writers
     - Raft for membership votes, migration commits, ownership changes (VS-17)
   * - Node liveness
     - Static membership; no probes
     - Lease-based liveness (etcd lease), expiry = membership signal, fed to health (VS-19)
   * - Auto-rebalance
     - Read-only ``recommend_rebalance()``
     - Scheduler executing the validated migration lifecycle with throttle/pause/resume (VS-20)
   * - Distributed service discovery
     - None (violates "no fabricated membership")
     - Coordinated registration through the spine; no mDNS/hardcoding (VS-15/17)

Data plane (act)
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Capability
     - Today
     - GA 1.0
   * - Remote writes
     - ``REMOTE`` decision raises; no fabrication
     - gRPC data-plane RPC forwards to the owner; ack only on real apply + quorum (VS-16)
   * - Replication
     - In-process transport abstraction
     - Real network transport with backpressure, deadlines, retries, idempotency (VS-16)
   * - Replica promotion / failover
     - Explicitly non-implemented
     - Auditable promotion via health + lease expiry; promotion only after caught-up state (VS-18)
   * - Background health probing
     - Runtime-derived only
     - Active probes; readiness gates routing; leases expire on missed beats (VS-19)
   * - Distributed query
     - Local fan-out only
     - Cross-node fan-out + deterministic top-K merge with consistency selection (VS-21)
   * - Cold storage
     - Local segments
     - S3 tiering + lifecycle + CDC for DR replay (VS-23)

Index & query quality
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Capability
     - Today
     - GA 1.0
   * - HNSW / IVF-PQ
     - Real
     - Real, filter-aware variants verified on ann-benchmarks
   * - ScaNN
     - Simulation wrapper
     - Real backend (TensorFlow-based) or equivalent GPU path (VS-26)
   * - Quantization
     - Stubs
     - BF16 / SQ8 (and INT8 where supported) in the index layer (VS-26)
   * - Re-ranking
     - Cross-encoder/MMR stub
     - Full second-stage + hybrid BM25/vector (VS-27)

Housing & platform
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Capability
     - Today
     - GA 1.0
   * - Metadata store (SQL)
     - JSON ``MetadataStore``
     - PostgreSQL + alembic migrations (VS-22)
   * - Multi-tenancy
     - Single-tenant namespaces
     - Per-tenant quotas, rate limits, isolation, metering, per-tenant keys (VS-24)
   * - RBAC
     - OPA rego at gateway
     - Per-namespace scoping; policy evaluation in data-plane path (VS-25)
   * - Inter-node security
     - mTLS material exists
     - mTLS mandatory on all data-plane RPC; secret rotation (VS-25)
   * - Observability
     - Prometheus/Grafana/Jaeger stub/ELK
     - Distributed tracing end-to-end, SLOs with alerts, runbooks (VS-28/29)
   * - DR
     - None
     - S3 + CDC, backup/restore, PITR, cross-region replay (VS-23)

Key flows at GA
---------------

Write path (remote-capable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Client → Gateway (auth/rate/RBAC) → IngestService → placement (VS-12) → LOCAL/REMOTE
   → [LOCAL] WAL → primary index → replicate to followers over network (VS-16) → quorum
   ack → ACK

A ``REMOTE`` decision is forwarded to the current primary owner over the
data-plane RPC; the local node never fabricates the ack. Ownership-commit conflicts
(concurrent migration) surface as deterministic errors, never silent reroute.

Migration / rebalance (VS-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Rebalance scheduler picks candidate (VS-13 recommend_rebalance) → create_migration →
   prepare → start_copy (network transfer / export→import → digest verify) → verify →
   commit (ownership in shard record + Raft) → finalize_source

The scheduler is explicitly **a caller of the validated migration lifecycle**, never a
mechanism that moves ownership directly.

Failover (VS-18)
~~~~~~~~~~~~~~~~

Replica health (VS-09) + lease expiry (VS-19) → membership signal → explicit,
operator-visible promotion decision → shard enters the existing ``DRAINING`` path →
owner moves to a caught-up replica via the ownership boundary. Restart never replays
this; state is always re-derived.

New components introduced between now and GA
--------------------------------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Module (target)
     - Purpose
   * - ``RpcTransport`` (data plane)
     - ``vsector/transport/grpc.py``
     - Forward/mutate operations by routed decision; one transport for replication + migration + query fan-out
   * - ``LeaseRegistry``
     - ``vsector/cluster/leases.py``
     - etcd-lease-backed node liveness feeding membership + health
   * - ``RaftMetadata``
     - ``vsector/control/raft.py``
     - Linearized control-plane commit log (membership votes, ownership commits, migration commits)
   * - ``RebalanceController``
     - ``vsector/control/rebalance.py``
     - Drives VS-13 migrations with policy (throttle, limits, pause/resume)
   * - ``ProbeManager``
     - ``vsector/cluster/probes.py``
     - Active health/readiness probes maintained as runtime-derived health
   * - ``QueryFanout`` / ``MergeTopK``
     - ``vsector/query/distributed.py``
     - Cross-node read fan-out + deterministic global merge
   * - ``S3Tier``
     - ``vsector/storage/s3.py`` (existing module, extended)
     - Cold segment lifecycle + CDC
   * - ``TenantManager``
     - ``vsector/models/tenant.py``
     - Quotas, isolation, metering, keys

Anti-goals at GA 1.0
--------------------

* A *silent* anything: promotion, failover, migration, or ownership change that
  "just happens" with no observable record is a bug, not a feature.
* Bespoke consensus replacing etcd: where etcd gives a linearized, durable,
  watchable store, Vsector uses it; Raft is used only where etcd semantics are
  insufficient (multi-key linearizable commits).
* No hardcoding of node topology; discovery stays coordinated, never random.
* Scale is not claimed until it is measured: the p99 targets (``<50ms @1M``,
  ``<200ms @1T``) gate the GA certification, not the roadmap.

See also
--------

* :doc:`roadmap` — milestone-by-milestone plan with a timeline to GA 1.0.
* :doc:`index` — layered view, module spec, config, and the implemented-vs-not tables.