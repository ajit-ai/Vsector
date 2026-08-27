"""Prometheus metrics."""
from prometheus_client import Counter, Histogram, Gauge

INGEST_COUNTER = Counter("vsector_ingest_total", "Total ingested vectors", ["namespace", "status"])
QUERY_COUNTER = Counter("vsector_query_total", "Total queries", ["namespace"])
QUERY_LATENCY = Histogram("vsector_query_latency_ms", "Query latency ms", ["namespace"], buckets=[5,10,25,50,100,200,500,1000])
INDEX_SIZE = Gauge("vsector_index_vectors", "Vectors per shard", ["namespace", "shard_id", "index_type"])
WAL_SIZE = Gauge("vsector_wal_bytes", "WAL bytes", ["shard_id"])
SHARD_COUNT = Gauge("vsector_shard_count", "Shard count", ["namespace"])
