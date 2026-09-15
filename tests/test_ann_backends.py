"""VS-09.1: ANN correctness tests — ef_search typing, backend identity, native fallback."""
from __future__ import annotations

import uuid

import pytest
import numpy as np

from vsector.index.factory import create_index
from vsector.index.hnsw import HAS_HNSWLIB
from vsector.index.ivf_pq import HAS_FAISS
from vsector.storage.metadata import MetadataStore
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard
from vsector.models.namespace import Namespace, DistanceMetric, IndexType
from vsector.ingest.service import IngestService
from vsector.query.engine import QueryEngine


def _vecs(n: int, dim: int = 4, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.tolist()


def _uid(i) -> str:
    return str(uuid.UUID(int=i))


def _ids(n: int, seed_i: int = 0) -> list[str]:
    return [_uid(seed_i + i + 1) for i in range(n)]


def _stack(base: str, ns_name: str = "ann1", index_type: IndexType = IndexType.FLAT, dim: int = 4):
    md = MetadataStore(path=f"{base}/metadata.json")
    etcd = EtcdStore()
    router = ShardRouter(etcd)
    ingest = IngestService(md, router, base_dir=base)
    query = QueryEngine(md, router, ingest)
    ns = Namespace(name=ns_name, dimension=dim, index_type=index_type, distance_metric=DistanceMetric.COSINE)
    md.create(ns)
    router.register_shard(Shard(namespace=ns_name, node_id="n1", id="ann-shard"))
    return ingest, query, ns


def _ctx(ingest: IngestService, ns: str):
    shard = ingest.router.route(ns, "some-record-id")
    return ingest._get_or_create_ctx(ns, shard)


class _SpyIndex:
    """Records ef_search/nprobe received, asserts ef_search is never an enum."""

    def __init__(self, real):
        self.real = real
        self.ef_search_calls: list = []

    def __getattr__(self, item):
        return getattr(self.real, item)

    def search(self, query, top_k=10, ef_search=None, nprobe=None, filter_fn=None):
        assert not isinstance(ef_search, (str, IndexType)), f"ef_search must be numeric or None, got {type(ef_search)!r}"
        self.ef_search_calls.append(ef_search)
        return self.real.search(query, top_k=top_k, ef_search=ef_search, nprobe=nprobe, filter_fn=filter_fn)


# --- backend identity -------------------------------------------------------

def test_flat_backend_identity():
    idx = create_index("FLAT", dimension=4, metric="cosine")
    assert idx.backend_name == "FlatIndex"
    assert idx.is_native_backend is True
    assert idx.degraded is False


def test_hnsw_backend_identity_reflects_actual_backend():
    idx = create_index("HNSW", dimension=4, metric="cosine")
    if HAS_HNSWLIB:
        assert idx.backend_name == "HNSW"
        assert idx.is_native_backend is True
        assert idx.degraded is False
    else:
        assert idx.backend_name == "FlatIndex"  # honest: not falsely reported as HNSW
        assert idx.is_native_backend is False
        assert idx.degraded is True


def test_ivf_pq_backend_identity_reflects_actual_backend():
    idx = create_index("IVF_PQ", dimension=8, metric="cosine")
    if HAS_FAISS:
        assert idx.backend_name == "FAISS"
        assert idx.is_native_backend is True
    else:
        assert idx.backend_name == "FlatIndex"
        assert idx.is_native_backend is False
        assert idx.degraded is True


def test_scann_backend_identity_is_simulated():
    idx = create_index("SCANN", dimension=4, metric="cosine")
    assert idx.backend_name == "FlatIndex"  # no real ScaNN backend
    assert idx.is_native_backend is False
    assert idx.degraded is True


def test_factory_never_reports_fake_native():
    for t in ("HNSW", "IVF_PQ"):
        idx = create_index(t, dimension=4, metric="cosine")
        if idx.degraded:
            assert idx.backend_name == "FlatIndex"
            assert idx.is_native_backend is False


def test_stats_expose_backend_identity(tmp_path):
    from vsector.models.namespace import IndexType as IT

    ingest, _, _ = _stack(str(tmp_path), index_type=IT.HNSW)
    ctx = _ctx(ingest, "ann1")
    ctx.index.add(["a"], np.zeros((1, 4), dtype=np.float32), [{}])
    stats = ingest.stats()
    key = "ann1:ann-shard"
    assert key in stats
    assert stats[key]["backend_name"] in ("HNSW", "FlatIndex")
    assert stats[key]["is_native_backend"] is (stats[key]["backend_name"] == "HNSW")


# --- ef_search typing (never pass IndexType enum) ---------------------------

@pytest.mark.asyncio
async def test_ef_search_default_is_numeric_and_never_enum(tmp_path):
    base = str(tmp_path)
    ingest, query, ns = _stack(base)
    vecs = _vecs(8)
    ids = _ids(8)
    await ingest.upsert("ann1", [{"id": ids[i], "vector": vecs[i]} for i in range(8)])
    ctx = _ctx(ingest, "ann1")
    ctx.index = _SpyIndex(ctx.index)  # type: ignore

    # default path (ef_search=None -> numeric 100), must not pass an enum
    r1 = await query.query("ann1", vecs[0], top_k=3, include_vector=True)  # bypass cache
    assert len(r1["results"]) == 3
    # explicit numeric
    r2 = await query.query("ann1", vecs[1], top_k=2, ef_search=50, include_vector=True)
    assert len(r2["results"]) == 2

    calls = ctx.index.ef_search_calls  # type: ignore
    assert calls, "search() must be reached"
    for c in calls:
        assert isinstance(c, int) or c is None


@pytest.mark.asyncio
async def test_query_with_explicit_ef_search_works(tmp_path):
    base = str(tmp_path)
    ingest, query, ns = _stack(base, index_type=IndexType.FLAT)
    vecs = _vecs(6)
    ids = _ids(6)
    await ingest.upsert("ann1", [{"id": ids[i], "vector": vecs[i]} for i in range(6)])
    for ef in (None, 10, 25, 200):
        r = await query.query("ann1", vecs[0], top_k=2, ef_search=ef, include_vector=True)
        assert len(r["results"]) == 2


# --- deterministic NN correctness -------------------------------------------

@pytest.mark.asyncio
async def test_flat_returns_deterministic_exact_nn(tmp_path):
    base = str(tmp_path)
    ingest, query, _ = _stack(base)
    vecs = _vecs(12, seed=42)
    ids = _ids(12)
    await ingest.upsert("ann1", [{"id": ids[i], "vector": vecs[i]} for i in range(12)])
    # brute-force ground truth via numpy
    q = np.array(vecs[0])
    expected = np.argsort(-(np.vstack(vecs) @ q))[:3].tolist()
    out = await query.query("ann1", vecs[0], top_k=3, include_vector=True)
    got = [ids.index(r["id"]) for r in out["results"]]
    assert got == expected
    # deterministic across repeated calls
    out2 = await query.query("ann1", vecs[0], top_k=3, include_vector=True)
    assert [r["id"] for r in out2["results"]] == [r["id"] for r in out["results"]]


@pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed (native HNSW test)")
def test_native_hnsw_returns_correct_nearest_neighbors():
    dim = 8
    idx = create_index("HNSW", dimension=dim, metric="cosine")
    assert idx.backend_name == "HNSW"
    vecs = _vecs(200, dim=dim, seed=7)
    arr = np.array(vecs, dtype=np.float32)
    idx.add([f"v{i}" for i in range(200)], arr)
    # top-1 recall vs exact brute force for every vector
    hits = 0
    gt = np.vstack(vecs)
    for i in range(200):
        q = arr[i]
        expected_top = int(np.argsort(-(gt @ q))[:1][0])
        res = idx.search(np.asarray(q), top_k=1)
        hit = int(res[0].id.lstrip("v")) if res else -1
        if hit == expected_top:
            hits += 1
    assert hits >= 195, f"native HNSW top-1 recall too low: {hits}/200"


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed (native FAISS IVF-PQ test)")
def test_native_faiss_ivf_pq_returns_correct_nearest_neighbors():
    dim = 8
    idx = create_index("IVF_PQ", dimension=dim, metric="cosine")
    assert idx.backend_name == "FAISS"
    vecs = _vecs(200, dim=dim, seed=11)
    arr = np.array(vecs, dtype=np.float32)
    idx.add([f"v{i}" for i in range(200)], arr)
    idx.search  # train triggered internally
    gt = np.vstack(vecs)
    q = arr[0]
    expected_top = int(np.argsort(-(gt @ q))[:1][0])
    res = idx.search(np.asarray(q), top_k=1)
    assert res and int(res[0].id.lstrip("v")) == expected_top