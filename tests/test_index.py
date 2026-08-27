import numpy as np
from vsector.index.flat import FlatIndex
from vsector.index.hnsw import HNSWIndex
from vsector.index.factory import create_index

def test_flat():
    idx = FlatIndex(dimension=4)
    vecs = np.random.randn(5,4).astype(np.float32)
    idx.add([f"id{i}" for i in range(5)], vecs, [{"k":i} for i in range(5)])
    res = idx.search(np.random.randn(4), top_k=2)
    assert len(res)==2

def test_hnsw():
    idx = HNSWIndex(dimension=4, M=16)
    vecs = np.random.randn(10,4).astype(np.float32)
    idx.add([f"h{i}" for i in range(10)], vecs)
    res = idx.search(np.random.randn(4), top_k=3)
    assert len(res) <=3

def test_factory():
    assert create_index("HNSW", 4)
    assert create_index("FLAT", 4)
