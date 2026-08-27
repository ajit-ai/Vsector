from vsector.sharding.hash_ring import RendezvousHash
from vsector.sharding.router import ShardRouter, EtcdStore
from vsector.sharding.shard import Shard

def test_hrw():
    h = RendezvousHash(["a","b","c"])
    assert h.get_node("key1") in ["a","b","c"]
    top2 = h.get_nodes("key1", 2)
    assert len(top2)==2

def test_router():
    etcd = EtcdStore()
    router = ShardRouter(etcd, cache_ttl_s=1)
    s1 = Shard(namespace="ns", node_id="n1")
    s2 = Shard(namespace="ns", node_id="n2")
    router.register_shard(s1); router.register_shard(s2)
    picked = router.route("ns", "record-123")
    assert picked.id in [s1.id, s2.id]
    all_shards = router.route_for_query("ns")
    assert len(all_shards)==2
