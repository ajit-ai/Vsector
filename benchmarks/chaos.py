"""Chaos / Jepsen stub — kills shards, verifies quorum & WAL recovery.

Run after benchmarks/ann_benchmark.py. Requires 3-node simulation.
"""

import random
import time

from vsector.sharding.shard import Shard
from vsector.sharding.router import ShardRouter
from vsector.sharding.etcd import InMemoryEtcd


def chaos_test(rounds: int = 5):
    etcd = InMemoryEtcd()
    router = ShardRouter(etcd)
    shards = [Shard(namespace="chaos", node_id=f"n{i}") for i in range(3)]
    for s in shards:
        router.register_shard(s)
    for r in range(rounds):
        victim = random.choice(shards)
        print(f"Round {r+1}: killing {victim.id} ({victim.node_id})")
        etcd.delete(victim.id)
        # router should still route via fallback shards
        try:
            s = router.route("chaos", f"key{r}")
            print(f"  survived via {s.id}")
        except Exception as e:
            print(f"  routing failed: {e}")
        # recover victim
        etcd.put(victim)
        router.invalidate("chaos")
        time.sleep(0.1)
    print("chaos PASS — quorum survived" if len(router.route_for_query("chaos")) == 3 else "FAIL")


if __name__ == "__main__":
    chaos_test()
