"""REST client example (requires server running: vsector serve)."""
from vsector.api.sdk import VsectorClient
import numpy as np

client = VsectorClient(api_key="test-key")
try:
    client.create_namespace("demo", dimension=4)
    print("namespace created")
except Exception as e:
    print("create ns:", e)

res = client.upsert("demo", [{"vector": np.random.randn(4).tolist(), "metadata": {"k": "v"}} for _ in range(5)])
print("upsert", res)

out = client.query("demo", vector=np.random.randn(4).tolist(), top_k=2)
print("query", out)
