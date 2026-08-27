from fastapi.testclient import TestClient
from vsector.api.rest import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code==200

def test_namespace_flow():
    # create
    r = client.post("/v1/namespaces", json={"name":"api_ns","dimension":4}, headers={"X-API-Key":"test-key"})
    assert r.status_code in (200,409)
    # stats
    r = client.get("/v1/namespaces/api_ns/stats")
    assert r.status_code==200
    # upsert
    r = client.post("/v1/vectors/upsert", json={"namespace":"api_ns","vectors":[{"vector":[1,0,0,0],"metadata":{"t":"a"}}]}, headers={"X-API-Key":"test-key"})
    assert r.status_code==200
    assert r.json()["upserted"]==1
    # query
    r = client.post("/v1/vectors/query", json={"namespace":"api_ns","vector":[1,0,0,0],"top_k":1}, headers={"X-API-Key":"test-key"})
    assert r.status_code==200
    assert "results" in r.json()
