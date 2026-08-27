"""Lightweight Python SDK for Vsector - mirrors External System Communication Layer."""
from __future__ import annotations

import httpx

class VsectorClient:
    """REST + gRPC ready client.

    Example:
        client = VsectorClient(base_url="http://localhost:8080/v1", api_key="test-key")
        client.create_namespace(name="products", dimension=1536)
        client.upsert(namespace="products", vectors=[{"id": "...", "vector": [...], "metadata": {}}])
        res = client.query(namespace="products", vector=[...], top_k=10)
    """

    def __init__(self, base_url: str = "http://localhost:8080/v1", api_key: str | None = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout, headers={"X-API-Key": api_key} if api_key else {})

    def create_namespace(self, name: str, dimension: int, distance_metric: str = "COSINE", index_type: str = "HNSW", **kwargs):
        r = self._client.post(f"{self.base_url}/namespaces", json={"name": name, "dimension": dimension, "distance_metric": distance_metric, "index_type": index_type, **kwargs})
        r.raise_for_status()
        return r.json()

    def delete_namespace(self, name: str):
        r = self._client.delete(f"{self.base_url}/namespaces/{name}")
        r.raise_for_status()
        return r.json()

    def stats(self, name: str):
        r = self._client.get(f"{self.base_url}/namespaces/{name}/stats")
        r.raise_for_status()
        return r.json()

    def upsert(self, namespace: str, vectors: list[dict], idempotency_key: str | None = None):
        body = {"namespace": namespace, "vectors": vectors}
        if idempotency_key:
            body["idempotency_key"] = idempotency_key  # type: ignore
        r = self._client.post(f"{self.base_url}/vectors/upsert", json=body)
        r.raise_for_status()
        return r.json()

    def query(self, namespace: str, vector: list[float], top_k: int = 10, filters: dict | None = None, **kwargs):
        body = {"namespace": namespace, "vector": vector, "top_k": top_k, "filters": filters or {}, **kwargs}
        r = self._client.post(f"{self.base_url}/vectors/query", json=body)
        r.raise_for_status()
        return r.json()

    def fetch(self, namespace: str, ids: list[str]):
        r = self._client.post(f"{self.base_url}/vectors/fetch", json={"namespace": namespace, "ids": ids})
        r.raise_for_status()
        return r.json()

    def delete(self, namespace: str, ids: list[str] | None = None, filter: dict | None = None):
        r = self._client.post(f"{self.base_url}/vectors/delete", json={"namespace": namespace, "ids": ids, "filter": filter})
        r.raise_for_status()
        return r.json()

    def health(self):
        r = self._client.get("http://localhost:8080/health")
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()
