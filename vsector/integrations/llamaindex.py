"""LlamaIndex vectorstore integration for Vsector."""

from __future__ import annotations

try:
    from llama_index.vector_stores.types import BasePydanticVectorStore, VectorStoreQuery  # type: ignore

    HAS_LLAMA = True
except Exception:
    HAS_LLAMA = False

    class BasePydanticVectorStore:  # type: ignore
        pass


class VsectorLlamaIndex(BasePydanticVectorStore):
    stores_text = True

    def __init__(self, client, namespace: str = "default"):
        self.client = client
        self.namespace = namespace
        super().__init__()

    def add(self, nodes, **kwargs):
        vectors = [n.embedding for n in nodes]
        payload = [{"id": n.node_id, "vector": v, "metadata": n.metadata | {"text": n.text}} for n, v in zip(nodes, vectors)]
        self.client.upsert(self.namespace, payload)
        return [n.node_id for n in nodes]

    def query(self, query, **kwargs):
        res = self.client.query(self.namespace, vector=query.query_embedding, top_k=query.similarity_top_k)
        # map to LlamaIndex response stub
        from llama_index.vector_stores.types import VectorStoreQueryResult  # type: ignore

        ids = [r["id"] for r in res.get("results", [])]
        sims = [r["score"] for r in res.get("results", [])]
        return VectorStoreQueryResult(nodes=None, similarities=sims, ids=ids)
