"""LangChain vectorstore integration for Vsector."""

from __future__ import annotations

from typing import Any

try:
    from langchain.vectorstores.base import VectorStore  # type: ignore
    from langchain.embeddings.base import Embeddings  # type: ignore

    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False

    class VectorStore:  # type: ignore
        pass


class VsectorLangChain(VectorStore):
    """LangChain VectorStore backed by Vsector REST."""

    def __init__(self, client, namespace: str, embedding: Any = None):
        self.client = client
        self.namespace = namespace
        self.embedding = embedding

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None, **kwargs) -> list[str]:
        if self.embedding is None:
            raise ValueError("embedding required")
        vectors = self.embedding.embed_documents(texts)
        payload = [{"vector": v, "metadata": (metadatas[i] if metadatas else {} | {"text": texts[i]})} for i, v in enumerate(vectors)]
        res = self.client.upsert(self.namespace, payload)
        return res.get("ids", [])

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        if self.embedding is None:
            raise ValueError("embedding required")
        qv = self.embedding.embed_query(query)
        res = self.client.query(self.namespace, vector=qv, top_k=k)
        return [(r["metadata"].get("text", r["id"]), r["score"]) for r in res.get("results", [])]

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, client=None, namespace="default", **kwargs):
        vs = cls(client=client, namespace=namespace, embedding=embedding)
        vs.add_texts(texts, metadatas)
        return vs
