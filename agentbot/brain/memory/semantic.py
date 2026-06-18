"""Long-term semantic memory (RAG) — vector store over past episodes / skill docs.

Uses Chroma if installed (``agentbot[vector]``); otherwise a no-op shim so the
scaffold imports and runs without the heavy dependency. Real recall is wired in
the MVP/Phase-2 work.
"""
from __future__ import annotations

from typing import Optional


class SemanticMemory:
    def __init__(self, backend: str = "none", path: str = "agentbot/var/chroma") -> None:
        self.backend = backend
        self._col = None
        if backend == "chroma":
            try:
                import chromadb
                client = chromadb.PersistentClient(path=path)
                self._col = client.get_or_create_collection("agentbot")
            except Exception:  # noqa: BLE001 - optional dependency; degrade to no-op
                self._col = None

    @property
    def enabled(self) -> bool:
        return self._col is not None

    def add(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        if self._col is not None:
            self._col.add(ids=[doc_id], documents=[text], metadatas=[metadata or {}])

    def query(self, text: str, k: int = 3) -> list[str]:
        if self._col is None:
            return []
        res = self._col.query(query_texts=[text], n_results=k)
        docs = res.get("documents") or [[]]
        return docs[0] if docs else []
