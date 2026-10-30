"""Retrieval over the disease knowledge base: query text in, cited `Passage`s out.

Agentic concept: the retrieval half of agentic RAG. The Diagnostician decides what
to ask (and when to ask again with a rewritten query); this class only fetches. Each
returned Passage carries the document id, title and page that a citation points to.
"""

from __future__ import annotations

from pathlib import Path

from pondguard.kb.embeddings import Embedder
from pondguard.kb.ingest import collection_name, open_store
from pondguard.models import Passage
from pondguard.paths import DATA_DIR


class Retriever:
    def __init__(self, embedder: Embedder, store_dir: Path | None = None) -> None:
        self.embedder = embedder
        self._client = open_store(store_dir or DATA_DIR / "chroma")
        self._name = collection_name(embedder)
        try:
            self._collection = self._client.get_collection(self._name)
        except Exception as exc:
            raise RuntimeError(
                f"knowledge base '{self._name}' not found; run `pondguard ingest` first"
            ) from exc

    def search(self, query: str, k: int = 5, disease: str | None = None) -> list[Passage]:
        where = {"disease": disease} if disease else None
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where=where,
        )
        passages = []
        for pid, text, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            passages.append(
                Passage(
                    id=pid,
                    text=text,
                    source=meta["source"],
                    title=meta["title"],
                    disease=None if meta["disease"] == "none" else meta["disease"],
                    section=None,
                    page=int(meta["page"]),
                    score=round(1.0 - float(dist), 4),
                )
            )
        return passages

    def texts(self, ids: list[str]) -> dict[str, str]:
        """Passage text by id (for the judge; the text is never written to the results)."""
        if not ids:
            return {}
        got = self._collection.get(ids=list(dict.fromkeys(ids)))
        return dict(zip(got["ids"], got["documents"], strict=True))

    def count(self) -> int:
        return self._collection.count()
