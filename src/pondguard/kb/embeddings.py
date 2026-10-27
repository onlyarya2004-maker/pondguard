"""Text embedders for the knowledge base.

Two interchangeable embedders:
- `BgeEmbedder`: BAAI/bge-small-en-v1.5 via sentence-transformers, on CPU. The real
  one. Downloaded once into the project's cache/huggingface folder.
- `HashEmbedder`: a deterministic word-hashing embedder. No model, no download. Used
  by tests and CI (mock mode) so retrieval can be exercised offline. It matches
  shared words only, so it is NOT a stand-in for retrieval quality.

Each embedder has a `name`; the Chroma collection name includes it, so the two
never mix in one store.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeEmbedder:
    name = "bge"

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(BGE_MODEL, device="cpu")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._load().encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True)
        return vector.tolist()


class HashEmbedder:
    name = "hash"
    DIM = 512

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.DIM)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vec[int.from_bytes(digest[:4], "little") % self.DIM] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embedder(kind: str) -> Embedder:
    if kind == "bge":
        return BgeEmbedder()
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedder: {kind}")
