from __future__ import annotations

from pathlib import Path
import time
from typing import Protocol

import numpy as np

from vector_store.config import Settings
from vector_store.errors import VectorStoreError
from vector_store.faiss_store import FaissVectorStore


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SemanticRetriever:
    def __init__(
        self,
        store_dir: Path,
        embedder: EmbeddingClient,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = embedder
        self.store = FaissVectorStore.load(store_dir, self.settings.embedding_dimension)

    def retrieve(self, query: str, top_k: int | None = None) -> dict[str, object]:
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreError("EMPTY_QUERY", "query must be a non-empty string", 422)
        k = self.settings.initial_top_k if top_k is None else top_k
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise VectorStoreError("EMPTY_QUERY", "top_k must be a positive integer", 422)
        try:
            embedding_started = time.perf_counter()
            vectors = self.embedder.embed([query.strip()])
            embedding_elapsed_ms = (time.perf_counter() - embedding_started) * 1000
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("EMBEDDING_FAILURE", f"Could not embed query: {exc}", 500) from exc
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (1, self.settings.embedding_dimension):
            raise VectorStoreError(
                "INVALID_EMBEDDING_DIMENSION",
                f"Expected query vector dimension {self.settings.embedding_dimension}, got {matrix.shape}",
                500,
            )
        faiss_started = time.perf_counter()
        matches = self.store.search(matrix[0], k)
        faiss_elapsed_ms = (time.perf_counter() - faiss_started) * 1000
        results = []
        for score, metadata in matches:
            results.append({**metadata, "score": score})
        return {
            "query": query.strip(),
            "count": len(results),
            "results": results,
            "timings_ms": {
                "query_embedding": float(getattr(self.embedder, "last_latency_ms", None) or embedding_elapsed_ms),
                "faiss_search": faiss_elapsed_ms,
            },
        }
