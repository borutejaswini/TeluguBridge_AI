from __future__ import annotations

import json
import urllib.error
import urllib.request
import time
from typing import Any

import numpy as np

from vector_store.errors import VectorStoreError


class BgeEmbeddingClient:
    """Client for the existing long-lived CPU BGE-M3 service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8005", timeout_seconds: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.last_latency_ms: float | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise VectorStoreError("EMPTY_QUERY", "Embedding input must contain non-empty text", 422)
        if len(texts) > 32:
            raise VectorStoreError("EMBEDDING_FAILURE", "BGE-M3 service accepts at most 32 texts per request", 422)
        payload = json.dumps({"texts": texts}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data: Any = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise VectorStoreError("EMBEDDING_FAILURE", f"BGE-M3 service returned HTTP {exc.code}: {detail}", 503) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError("EMBEDDING_FAILURE", f"Could not get embeddings from {self.base_url}: {exc}", 503) from exc
        if not isinstance(data, dict):
            raise VectorStoreError("EMBEDDING_FAILURE", "BGE-M3 service returned an invalid response", 503)
        if data.get("model") != "BAAI/bge-m3":
            raise VectorStoreError("EMBEDDING_FAILURE", f"Expected BAAI/bge-m3, got {data.get('model')!r}", 503)
        if data.get("device") != "cpu" or data.get("dimension") != 1024 or data.get("count") != len(texts):
            raise VectorStoreError("EMBEDDING_FAILURE", "BGE-M3 service response configuration/count mismatch", 503)
        try:
            vectors = np.asarray(data["embeddings"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError("EMBEDDING_FAILURE", f"BGE-M3 returned malformed embeddings: {exc}", 503) from exc
        if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
            raise VectorStoreError(
                "INVALID_EMBEDDING_DIMENSION", f"Expected finite ({len(texts)}, 1024) embeddings, got {vectors.shape}", 500
            )
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise VectorStoreError("EMBEDDING_FAILURE", "BGE-M3 service returned non-normalized embeddings", 500)
        reported_latency = data.get("latency_ms")
        self.last_latency_ms = float(reported_latency) if isinstance(reported_latency, (int, float)) else (time.perf_counter() - started) * 1000
        return np.ascontiguousarray(vectors, dtype=np.float32)
