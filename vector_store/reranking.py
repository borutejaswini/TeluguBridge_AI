from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from vector_store.errors import VectorStoreError


class BgeRerankerClient:
    """HTTP client for the process-long CPU BGE CrossEncoder service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8003", timeout_seconds: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.last_latency_ms: float | None = None
        self.model_load_seconds: float | None = None

    def _get_health(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/health", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 10.0)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError("RERANKER_MODEL_LOAD_FAILED", f"Reranker service is unavailable: {exc}", 503) from exc
        if not isinstance(data, dict) or not data.get("loaded"):
            raise VectorStoreError("RERANKER_MODEL_LOAD_FAILED", "Reranker service has not loaded its model", 503)
        if data.get("model") != "BAAI/bge-reranker-v2-m3" or data.get("device") != "cpu":
            raise VectorStoreError("RERANKER_MODEL_LOAD_FAILED", "Reranker service model/device does not match configuration", 503)
        self.model_load_seconds = float(data["model_load_seconds"]) if data.get("model_load_seconds") is not None else None
        return data

    def score(self, query: str, candidate_texts: list[str]) -> dict[str, object]:
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreError("EMPTY_QUERY", "query must be non-empty", 422)
        if not candidate_texts or len(candidate_texts) > 32 or any(not isinstance(text, str) or not text.strip() for text in candidate_texts):
            raise VectorStoreError("INVALID_RERANKER_RESULT", "reranker candidates must contain 1 to 32 non-empty texts", 422)
        self._get_health()
        payload = json.dumps({"query": query.strip(), "candidates": candidate_texts}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/rerank",
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
                body = json.loads(exc.read().decode("utf-8", errors="replace"))
                error = body.get("error", {}) if isinstance(body, dict) else {}
                code = error.get("code")
                message = error.get("message")
            except (json.JSONDecodeError, UnicodeDecodeError):
                code, message = None, None
            if code not in {"RERANKER_MODEL_LOAD_FAILED", "RERANKER_INFERENCE_FAILED", "INVALID_RERANKER_RESULT"}:
                code = "RERANKER_INFERENCE_FAILED"
            raise VectorStoreError(code, message or f"Reranker returned HTTP {exc.code}", 503 if exc.code >= 500 else 422) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError("RERANKER_INFERENCE_FAILED", f"Reranker request failed: {exc}", 503) from exc
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        if not isinstance(data, dict) or data.get("model") != "BAAI/bge-reranker-v2-m3" or data.get("device") != "cpu":
            raise VectorStoreError("INVALID_RERANKER_RESULT", "Reranker returned unexpected model/device metadata", 500)
        try:
            scores = np.asarray(data["scores"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError("INVALID_RERANKER_RESULT", f"Reranker returned invalid scores: {exc}", 500) from exc
        if scores.shape != (len(candidate_texts),) or not np.isfinite(scores).all():
            raise VectorStoreError(
                "INVALID_RERANKER_RESULT",
                f"Expected {len(candidate_texts)} finite reranker scores, got shape {scores.shape}",
                500,
            )
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise VectorStoreError("INVALID_RERANKER_RESULT", "Sigmoid reranker scores must be in [0, 1]", 500)
        latency = data.get("latency_ms")
        return {
            "scores": [float(value) for value in scores],
            "latency_ms": float(latency) if isinstance(latency, (int, float)) else self.last_latency_ms,
            "model": data["model"],
            "device": data["device"],
        }


class RerankedRetriever:
    def __init__(self, retriever: Any, reranker: Any, initial_top_k: int = 10, final_top_k: int = 5) -> None:
        if initial_top_k < 1 or final_top_k < 1:
            raise ValueError("initial_top_k and final_top_k must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        *,
        rerank: bool = True,
        initial_top_k: int | None = None,
        final_top_k: int | None = None,
        include_candidates: bool = False,
    ) -> dict[str, object]:
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreError("EMPTY_QUERY", "query must be non-empty", 422)
        initial_k = self.initial_top_k if initial_top_k is None else initial_top_k
        final_k = self.final_top_k if final_top_k is None else final_top_k
        if not isinstance(initial_k, int) or isinstance(initial_k, bool) or initial_k < 1:
            raise VectorStoreError("INVALID_TOP_K", "initial_top_k must be positive", 422)
        if not isinstance(final_k, int) or isinstance(final_k, bool) or final_k < 1:
            raise VectorStoreError("INVALID_TOP_K", "final_top_k must be positive", 422)
        faiss_result = self.retriever.retrieve(query.strip(), initial_k)
        candidates = faiss_result["results"]
        timings = {
            "query_embedding_ms": float(faiss_result.get("timings_ms", {}).get("query_embedding", 0.0)),
            "faiss_search_ms": float(faiss_result.get("timings_ms", {}).get("faiss_search", 0.0)),
            "reranker_ms": 0.0,
        }
        if rerank:
            reranker_result = self.reranker.score(query.strip(), [str(item["text"]) for item in candidates])
            scores = reranker_result.get("scores")
            if not isinstance(scores, list) or len(scores) != len(candidates):
                raise VectorStoreError("INVALID_RERANKER_RESULT", "Reranker must return one score per FAISS candidate", 500)
            parsed_scores: list[float] = []
            for score in scores:
                try:
                    value = float(score)
                except (TypeError, ValueError) as exc:
                    raise VectorStoreError("INVALID_RERANKER_RESULT", "Reranker scores must be numeric", 500) from exc
                if not math.isfinite(value):
                    raise VectorStoreError("INVALID_RERANKER_RESULT", "Reranker scores must be finite", 500)
                parsed_scores.append(value)
            ranked = [({**candidate, "faiss_score": candidate["score"], "reranker_score": score}, position)
                      for position, (candidate, score) in enumerate(zip(candidates, parsed_scores))]
            ranked.sort(key=lambda pair: (-pair[0]["reranker_score"], pair[1]))
            final_results = [item for item, _ in ranked[:final_k]]
            timings["reranker_ms"] = float(reranker_result.get("latency_ms", getattr(self.reranker, "last_latency_ms", 0.0)) or 0.0)
        else:
            final_results = [
                {**candidate, "faiss_score": candidate["score"], "reranker_score": None}
                for candidate in candidates[:final_k]
            ]
        timings["total_ms"] = sum(timings.values())
        response: dict[str, object] = {
            "query": query.strip(),
            "initial_top_k": initial_k,
            "final_top_k": final_k,
            "candidate_count": len(candidates),
            "count": len(final_results),
            "reranker_used": rerank,
            "results": final_results,
            "timings_ms": timings,
        }
        if include_candidates:
            response["faiss_candidates"] = candidates
        return response

