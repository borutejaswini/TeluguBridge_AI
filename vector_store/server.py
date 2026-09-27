"""Minimal stdlib HTTP server for FAISS retrieval; keeps FAISS in its isolated venv."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vector_store.config import Settings
from vector_store.embeddings import BgeEmbeddingClient
from vector_store.errors import VectorStoreError
from vector_store.reranking import BgeRerankerClient, RerankedRetriever
from vector_store.retriever import SemanticRetriever

settings = Settings.from_env()
active_retriever: RerankedRetriever | None = None


class RetrievalHandler(BaseHTTPRequestHandler):
    server_version = "TeluguBridgeRetrieval/1"

    def _write(self, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write(404, {"error": {"code": "NOT_FOUND", "message": "Route not found"}})
            return
        self._write(200, {
            "status": "ok" if active_retriever is not None else "not_ready",
            "service": "faiss-retrieval",
            "loaded": active_retriever is not None,
            "index": str(settings.store_dir.relative_to(PROJECT_ROOT)) if settings.store_dir.is_relative_to(PROJECT_ROOT) else "configured",
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "initial_top_k": settings.initial_top_k,
            "final_top_k": settings.final_top_k,
            "reranker_model": settings.reranker_model if settings.reranker_enabled else None,
        })

    def do_POST(self) -> None:
        if self.path != "/retrieve":
            self._write(404, {"error": {"code": "NOT_FOUND", "message": "Route not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 2_000_000:
                raise ValueError("request body must be between 1 byte and 2 MiB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            query = payload.get("query") if isinstance(payload, dict) else None
            if not isinstance(query, str) or not query.strip():
                self._write(422, {"error": {"code": "EMPTY_QUERY", "message": "query must be a non-empty string"}})
                return
            if active_retriever is None:
                self._write(503, {"error": {"code": "RETRIEVAL_NOT_READY", "message": "FAISS retrieval has not loaded"}})
                return
            result = active_retriever.retrieve(
                query,
                rerank=settings.reranker_enabled,
                initial_top_k=settings.initial_top_k,
                final_top_k=settings.final_top_k,
                include_candidates=bool(payload.get("include_candidates", True)),
            )
            self._write(200, result)
        except VectorStoreError as exc:
            self._write(exc.status_code, {"error": {"code": exc.code, "message": exc.message}})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._write(400, {"error": {"code": "INVALID_REQUEST", "message": str(exc)}})
        except Exception as exc:
            self._write(500, {"error": {"code": "RETRIEVAL_FAILURE", "message": f"Retrieval stage failed: {type(exc).__name__}: {exc}"}})

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main() -> None:
    global active_retriever
    embedder = BgeEmbeddingClient(settings.embedding_url, settings.request_timeout_seconds)
    semantic = SemanticRetriever(settings.store_dir, embedder, settings)
    reranker = BgeRerankerClient(settings.reranker_url, settings.request_timeout_seconds)
    active_retriever = RerankedRetriever(semantic, reranker, settings.initial_top_k, settings.final_top_k)
    server = ThreadingHTTPServer(("127.0.0.1", 8004), RetrievalHandler)
    print("FAISS retrieval loaded; listening on 127.0.0.1:8004", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
