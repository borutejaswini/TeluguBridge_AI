from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vector_store.config import Settings
from vector_store.embeddings import BgeEmbeddingClient
from vector_store.errors import VectorStoreError
from vector_store.reranking import BgeRerankerClient, RerankedRetriever
from vector_store.retriever import SemanticRetriever


QUERIES = [
    "What are the symptoms of stem borer in rice?",
    "What are the management practices for brown plant hopper?",
    "What diseases affect rice crops?",
    "What pesticide dosage information is given for rice pests?",
    "What are the symptoms and management of rice blast?",
]


def _brief(row: dict[str, object], rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": row["chunk_id"],
        "document_id": row["document_id"],
        "faiss_score": row.get("faiss_score", row["score"]),
        "reranker_score": row.get("reranker_score"),
        "source_filename": row["source_filename"],
        "source_relative_path": row["source_relative_path"],
        "page_number": row["page_number"],
        "section": row["section"],
        "extraction_method": row["extraction_method"],
        "classification": row["classification"],
        "text_excerpt": str(row["text"])[:200],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Compare FAISS top-5 with BGE-reranked top-5 on real paddy.pdf vectors")
    parser.add_argument("--store-dir", type=Path, default=settings.store_dir)
    parser.add_argument("--embedding-url", default=settings.embedding_url)
    parser.add_argument("--reranker-url", default=settings.reranker_url)
    args = parser.parse_args()
    retriever = SemanticRetriever(
        args.store_dir.resolve(),
        BgeEmbeddingClient(args.embedding_url, settings.request_timeout_seconds),
        settings,
    )
    combined = RerankedRetriever(
        retriever,
        BgeRerankerClient(args.reranker_url, settings.request_timeout_seconds),
        settings.initial_top_k,
        settings.final_top_k,
    )
    reports = []
    try:
        for query in QUERIES:
            result = combined.retrieve(query, rerank=True, include_candidates=True)
            reports.append({
                "query": query,
                "faiss_top10": [_brief(row, rank) for rank, row in enumerate(result["faiss_candidates"], start=1)],
                "faiss_only_top5": [
                    _brief(row, rank)
                    for rank, row in enumerate(result["faiss_candidates"][: settings.final_top_k], start=1)
                ],
                "reranked_top5": [_brief(row, rank) for rank, row in enumerate(result["results"], start=1)],
                "candidate_count": result["candidate_count"],
                "timings_ms": result["timings_ms"],
            })
    except VectorStoreError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({
        "reranker_model": settings.reranker_model,
        "device": settings.reranker_device,
        "initial_top_k": settings.initial_top_k,
        "final_top_k": settings.final_top_k,
        "queries": reports,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
