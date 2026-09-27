from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vector_store.config import Settings
from vector_store.embeddings import BgeEmbeddingClient
from vector_store.errors import VectorStoreError
from vector_store.retriever import SemanticRetriever


QUERIES = [
    "What are the symptoms of stem borer in rice?",
    "What are the management practices for brown plant hopper?",
    "What pesticide dosage information is given for rice pests?",
    "What are the diseases affecting rice?",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Run real English semantic retrieval against the paddy FAISS store")
    parser.add_argument("--store-dir", type=Path, default=settings.store_dir)
    parser.add_argument("--embedding-url", default=settings.embedding_url)
    parser.add_argument("--top-k", type=int, default=settings.initial_top_k)
    parser.add_argument("--display-k", type=int, default=10, help="number of top results to include in printed report")
    args = parser.parse_args()
    if args.top_k < 1 or args.display_k < 1:
        parser.error("--top-k and --display-k must be positive")
    embedder = BgeEmbeddingClient(args.embedding_url, settings.request_timeout_seconds)
    try:
        retriever = SemanticRetriever(args.store_dir.resolve(), embedder, settings)
        results = [retriever.retrieve(query, args.top_k) for query in QUERIES]
    except VectorStoreError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    report = [
        {
            "query": result["query"],
            "retrieval_result_count": result["count"],
            "displayed_results": result["results"][: args.display_k],
        }
        for result in results
    ]
    print(json.dumps({"model": "BAAI/bge-m3", "device": "cpu", "index_type": "IndexFlatIP", "queries": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
