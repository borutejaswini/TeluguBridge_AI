from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from vector_store.config import DEFAULT_SOURCE, DEFAULT_STORE, PROJECT_ROOT, Settings
from vector_store.embeddings import BgeEmbeddingClient
from vector_store.errors import VectorStoreError
from vector_store.faiss_store import DIMENSION, save_store
from vector_store.schemas import read_knowledge_chunks


def build(source: Path, store_dir: Path, embedder: BgeEmbeddingClient) -> dict[str, object]:
    chunks = read_knowledge_chunks(source)
    texts = [str(chunk["text"]) for chunk in chunks]
    vectors = embedder.embed(texts)
    try:
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise VectorStoreError("INVALID_METADATA", f"Could not checksum source JSONL: {exc}", 422) from exc
    source_document_ids = sorted({str(chunk["document_id"]) for chunk in chunks})
    return save_store(
        store_dir,
        vectors,
        chunks,
        {
            "source_jsonl": source.resolve().relative_to(PROJECT_ROOT).as_posix()
            if source.resolve().is_relative_to(PROJECT_ROOT)
            else str(source.resolve()),
            "source_sha256": source_sha256,
            "source_document_ids": source_document_ids,
            "build_configuration": {
                "embedding_batch_request_size": min(32, len(chunks)),
                "embedding_service_url": getattr(embedder, "base_url", "external embedding provider"),
                "vector_order": "unchanged source JSONL order after KNOWLEDGE filter",
            },
        },
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Build a persistent FAISS index from Phase 5 agriculture chunks")
    parser.add_argument("--source", type=Path, default=settings.source_jsonl)
    parser.add_argument("--output-dir", type=Path, default=settings.store_dir)
    parser.add_argument("--embedding-url", default=settings.embedding_url)
    args = parser.parse_args()
    embedder = BgeEmbeddingClient(args.embedding_url, settings.request_timeout_seconds)
    try:
        manifest = build(args.source.resolve(), args.output_dir.resolve(), embedder)
    except VectorStoreError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": "built", "store_dir": str(args.output_dir.resolve()), **manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
