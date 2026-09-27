from __future__ import annotations

import os

import pytest

from vector_store.config import Settings
from vector_store.embeddings import BgeEmbeddingClient
from vector_store.reranking import BgeRerankerClient, RerankedRetriever
from vector_store.retriever import SemanticRetriever


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_RERANKER_TESTS") != "1",
    reason="set RUN_REAL_RERANKER_TESTS=1 after starting real BGE-M3 and BGE reranker services",
)


@pytest.fixture(scope="module")
def real_service():
    settings = Settings.from_env()
    base = SemanticRetriever(
        settings.store_dir,
        BgeEmbeddingClient(settings.embedding_url, settings.request_timeout_seconds),
        settings,
    )
    reranker = BgeRerankerClient(settings.reranker_url, settings.request_timeout_seconds)
    return RerankedRetriever(base, reranker, settings.initial_top_k, settings.final_top_k)


@pytest.mark.parametrize(
    "query",
    [
        "What are the symptoms of stem borer in rice?",
        "What are the management practices for brown plant hopper?",
        "What diseases affect rice crops?",
        "What pesticide dosage information is given for rice pests?",
        "What are the symptoms and management of rice blast?",
    ],
)
def test_real_paddy_faiss_candidates_are_reranked_with_metadata(real_service, query):
    response = real_service.retrieve(query, rerank=True, include_candidates=True)
    candidates = response["faiss_candidates"]
    results = response["results"]
    valid_ids = {record["chunk_id"] for record in real_service.retriever.store.metadata}
    assert len(candidates) == 10
    assert response["reranker_used"] is True
    assert len(results) <= 5
    assert all(row["chunk_id"] in valid_ids for row in results)
    assert all(row["score"] == row["faiss_score"] and isinstance(row["reranker_score"], float) for row in results)
    assert [row["reranker_score"] for row in results] == sorted(
        (row["reranker_score"] for row in results), reverse=True
    )
    assert all(row["source_filename"] == "paddy.pdf" and row["page_number"] > 0 and row["section"] for row in results)
    assert response["timings_ms"]["query_embedding_ms"] >= 0
    assert response["timings_ms"]["faiss_search_ms"] >= 0
    assert response["timings_ms"]["reranker_ms"] >= 0
