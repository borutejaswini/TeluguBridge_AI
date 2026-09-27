from __future__ import annotations

import os

import pytest

from vector_store.config import Settings
from vector_store.embeddings import BgeEmbeddingClient
from vector_store.retriever import SemanticRetriever


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_BGE_TESTS") != "1",
    reason="set RUN_REAL_BGE_TESTS=1 with the real BGE-M3 service running to execute these integration checks",
)


@pytest.fixture(scope="module")
def real_retriever():
    settings = Settings.from_env()
    return SemanticRetriever(
        settings.store_dir,
        BgeEmbeddingClient(settings.embedding_url, settings.request_timeout_seconds),
        settings,
    )


@pytest.mark.parametrize(
    ("query", "relevant_section"),
    [
        ("What are the symptoms of stem borer in rice?", "Stem Borer"),
        ("What are the management practices for brown plant hopper?", "Brown Plant Hopper (BPH)"),
        ("What pesticide dosage information is given for rice pests?", "Pesticide"),
        ("What are the diseases affecting rice?", "Rice Blast"),
    ],
)
def test_real_paddy_retrieval_returns_relevant_source_chunks(real_retriever, query, relevant_section):
    retrieved = real_retriever.retrieve(query, top_k=10)
    results = retrieved["results"]
    known_ids = {row["chunk_id"] for row in real_retriever.store.metadata}
    assert len(results) == 10
    assert all(result["chunk_id"] in known_ids for result in results)
    assert any(relevant_section.casefold() in (result["section"] or "").casefold() for result in results)
    assert all(isinstance(result["score"], float) for result in results)
    assert all(isinstance(result["page_number"], int) and result["page_number"] > 0 for result in results)
    assert all(result["section"] and result["source_filename"] == "paddy.pdf" for result in results)

