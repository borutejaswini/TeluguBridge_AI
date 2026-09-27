from __future__ import annotations

import numpy as np
import pytest

from vector_store.config import Settings
from vector_store.errors import VectorStoreError
from vector_store.reranking import RerankedRetriever


def candidate(chunk_id: str, score: float, page: int) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "source_filename": "paddy.pdf",
        "source_relative_path": "paddy.pdf",
        "page_number": page,
        "section": f"Section {page}",
        "extraction_method": "text",
        "classification": "KNOWLEDGE",
        "text": f"agriculture chunk {chunk_id}",
        "score": score,
    }


class FakeRetriever:
    def __init__(self):
        self.called = []
        self.candidates = [candidate(f"chunk-{i}", 1.0 - i / 20, i) for i in range(1, 11)]

    def retrieve(self, query, top_k):
        self.called.append((query, top_k))
        return {"query": query, "results": self.candidates[:top_k], "count": min(top_k, 10),
                "timings_ms": {"query_embedding": 4.0, "faiss_search": 0.2}}


class FakeReranker:
    last_latency_ms = 3.0

    def __init__(self, scores=None, error=None):
        self.calls = []
        self.scores = scores if scores is not None else [0.1 * n for n in range(10)]
        self.error = error

    def score(self, query, texts):
        self.calls.append((query, texts))
        if self.error:
            raise self.error
        return {"scores": self.scores, "latency_ms": self.last_latency_ms, "model": "BAAI/bge-reranker-v2-m3", "device": "cpu"}


def test_reranker_configuration_names_and_defaults():
    assert Settings().initial_top_k == 10
    assert Settings().final_top_k == 5
    assert Settings().reranker_model == "BAAI/bge-reranker-v2-m3"
    assert Settings().reranker_device == "cpu"


def test_rerank_sorts_scores_descending_and_preserves_faiss_score_and_metadata():
    retriever, reranker = FakeRetriever(), FakeReranker()
    result = RerankedRetriever(retriever, reranker).retrieve("rice symptoms", initial_top_k=10, final_top_k=5)
    assert result["reranker_used"] is True
    assert result["initial_top_k"] == 10
    assert result["final_top_k"] == 5
    assert result["count"] == 5
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 10
    values = result["results"]
    assert [row["chunk_id"] for row in values] == [f"chunk-{i}" for i in range(10, 5, -1)]
    assert [row["reranker_score"] for row in values] == sorted((row["reranker_score"] for row in values), reverse=True)
    for row in values:
        assert row["score"] == row["faiss_score"]
        assert row["source_filename"] == "paddy.pdf"
        assert row["source_relative_path"] == "paddy.pdf"
        assert row["page_number"] > 0 and row["section"] and row["text"]
        assert row["extraction_method"] == "text" and row["classification"] == "KNOWLEDGE"
    assert result["timings_ms"] == {"query_embedding_ms": 4.0, "faiss_search_ms": 0.2, "reranker_ms": 3.0, "total_ms": 7.2}


def test_disabled_reranker_returns_faiss_top_final_and_never_calls_model():
    retriever, reranker = FakeRetriever(), FakeReranker(error=AssertionError("must not be called"))
    result = RerankedRetriever(retriever, reranker).retrieve("rice", rerank=False)
    assert result["reranker_used"] is False
    assert [row["chunk_id"] for row in result["results"]] == [f"chunk-{i}" for i in range(1, 6)]
    assert all(row["reranker_score"] is None for row in result["results"])
    assert reranker.calls == []


def test_enabled_reranker_failure_is_not_silently_downgraded():
    retriever = FakeRetriever()
    reranker = FakeReranker(error=VectorStoreError("RERANKER_INFERENCE_FAILED", "unavailable", 503))
    with pytest.raises(VectorStoreError) as error:
        RerankedRetriever(retriever, reranker).retrieve("rice", rerank=True)
    assert error.value.code == "RERANKER_INFERENCE_FAILED"


def test_wrong_number_or_invalid_scores_are_structured():
    retriever = FakeRetriever()
    for scores in ([0.5], [0.2, float("nan")] + [0.0] * 8):
        reranker = FakeReranker(scores=scores)
        with pytest.raises(VectorStoreError) as error:
            RerankedRetriever(retriever, reranker).retrieve("rice")
        assert error.value.code == "INVALID_RERANKER_RESULT"


def test_empty_query_and_top_k_are_rejected():
    service = RerankedRetriever(FakeRetriever(), FakeReranker())
    for kwargs in ({"query": " "}, {"query": "rice", "initial_top_k": 0}, {"query": "rice", "final_top_k": 0}):
        query = kwargs.pop("query")
        with pytest.raises(VectorStoreError):
            service.retrieve(query, **kwargs)

