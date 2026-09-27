from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vector_store.errors import VectorStoreError
from vector_store.faiss_store import FaissVectorStore, save_store, validate_vectors
from vector_store.retriever import SemanticRetriever
from vector_store.schemas import read_knowledge_chunks


def row(chunk_id: str, page: int, section: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "source_filename": "paddy.pdf",
        "source_relative_path": "paddy.pdf",
        "page_number": page,
        "section": section,
        "extraction_method": "text",
        "ocr_used": False,
        "classification": "KNOWLEDGE",
        "text": text,
    }


def unit_vectors(count: int) -> np.ndarray:
    vectors = np.zeros((count, 1024), dtype=np.float32)
    for index in range(count):
        vectors[index, index] = 1.0
    return vectors


def store_fixture(tmp_path: Path, count: int = 3) -> tuple[Path, list[dict[str, object]]]:
    metadata = [row(f"chunk-{n}", n, f"Section {n}", f"Rice agriculture content {n}") for n in range(1, count + 1)]
    path = tmp_path / "store"
    save_store(path, unit_vectors(count), metadata, {"source_sha256": "test-source"})
    return path, metadata


def test_embedding_dimensions_and_normalization_are_validated():
    assert validate_vectors(unit_vectors(1), 1).shape == (1, 1024)
    with pytest.raises(VectorStoreError) as dim_error:
        validate_vectors(np.zeros((1, 4)), 1)
    assert dim_error.value.code == "INVALID_EMBEDDING_DIMENSION"
    with pytest.raises(VectorStoreError) as norm_error:
        validate_vectors(np.zeros((1, 1024)), 1)
    assert norm_error.value.code == "INVALID_EMBEDDING_DIMENSION"


def test_persists_index_metadata_and_alignment_in_source_order(tmp_path):
    path, expected = store_fixture(tmp_path)
    loaded = FaissVectorStore.load(path)
    assert loaded.index.ntotal == len(expected) == 3
    assert loaded.index.d == 1024
    assert type(loaded.index).__name__ == "IndexFlatIP"
    assert [item["chunk_id"] for item in loaded.metadata] == [item["chunk_id"] for item in expected]
    assert (path / "index.faiss").is_file()
    assert (path / "metadata.json").is_file()
    assert (path / "store_manifest.json").is_file()


def test_duplicate_chunk_ids_are_rejected(tmp_path):
    metadata = [row("same", 1, "A", "one"), row("same", 2, "B", "two")]
    with pytest.raises(VectorStoreError, match="duplicate|Duplicate") as error:
        save_store(tmp_path / "store", unit_vectors(2), metadata, {})
    assert error.value.code == "INVALID_METADATA"


def test_index_metadata_mismatch_is_a_structured_error(tmp_path):
    path, _ = store_fixture(tmp_path, count=3)
    data = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    data["records"].pop()
    (path / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(VectorStoreError) as error:
        FaissVectorStore.load(path)
    assert error.value.code == "VECTOR_STORE_MISMATCH"


def test_invalid_json_and_missing_store_are_structured(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(VectorStoreError) as invalid:
        read_knowledge_chunks(bad)
    assert invalid.value.code == "INVALID_METADATA"
    with pytest.raises(VectorStoreError) as missing:
        FaissVectorStore.load(tmp_path / "missing")
    assert missing.value.code == "VECTOR_STORE_NOT_FOUND"


def test_build_reader_filters_knowledge_and_maps_phase5_path_field(tmp_path):
    source = tmp_path / "chunks.jsonl"
    knowledge = {
        "chunk_id": "a", "document_id": "doc", "source_filename": "paddy.pdf",
        "relative_source_path": "paddy.pdf", "page_number": 1, "section": "Stem Borer",
        "extraction_method": "text", "ocr_used": False, "classification": "KNOWLEDGE", "text": "Symptoms",
    }
    navigation = {**knowledge, "chunk_id": "b", "classification": "NAVIGATION"}
    source.write_text("\n".join(json.dumps(item) for item in [knowledge, navigation]) + "\n", encoding="utf-8")
    records = read_knowledge_chunks(source)
    assert [item["chunk_id"] for item in records] == ["a"]
    assert records[0]["source_relative_path"] == "paddy.pdf"


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return unit_vectors(len(texts))


def test_retrieval_returns_ranked_scores_text_and_source_metadata(tmp_path):
    path, metadata = store_fixture(tmp_path)
    embedder = FakeEmbedder()
    retriever = SemanticRetriever(path, embedder)
    result = retriever.retrieve("rice query", top_k=2)
    assert result["query"] == "rice query"
    assert result["count"] == 2
    assert len(embedder.calls) == 1
    scores = [item["score"] for item in result["results"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    first = result["results"][0]
    assert first["chunk_id"] in {item["chunk_id"] for item in metadata}
    assert first["source_filename"] == "paddy.pdf"
    assert first["source_relative_path"] == "paddy.pdf"
    assert first["page_number"] in {1, 2, 3}
    assert first["section"]
    assert first["text"]


def test_retrieval_caps_top_k_to_vector_count_and_rejects_empty_query(tmp_path):
    path, _ = store_fixture(tmp_path, count=2)
    retriever = SemanticRetriever(path, FakeEmbedder())
    assert retriever.retrieve("rice", top_k=10)["count"] == 2
    with pytest.raises(VectorStoreError) as error:
        retriever.retrieve("   ")
    assert error.value.code == "EMPTY_QUERY"


def test_build_from_jsonl_keeps_ids_and_metadata_order_deterministic(tmp_path):
    from vector_store.build import build

    source = tmp_path / "chunks.jsonl"
    raw_rows = []
    for n, section in enumerate(["BPH", "Stem Borer"], start=1):
        raw_rows.append({
            "chunk_id": f"chunk-{n}", "document_id": "doc-1", "source_filename": "paddy.pdf",
            "relative_source_path": "paddy.pdf", "page_number": n, "section": section,
            "extraction_method": "text", "ocr_used": False, "classification": "KNOWLEDGE", "text": f"content {n}",
        })
    source.write_text("\n".join(json.dumps(item) for item in raw_rows) + "\n", encoding="utf-8")
    first_path = tmp_path / "store-one"
    second_path = tmp_path / "store-two"
    first_embedder = FakeEmbedder()
    second_embedder = FakeEmbedder()
    build(source, first_path, first_embedder)
    build(source, second_path, second_embedder)
    first = FaissVectorStore.load(first_path)
    second = FaissVectorStore.load(second_path)
    assert [item["chunk_id"] for item in first.metadata] == ["chunk-1", "chunk-2"]
    assert first.metadata == second.metadata
    assert first.index.ntotal == second.index.ntotal == 2
    assert first_embedder.calls == second_embedder.calls == [["content 1", "content 2"]]
    assert (first_path / "index.faiss").read_bytes() == (second_path / "index.faiss").read_bytes()
