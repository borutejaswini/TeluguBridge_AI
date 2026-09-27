from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from vector_store.errors import VectorStoreError
from vector_store.schemas import validate_metadata


DIMENSION = 1024
METADATA_SCHEMA_VERSION = 1


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temp_path = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temp_path).replace(path)
    except OSError as exc:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise VectorStoreError("FAISS_SEARCH_FAILURE", f"Could not persist {path.name}: {exc}", 500) from exc


def _metadata_payload(records: list[dict[str, object]]) -> bytes:
    body = {"schema_version": METADATA_SCHEMA_VERSION, "records": records}
    return (json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _index_bytes(index: Any) -> bytes:
    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as stream:
            path = stream.name
        faiss.write_index(index, path)
        return Path(path).read_bytes()
    except Exception as exc:
        raise VectorStoreError("FAISS_SEARCH_FAILURE", f"Could not serialize FAISS index: {exc}", 500) from exc
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


def validate_vectors(vectors: Any, count: int, dimension: int = DIMENSION) -> np.ndarray:
    try:
        array = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise VectorStoreError("INVALID_EMBEDDING_DIMENSION", f"Embeddings are not a numeric matrix: {exc}", 422) from exc
    if array.shape != (count, dimension):
        raise VectorStoreError(
            "INVALID_EMBEDDING_DIMENSION", f"Expected embedding matrix ({count}, {dimension}), got {array.shape}", 422
        )
    if not np.isfinite(array).all():
        raise VectorStoreError("INVALID_EMBEDDING_DIMENSION", "Embeddings contain NaN or infinite values", 422)
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise VectorStoreError("INVALID_EMBEDDING_DIMENSION", "All stored embeddings must be normalized", 422)
    return np.ascontiguousarray(array, dtype=np.float32)


def build_index(vectors: Any, dimension: int = DIMENSION) -> Any:
    try:
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index
    except Exception as exc:
        raise VectorStoreError("FAISS_SEARCH_FAILURE", f"Could not construct IndexFlatIP: {exc}", 500) from exc


def save_store(
    store_dir: Path,
    vectors: Any,
    metadata: list[dict[str, object]],
    manifest_fields: dict[str, object],
) -> dict[str, object]:
    records = validate_metadata(metadata)
    if len(records) != len(metadata):
        raise VectorStoreError("VECTOR_STORE_MISMATCH", "Metadata count changed during validation", 500)
    ids = [str(row["chunk_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise VectorStoreError("INVALID_METADATA", "Duplicate chunk IDs cannot be indexed", 422)
    matrix = validate_vectors(vectors, len(records))
    index = build_index(matrix)
    if index.ntotal != len(records):
        raise VectorStoreError("VECTOR_STORE_MISMATCH", f"index.ntotal={index.ntotal}, metadata={len(records)}", 500)

    store_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = store_dir / "metadata.json"
    index_path = store_dir / "index.faiss"
    manifest_path = store_dir / "store_manifest.json"
    metadata_bytes = _metadata_payload(records)
    index_bytes = _index_bytes(index)
    manifest = {
        "schema_version": 1,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": DIMENSION,
        "normalized": True,
        "device": "cpu",
        "faiss_index_type": "IndexFlatIP",
        "similarity": "inner_product (cosine for normalized vectors)",
        "vector_count": int(index.ntotal),
        "metadata_count": len(records),
        "metadata_order": "source JSONL row order, knowledge rows only",
        "metadata_file": metadata_path.name,
        "index_file": index_path.name,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        **manifest_fields,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    # Write manifest last; its presence signals a complete build. Each file is
    # individually replaced atomically and loading validates the pair.
    _atomic_bytes(index_path, index_bytes)
    _atomic_bytes(metadata_path, metadata_bytes)
    _atomic_bytes(manifest_path, manifest_bytes)
    return manifest


class FaissVectorStore:
    def __init__(self, store_dir: Path, index: Any, metadata: list[dict[str, object]]) -> None:
        self.store_dir = store_dir
        self.index = index
        self.metadata = metadata

    @classmethod
    def load(cls, store_dir: Path, dimension: int = DIMENSION) -> "FaissVectorStore":
        index_path = store_dir / "index.faiss"
        metadata_path = store_dir / "metadata.json"
        manifest_path = store_dir / "store_manifest.json"
        missing = [path.name for path in (index_path, metadata_path, manifest_path) if not path.is_file()]
        if missing:
            raise VectorStoreError("VECTOR_STORE_NOT_FOUND", f"Vector store files missing in {store_dir}: {', '.join(missing)}", 404)
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError("INVALID_METADATA", f"Vector store metadata/manifest is invalid: {exc}", 500) from exc
        if not isinstance(raw_metadata, dict) or raw_metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
            raise VectorStoreError("INVALID_METADATA", "Unsupported metadata schema", 500)
        metadata = validate_metadata(raw_metadata.get("records"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise VectorStoreError("VECTOR_STORE_MISMATCH", "Store manifest is missing or has an unsupported schema", 500)
        if manifest.get("embedding_model") != "BAAI/bge-m3" or manifest.get("embedding_dimension") != dimension:
            raise VectorStoreError("INVALID_EMBEDDING_DIMENSION", "Store manifest embedding configuration is incompatible", 500)
        if manifest.get("normalized") is not True or manifest.get("faiss_index_type") != "IndexFlatIP":
            raise VectorStoreError("VECTOR_STORE_MISMATCH", "Store manifest does not describe normalized IndexFlatIP vectors", 500)
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError("VECTOR_STORE_MISMATCH", f"Could not load FAISS index: {exc}", 500) from exc
        if index.d != dimension:
            raise VectorStoreError("INVALID_EMBEDDING_DIMENSION", f"Index dimension {index.d} != {dimension}", 500)
        if index.ntotal != len(metadata):
            raise VectorStoreError("VECTOR_STORE_MISMATCH", f"index.ntotal={index.ntotal}, metadata={len(metadata)}", 500)
        if manifest.get("vector_count") != index.ntotal or manifest.get("metadata_count") != len(metadata):
            raise VectorStoreError("VECTOR_STORE_MISMATCH", "Store manifest counts do not match index and metadata", 500)
        if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise VectorStoreError("VECTOR_STORE_MISMATCH", "Expected an exact IndexFlatIP index", 500)
        expected_meta_hash = manifest.get("metadata_sha256") if isinstance(manifest, dict) else None
        actual_meta_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if expected_meta_hash != actual_meta_hash:
            raise VectorStoreError("VECTOR_STORE_MISMATCH", "Metadata checksum does not match the store manifest", 500)
        expected_index_hash = manifest.get("index_sha256")
        actual_index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
        if expected_index_hash != actual_index_hash:
            raise VectorStoreError("VECTOR_STORE_MISMATCH", "FAISS index checksum does not match the store manifest", 500)
        try:
            stored_vectors = index.reconstruct_n(0, index.ntotal)
        except Exception as exc:
            raise VectorStoreError("VECTOR_STORE_MISMATCH", f"Could not validate stored vectors: {exc}", 500) from exc
        validate_vectors(stored_vectors, len(metadata), dimension)
        return cls(store_dir, index, metadata)

    def search(self, query_vector: Any, top_k: int) -> list[tuple[float, dict[str, object]]]:
        if top_k < 1:
            raise VectorStoreError("EMPTY_QUERY", "top_k must be positive", 422)
        vector = validate_vectors(np.asarray(query_vector).reshape(1, -1), 1)
        if self.index.ntotal != len(self.metadata):
            raise VectorStoreError("VECTOR_STORE_MISMATCH", f"index.ntotal={self.index.ntotal}, metadata={len(self.metadata)}", 500)
        k = min(top_k, self.index.ntotal)
        try:
            scores, positions = self.index.search(vector, k)
        except Exception as exc:
            raise VectorStoreError("FAISS_SEARCH_FAILURE", f"FAISS search failed: {exc}", 500) from exc
        pairs = []
        for score, position in zip(scores[0].tolist(), positions[0].tolist()):
            if position < 0 or position >= len(self.metadata):
                raise VectorStoreError("VECTOR_STORE_MISMATCH", f"FAISS returned invalid metadata row {position}", 500)
            pairs.append((float(score), self.metadata[position]))
        pairs.sort(key=lambda item: (-item[0], self.metadata.index(item[1])))
        return pairs
