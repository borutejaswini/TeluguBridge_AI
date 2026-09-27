from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vector_store.errors import VectorStoreError


REQUIRED_METADATA_FIELDS = (
    "chunk_id",
    "document_id",
    "source_filename",
    "source_relative_path",
    "page_number",
    "section",
    "extraction_method",
    "ocr_used",
    "classification",
    "text",
)


def normalize_chunk(raw: Any, line_number: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise VectorStoreError("INVALID_METADATA", f"JSONL row {line_number} must be an object", 422)
    relative_path = raw.get("source_relative_path", raw.get("relative_source_path"))
    row = {
        "chunk_id": raw.get("chunk_id"),
        "document_id": raw.get("document_id"),
        "source_filename": raw.get("source_filename"),
        "source_relative_path": relative_path,
        "page_number": raw.get("page_number"),
        "section": raw.get("section"),
        "extraction_method": raw.get("extraction_method"),
        "ocr_used": raw.get("ocr_used"),
        "classification": raw.get("classification"),
        "text": raw.get("text"),
    }
    for key in ("chunk_id", "document_id", "source_filename", "source_relative_path", "extraction_method", "classification", "text"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise VectorStoreError("INVALID_METADATA", f"JSONL row {line_number} has invalid {key}", 422)
    if not isinstance(row["page_number"], int) or isinstance(row["page_number"], bool) or row["page_number"] < 1:
        raise VectorStoreError("INVALID_METADATA", f"JSONL row {line_number} has invalid page_number", 422)
    if row["section"] is not None and not isinstance(row["section"], str):
        raise VectorStoreError("INVALID_METADATA", f"JSONL row {line_number} has invalid section", 422)
    if not isinstance(row["ocr_used"], bool):
        raise VectorStoreError("INVALID_METADATA", f"JSONL row {line_number} has invalid ocr_used", 422)
    return row


def read_knowledge_chunks(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise VectorStoreError("VECTOR_STORE_NOT_FOUND", f"Phase 5 JSONL not found: {path}", 404)
    chunks: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VectorStoreError(
                        "INVALID_METADATA", f"Invalid JSON on JSONL row {line_number}: {exc.msg}", 422
                    ) from exc
                row = normalize_chunk(raw, line_number)
                if row["classification"] == "KNOWLEDGE":
                    chunks.append(row)
    except OSError as exc:
        raise VectorStoreError("INVALID_METADATA", f"Could not read Phase 5 JSONL: {exc}", 422) from exc
    if not chunks:
        raise VectorStoreError("INVALID_METADATA", "Phase 5 JSONL contains no KNOWLEDGE chunks", 422)
    ids = [str(row["chunk_id"]) for row in chunks]
    if len(ids) != len(set(ids)):
        raise VectorStoreError("INVALID_METADATA", "Duplicate chunk_id values in Phase 5 JSONL", 422)
    return chunks


def validate_metadata(records: Any) -> list[dict[str, object]]:
    if not isinstance(records, list) or not records:
        raise VectorStoreError("INVALID_METADATA", "Metadata records must be a non-empty JSON array", 500)
    validated = [normalize_chunk(row, index + 1) for index, row in enumerate(records)]
    ids = [str(row["chunk_id"]) for row in validated]
    if len(ids) != len(set(ids)):
        raise VectorStoreError("INVALID_METADATA", "Metadata contains duplicate chunk_id values", 500)
    return validated

