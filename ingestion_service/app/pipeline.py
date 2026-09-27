from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pymupdf as fitz

from app.chunking import chunk_page, make_chunk_id
from app.cleaning import clean_page_text, edge_lines, repeated_edge_lines
from app.config import PROJECT_ROOT, Settings, get_settings
from app.errors import IngestionError
from app.ocr import ocr_page

logger = logging.getLogger(__name__)
_NAVIGATION_HINTS = re.compile(r"\b(contents|table of contents|index|appendix|bibliography|references)\b", re.I)
_AGRICULTURE_HINTS = re.compile(
    r"\b(pest|disease|stem borer|bph|gall midge|rice hispa|leaf folder|rice blast|"
    r"sheath blight|bacterial leaf blight|management|symptom|insect|fung|crop|paddy|rice|"
    r"fertiliz|pesticide|insecticide|fungicide|dosage|application|etl)\b",
    re.I,
)


@dataclass(frozen=True)
class PageInfo:
    page_number: int
    extraction_method: str
    ocr_used: bool
    classification: str
    text_characters: int
    sections: list[str]


def _quality_sufficient(text: str, settings: Settings) -> bool:
    normalized = " ".join(text.split())
    if len(normalized) < settings.ocr_min_text_chars:
        return False
    alnum = sum(char.isalnum() for char in normalized)
    ratio = alnum / max(len(normalized), 1)
    words = re.findall(r"[\w%]+", normalized, flags=re.UNICODE)
    return ratio >= settings.ocr_min_alnum_ratio and len(words) >= 3


def classify_text(text: str) -> str:
    normalized = " ".join(text.split())
    alnum = sum(char.isalnum() for char in normalized)
    if alnum < 20:
        return "NOISE"
    if _NAVIGATION_HINTS.search(normalized) and not _AGRICULTURE_HINTS.search(normalized):
        words = normalized.split()
        numeric_or_dotted = sum(bool(re.fullmatch(r"[\d.IVXivx-]+", word.strip(".:;"))) for word in words)
        if numeric_or_dotted >= max(2, len(words) // 5):
            return "NAVIGATION"
    # Conservative default: substantive page text is knowledge unless it is
    # clearly navigation, so valid crop information is not discarded.
    return "KNOWLEDGE"


def _document_id(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise IngestionError("PDF_READ_ERROR", f"Could not read PDF bytes: {exc}", source=str(path)) from exc
    return digest.hexdigest()


def discover_pdfs(knowledge_root: Path) -> list[Path]:
    root = knowledge_root.resolve()
    if not root.is_dir():
        raise IngestionError("PDF_DISCOVERY_ERROR", f"Knowledge base directory does not exist: {root}")
    try:
        return sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
    except OSError as exc:
        raise IngestionError("PDF_DISCOVERY_ERROR", f"Could not scan {root}: {exc}") from exc


def _open_document(path: Path):
    try:
        document = fitz.open(path)
        if not document.is_pdf:
            document.close()
            raise IngestionError("INVALID_DOCUMENT", "File is not a valid PDF", source=str(path))
        return document
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Could not open PDF %s", path)
        raise IngestionError("INVALID_DOCUMENT", f"Could not open PDF: {exc}", source=str(path)) from exc


def _header_footer_signatures(path: Path) -> tuple[int, set[str]]:
    try:
        with _open_document(path) as document:
            count = len(document)
            edges = []
            for page in document:
                text = page.get_text("text", sort=True)
                edges.append(edge_lines(text))
        return count, repeated_edge_lines(edges, count)
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("PDF header/footer scan failed path=%s", path)
        raise IngestionError("PDF_EXTRACTION_ERROR", f"Could not inspect PDF pages: {exc}", source=str(path)) from exc


def _sections_in(text: str) -> list[str]:
    return [section for section, _ in chunk_page(text, target_chars=10**9) if section]


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            temp_name = stream.name
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        Path(temp_name).replace(path)
    except Exception as exc:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
        raise IngestionError("INGESTION_ERROR", f"Could not write intermediate JSONL: {exc}", source=str(path)) from exc


def _atomic_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
            temp_name = stream.name
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        Path(temp_name).replace(path)
    except Exception as exc:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
        raise IngestionError("INGESTION_ERROR", f"Could not write manifest: {exc}", source=str(path)) from exc


def ingest_pdf(
    path: Path,
    knowledge_root: Path,
    output_dir: Path,
    settings: Settings | None = None,
    *,
    ocr_runner: Callable[..., str] | None = None,
) -> dict[str, object]:
    settings = settings or get_settings()
    root = knowledge_root.resolve()
    source = path.resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise IngestionError("PDF_READ_ERROR", "PDF is outside the configured knowledge base", source=str(source)) from exc
    document_id = _document_id(source)
    page_count, repeated_lines = _header_footer_signatures(source)
    output_dir = output_dir.resolve()
    jsonl_path = output_dir / f"{document_id}.jsonl"
    manifest_path = output_dir / f"{document_id}.manifest.json"
    chunks: list[dict[str, object]] = []
    pages: list[dict[str, object]] = []
    total_chars = 0
    document_sections: set[str] = set()
    page_classes: list[str] = []
    run_ocr = ocr_runner or ocr_page
    continuation_section: str | None = None

    try:
        with _open_document(source) as document:
            if len(document) == 0:
                raise IngestionError("INVALID_DOCUMENT", "PDF contains no pages", source=relative)
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                try:
                    raw_text = page.get_text("text", sort=True)
                except Exception as exc:
                    logger.exception("Text extraction failed source=%s page=%d", relative, page_number)
                    raise IngestionError("PDF_EXTRACTION_ERROR", f"Page text extraction failed: {exc}", source=relative, page=page_number) from exc
                ocr_used = False
                method = "text"
                if not _quality_sufficient(raw_text, settings):
                    method = "ocr"
                    ocr_used = True
                    raw_text = run_ocr(page, settings, source=relative, page_number=page_number)
                text = clean_page_text(raw_text, repeated_lines)
                classification = classify_text(text)
                page_classes.append(classification)
                total_chars += len(text)
                page_chunks = chunk_page(
                    text,
                    settings.chunk_target_chars,
                    inherited_section=continuation_section,
                )
                page_sections: list[str] = []
                for chunk_index, (section, chunk_text) in enumerate(page_chunks, start=1):
                    if section and section not in page_sections:
                        page_sections.append(section)
                        document_sections.add(section)
                    chunks.append(
                        {
                            "document_id": document_id,
                            "source_filename": source.name,
                            "relative_source_path": relative,
                            "page_number": page_number,
                            "section": section,
                            "chunk_id": make_chunk_id(document_id, page_number, chunk_index, section, chunk_text),
                            "classification": classification,
                            "extraction_method": method,
                            "ocr_used": ocr_used,
                            "text": chunk_text,
                        }
                    )
                pages.append(asdict(PageInfo(page_number, method, ocr_used, classification, len(text), page_sections)))
                last_section = next((section for section, _ in reversed(page_chunks) if section), None)
                if last_section:
                    continuation_section = last_section
                logger.info(
                    "Ingested page source=%s page=%d method=%s ocr=%s chars=%d chunks=%d classification=%s",
                    relative, page_number, method, ocr_used, len(text), len(page_chunks), classification,
                )
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("PDF ingestion failed path=%s", source)
        raise IngestionError("PDF_READ_ERROR", f"Could not process PDF: {exc}", source=relative) from exc

    document_class = "KNOWLEDGE" if "KNOWLEDGE" in page_classes else (
        "NAVIGATION" if "NAVIGATION" in page_classes else "NOISE"
    )
    try:
        _atomic_jsonl(jsonl_path, chunks)
        manifest = {
            "document_id": document_id,
            "source_filename": source.name,
            "relative_source_path": relative,
            "page_count": page_count,
            "document_classification": document_class,
            "total_extracted_text_characters": total_chars,
            "chunk_count": len(chunks),
            "sections_detected": sorted(document_sections, key=str.casefold),
            "pages": pages,
            "chunks_file": jsonl_path.name,
        }
        _atomic_json(manifest_path, manifest)
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Could not persist ingestion output path=%s", source)
        raise IngestionError("INGESTION_ERROR", f"Could not persist ingestion result: {exc}", source=relative) from exc
    return {**manifest, "output_jsonl": str(jsonl_path), "output_manifest": str(manifest_path), "chunks": chunks}


def ingest_knowledge_base(
    knowledge_root: Path,
    output_dir: Path,
    settings: Settings | None = None,
    *,
    ocr_runner: Callable[..., str] | None = None,
) -> list[dict[str, object]]:
    results = []
    for pdf_path in discover_pdfs(knowledge_root):
        results.append(ingest_pdf(pdf_path, knowledge_root, output_dir, settings, ocr_runner=ocr_runner))
    return results
