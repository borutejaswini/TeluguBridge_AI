import json
from pathlib import Path

import fitz
import pytest

from app.chunking import chunk_page
from app.cleaning import clean_page_text, edge_lines, repeated_edge_lines
from app.config import Settings
from app.errors import IngestionError
from app.pipeline import _quality_sufficient, classify_text, discover_pdfs, ingest_pdf


def make_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path


def test_recursive_discovery(tmp_path):
    root = tmp_path / "knowledge_base"
    make_pdf(root / "paddy.pdf", ["Paddy information."])
    make_pdf(root / "nested" / "deep" / "wheat.pdf", ["Wheat information."])
    (root / "nested" / "no.pdf.txt").write_text("not a PDF")
    paths = discover_pdfs(root)
    assert [p.relative_to(root).as_posix() for p in paths] == ["nested/deep/wheat.pdf", "paddy.pdf"]


def test_page_extraction_and_metadata(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "nested" / "paddy.pdf", ["STEM BORER\nSymptoms\nDead heart in young crop."])
    result = ingest_pdf(path, root, tmp_path / "out", Settings(chunk_target_chars=200))
    row = result["chunks"][0]
    assert result["page_count"] == 1
    assert row["page_number"] == 1
    assert row["extraction_method"] == "text"
    assert row["ocr_used"] is False
    assert row["source_filename"] == "paddy.pdf"
    assert row["relative_source_path"] == "nested/paddy.pdf"
    assert row["classification"] == "KNOWLEDGE"
    assert "Dead heart" in row["text"]


def test_page_by_page_extraction_metadata(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "paddy.pdf", ["First page text about rice pests.", "Second page text about rice diseases."])
    result = ingest_pdf(
        path,
        root,
        tmp_path / "out",
        Settings(chunk_target_chars=200),
        ocr_runner=lambda *args, **kwargs: pytest.fail("OCR must not run when usable text was extracted"),
    )
    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    assert all(page["extraction_method"] == "text" for page in result["pages"])
    assert all(page["ocr_used"] is False for page in result["pages"])


def test_continued_section_metadata_crosses_page_boundary_without_rewriting_text(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(
        root / "paddy.pdf",
        ["Stem Borer\nSymptoms\nDead hearts appear in the young crop.", "Additional stem borer symptoms continue on the leaf sheath."],
    )
    result = ingest_pdf(path, root, tmp_path / "out", Settings())
    second_page = next(row for row in result["chunks"] if row["page_number"] == 2)
    assert second_page["section"] == "Stem Borer > Symptoms"
    assert second_page["text"] == "Additional stem borer symptoms continue on the leaf sheath."


def test_ocr_quality_decision():
    settings = Settings(ocr_min_text_chars=20, ocr_min_alnum_ratio=0.25)
    assert not _quality_sufficient("", settings)
    assert not _quality_sufficient("%%%%%%%", settings)
    assert _quality_sufficient("Rice stem borer symptoms are visible in the crop.", settings)
    assert _quality_sufficient("Rice management guidance.", Settings())


def test_repeated_header_footer_cleaning():
    pairs = [edge_lines("PADDY GUIDE\nUnique body one\nAGRI DEPT") for _ in range(3)]
    repeated = repeated_edge_lines(pairs, 3)
    cleaned = clean_page_text("PADDY GUIDE\n1\nPest details here\nAGRI DEPT", repeated)
    assert "PADDY GUIDE" not in cleaned
    assert "AGRI DEPT" not in cleaned
    assert "Pest details" in cleaned
    assert "\n1\n" not in f"\n{cleaned}\n"


def test_metadata_source_identity_and_page_number(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "paddy.pdf", ["Rice management guidance.", "Stem borer control measures."])
    result = ingest_pdf(path, root, tmp_path / "out")
    assert result["document_id"] == result["chunks"][0]["document_id"]
    assert {row["page_number"] for row in result["chunks"]} == {1, 2}


def test_ocr_fallback_flag_and_method(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "scan.pdf", [""])
    calls = []

    def fake_ocr(page, settings, *, source, page_number):
        calls.append((source, page_number))
        return "RICE BLAST\nSymptoms\nLesions on leaves and neck of the panicle."

    result = ingest_pdf(path, root, tmp_path / "out", ocr_runner=fake_ocr)
    assert calls == [("scan.pdf", 1)]
    assert result["chunks"][0]["extraction_method"] == "ocr"
    assert result["chunks"][0]["ocr_used"] is True


def test_insufficient_text_and_missing_tesseract_returns_structured_error(tmp_path, monkeypatch):
    from app import ocr

    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "scan.pdf", [""])
    with pytest.raises(IngestionError) as error:
        ingest_pdf(path, root, tmp_path / "out")
    assert error.value.code == "OCR_NOT_INSTALLED"
    assert error.value.page == 1


def test_semantic_chunking_keeps_section_context_and_dosage(tmp_path):
    text = (
        "Stem Borer\n\nSymptoms\nDead hearts and white ears are seen.\n\n"
        "Management\nApply 2 ml per litre of water to the affected crop."
    )
    chunks = chunk_page(text, 200)
    assert len(chunks) == 2
    joined = "\n".join(body for _, body in chunks)
    assert "Stem Borer" in joined and "2 ml per litre" in joined
    assert chunks[0][0] == "Stem Borer > Symptoms"
    assert chunks[1][0] == "Stem Borer > Management"
    assert "Dead hearts" in chunks[0][1]
    assert "Apply 2 ml per litre of water" in chunks[1][1]


def test_inline_field_labels_do_not_replace_parent_pest_heading():
    chunks = chunk_page(
        "Stem Borer\n� Pest Name: Stem Borer\n� Management & Control Measures: Apply 2 ml per litre.",
        200,
    )
    assert len(chunks) == 1
    assert chunks[0][0] == "Stem Borer"
    assert "Pest Name: Stem Borer" in chunks[0][1]
    assert "Apply 2 ml per litre" in chunks[0][1]


def test_chunk_ids_are_deterministic():
    text = "Rice Blast\n\nBrown lesions are present."
    assert chunk_page(text, 200) == chunk_page(text, 200)


def test_empty_page_is_handled_without_fake_chunk(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "empty.pdf", [""])
    result = ingest_pdf(path, root, tmp_path / "out", ocr_runner=lambda *a, **k: "")
    assert result["chunk_count"] == 0
    assert result["pages"][0]["classification"] == "NOISE"
    assert Path(result["output_jsonl"]).read_text(encoding="utf-8") == ""


def test_invalid_pdf_returns_structured_error(tmp_path):
    root = tmp_path / "knowledge_base"
    root.mkdir()
    path = root / "bad.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(IngestionError) as error:
        ingest_pdf(path, root, tmp_path / "out")
    assert error.value.code == "INVALID_DOCUMENT"
    assert error.value.as_dict()["error"]["code"] == "INVALID_DOCUMENT"


def test_document_classification():
    assert classify_text("Stem borer symptoms and management in paddy fields.") == "KNOWLEDGE"
    assert classify_text("Contents\n1 Introduction\n2 Appendix\n3 References") == "NAVIGATION"
    assert classify_text("123 -- 99") == "NOISE"


def test_chunk_ids_deterministic_and_distinct_for_duplicates(tmp_path):
    root = tmp_path / "knowledge_base"
    source = make_pdf(root / "paddy.pdf", ["Rice Blast\nBrown lesions.", "Rice Blast\nBrown lesions."])
    first = ingest_pdf(source, root, tmp_path / "out")
    second = ingest_pdf(source, root, tmp_path / "other")
    ids1 = [chunk["chunk_id"] for chunk in first["chunks"]]
    ids2 = [chunk["chunk_id"] for chunk in second["chunks"]]
    assert ids1 == ids2
    assert len(set(ids1)) == 2


def test_intermediate_jsonl_and_manifest_are_persisted(tmp_path):
    root = tmp_path / "knowledge_base"
    path = make_pdf(root / "paddy.pdf", ["Bacterial Leaf Blight\nYellowing and drying from leaf tips."])
    result = ingest_pdf(path, root, tmp_path / "out")
    rows = [json.loads(line) for line in Path(result["output_jsonl"]).read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(Path(result["output_manifest"]).read_text(encoding="utf-8"))
    assert rows[0]["chunk_id"] == result["chunks"][0]["chunk_id"]
    assert manifest["page_count"] == 1
    assert manifest["chunk_count"] == 1


def test_ocr_unavailable_is_not_silently_skipped(tmp_path, monkeypatch):
    from app import ocr

    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    with pytest.raises(IngestionError) as error:
        ocr.resolve_tesseract(Settings())
    assert error.value.code == "OCR_NOT_INSTALLED"
