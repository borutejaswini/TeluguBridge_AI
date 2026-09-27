# Phase 5 agriculture PDF ingestion

This isolated package discovers PDFs recursively under `knowledge_base/`, extracts one page at a time with PyMuPDF, and invokes Tesseract only for pages whose extracted text fails configured quality checks. It produces a deterministic JSONL file of semantic page-local chunks plus a JSON manifest under `data/processed_documents/`. No FAISS index is created.

Create its Python 3.12 environment with `setup.bat`, copy `.env.example` to `.env` if OCR settings need adjustment, and run `run_ingest.bat`. `TESSERACT_CMD` may name a Tesseract executable; blank uses `PATH`. `OCR_LANGUAGES` is a `+`-joined set of installed Tesseract language codes. If OCR is required and Tesseract is absent, the CLI returns structured `OCR_NOT_INSTALLED` and stops.

Each JSONL row includes document ID, source filename and relative path, 1-based source page, section (or null), deterministic chunk ID, classification, extraction method, OCR flag, and unchanged extracted text. The manifest records per-page extraction metadata, section names detected, totals, and the JSONL output name. Header/footer removal is conservative and based only on repeated page-edge text. Chunk boundaries follow paragraphs, headings, and list/table lines; oversized sentences remain intact rather than being cut at arbitrary character positions.

