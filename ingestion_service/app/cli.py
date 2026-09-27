from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.errors import IngestionError
from app.pipeline import discover_pdfs, ingest_pdf


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Discover, extract, OCR-fallback, and chunk agriculture PDFs")
    parser.add_argument("--knowledge-root", type=Path, default=PROJECT_ROOT / "knowledge_base")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed_documents")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = get_settings()
        paths = discover_pdfs(args.knowledge_root)
        results = [ingest_pdf(path, args.knowledge_root, args.output_dir, settings) for path in paths]
        print(json.dumps({"discovered_pdf_count": len(paths), "documents": [
            {k: v for k, v in result.items() if k != "chunks"} for result in results
        ]}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except IngestionError as exc:
        logging.getLogger(__name__).exception("Ingestion stopped code=%s source=%s page=%s", exc.code, exc.source, exc.page)
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

