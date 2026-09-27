from __future__ import annotations

import logging
import shutil
import subprocess

from app.config import Settings
from app.errors import IngestionError

logger = logging.getLogger(__name__)


def resolve_tesseract(settings: Settings) -> str:
    executable = settings.tesseract_cmd.strip()
    resolved = shutil.which(executable) if executable else shutil.which("tesseract")
    if executable and not resolved:
        from pathlib import Path

        candidate = Path(executable).expanduser()
        if candidate.is_file():
            resolved = str(candidate)
    if not resolved:
        raise IngestionError("OCR_NOT_INSTALLED", "OCR is required for this page, but Tesseract was not found")
    return resolved


def ocr_page(page, settings: Settings, *, source: str, page_number: int) -> str:
    try:
        command = resolve_tesseract(settings)
    except IngestionError as exc:
        raise IngestionError(exc.code, exc.message, source=source, page=page_number) from exc
    try:
        language_result = subprocess.run(
            [command, "--list-langs"], capture_output=True, text=True, timeout=30, check=False
        )
        if language_result.returncode != 0:
            raise IngestionError("OCR_ERROR", f"Tesseract language check failed: {language_result.stderr.strip()}", source=source, page=page_number)
        installed = {
            line.strip()
            for line in language_result.stdout.splitlines()
            if line.strip() and not line.lower().startswith("list of available")
        }
        requested = settings.ocr_languages.split("+")
        missing = [language for language in requested if language not in installed]
        if missing:
            raise IngestionError(
                "OCR_ERROR",
                f"Configured Tesseract language pack(s) unavailable: {', '.join(missing)}",
                source=source,
                page=page_number,
            )
        pixmap = page.get_pixmap(dpi=300, alpha=False)
        result = subprocess.run(
            [command, "stdin", "stdout", "-l", settings.ocr_languages, "--psm", "6"],
            input=pixmap.tobytes("png"),
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise IngestionError("OCR_ERROR", f"Tesseract OCR failed: {stderr}", source=source, page=page_number)
        return result.stdout.decode("utf-8", errors="replace")
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("OCR failed source=%s page=%d", source, page_number)
        raise IngestionError("OCR_ERROR", f"Tesseract OCR failed: {exc}", source=source, page=page_number) from exc
