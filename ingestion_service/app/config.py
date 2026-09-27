from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_local_env() -> None:
    env_file = PROJECT_ROOT / "ingestion_service" / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


@dataclass(frozen=True)
class Settings:
    tesseract_cmd: str = ""
    ocr_languages: str = "eng"
    ocr_min_text_chars: int = 20
    ocr_min_alnum_ratio: float = 0.25
    chunk_target_chars: int = 1800


def get_settings() -> Settings:
    _load_local_env()
    try:
        min_chars = int(os.getenv("OCR_MIN_TEXT_CHARS", "20"))
        min_ratio = float(os.getenv("OCR_MIN_ALNUM_RATIO", "0.25"))
        chunk_chars = int(os.getenv("CHUNK_TARGET_CHARS", "1800"))
    except ValueError as exc:
        from app.errors import IngestionError

        raise IngestionError("CONFIG_ERROR", f"Invalid numeric ingestion setting: {exc}") from exc
    if min_chars < 1 or not 0 <= min_ratio <= 1 or chunk_chars < 200:
        from app.errors import IngestionError

        raise IngestionError("CONFIG_ERROR", "OCR thresholds and chunk target are outside valid ranges")
    languages = os.getenv("OCR_LANGUAGES", "eng").strip()
    if not languages or any(not part.isalnum() for part in languages.split("+")):
        from app.errors import IngestionError

        raise IngestionError("CONFIG_ERROR", "OCR_LANGUAGES must be Tesseract language codes separated by '+'")
    return Settings(
        tesseract_cmd=os.getenv("TESSERACT_CMD", "").strip(),
        ocr_languages=languages,
        ocr_min_text_chars=min_chars,
        ocr_min_alnum_ratio=min_ratio,
        chunk_target_chars=chunk_chars,
    )
