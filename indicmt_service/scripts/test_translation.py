"""Real NLLB translation smoke test; run inside the indicmt_service environment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.service import NllbTranslationService


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    service = NllbTranslationService(get_settings())
    service.load()
    result = service.translate("ఈ రోజు రైతు పత్తి పంటను పరిశీలిస్తున్నాడు.")
    print(json.dumps({
        "model": result.model_id,
        "device": result.device,
        "input_language": result.input_language,
        "output_language": result.output_language,
        "model_load_seconds": round(service.load_seconds or 0, 3),
        "translation_latency_ms": round(result.latency_ms, 2),
        "input": "ఈ రోజు రైతు పత్తి పంటను పరిశీలిస్తున్నాడు.",
        "translation": result.translation,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
