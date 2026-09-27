"""Exercise the real model through FastAPI startup, health, and translate routes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    phrase = "ఈ రోజు రైతు పత్తి పంటను పరిశీలిస్తున్నాడు."
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        translated = client.post("/translate", json={"text": phrase})
        translated.raise_for_status()
        print(json.dumps({
            "health": health.json(),
            "translation_response": translated.json(),
            "input": phrase,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

