from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.service import BgeEmbeddingService


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run a real BGE-M3 embedding smoke test")
    parser.add_argument("text", nargs="?", default="పత్తి పంటలో తెగులు కనిపిస్తే ఏ చర్యలు తీసుకోవాలి?")
    args = parser.parse_args()
    service = BgeEmbeddingService(get_settings())
    started = time.perf_counter()
    service.load()
    result = service.embed([args.text])
    print(json.dumps({
        "model_load_seconds": round(service.load_seconds or 0, 3),
        "embedding_latency_ms": round(result.latency_ms, 2),
        "device": result.device,
        "model": result.model_id,
        "input_text": args.text,
        "dimension": result.dimension,
        "embedding_norm": round(sum(x * x for x in result.embeddings[0]) ** 0.5, 6),
        "total_seconds": round(time.perf_counter() - started, 3),
        "real_model_inference": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
