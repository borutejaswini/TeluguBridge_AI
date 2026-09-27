"""Standalone Whisper benchmark: python scripts/test_whisper.py AUDIO_FILE"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.whisper import WhisperService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    service = WhisperService()
    result = service.transcribe(args.audio)
    print(json.dumps({
        "audio_duration_seconds": round(result.duration_seconds, 3),
        "transcription": result.text,
        "inference_seconds": round(result.inference_seconds, 3),
        "real_time_factor": round(result.real_time_factor, 3),
        "peak_gpu_memory_mb": service.peak_gpu_memory_mb(),
        "device": result.device,
        "lora_loaded": result.lora_loaded,
        "model_load_seconds": round(service.model_load_seconds or 0, 3),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

