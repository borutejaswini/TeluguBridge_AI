"""LoRA verification benchmark: python scripts/test_whisper_lora.py AUDIO_FILE"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.whisper import WhisperService
from app.core.errors import ModelLoadError
from app.services.whisper import TARGET_SAMPLE_RATE
from app.core.logging import configure_logging


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    configure_logging("INFO")
    service = WhisperService()
    service.load()
    if not service._lora_loaded:
        raise ModelLoadError("LoRA test requires WHISPER_LORA_PATH to be configured and loaded")
    result = service.transcribe(args.audio)
    import torch

    print(json.dumps({
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "whisper_model": service.settings.whisper_model_id,
        "lora_loaded": result.lora_loaded,
        "adapter_path": service.settings.whisper_lora_path,
        "device": result.device,
        "model_dtype": result.model_dtype,
        "input_dtype": result.input_dtype,
        "audio_duration_seconds": round(result.duration_seconds, 3),
        "input_sample_rate_hz": sf.info(args.audio).samplerate,
        "processed_sample_rate_hz": TARGET_SAMPLE_RATE,
        "inference_seconds": round(result.inference_seconds, 3),
        "real_time_factor": round(result.real_time_factor, 3),
        "peak_gpu_memory_mb": service.peak_gpu_memory_mb(),
        "model_load_seconds": round(service.model_load_seconds or 0, 3),
        "transcription": result.text,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
