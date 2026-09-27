from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch

from app.config import get_settings
from app.service import IndicParlerService


def main() -> None:
    settings = get_settings()
    service = IndicParlerService(settings)
    service.load()
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model: {settings.model_id}")
    print("Device: cpu")
    print(f"Model load seconds: {service.load_seconds:.3f}")
    result = service.synthesize(
        "వరి పంటలో కాండం తొలుచు పురుగు లక్షణాలను గమనించి, అవసరమైతే స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
    )
    info = sf.info(str(result.audio_path))
    data, sample_rate = sf.read(str(result.audio_path), dtype="float32")
    if info.format != "WAV" or info.frames < 1 or info.duration <= 0 or sample_rate != result.sample_rate or not data.size:
        raise RuntimeError("Generated TTS output failed WAV readability/duration validation")
    print(f"Audio file: {result.audio_path}")
    print(f"WAV size bytes: {result.audio_path.stat().st_size}")
    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {info.channels}")
    print(f"Duration seconds: {info.duration:.3f}")
    print(f"Synthesis latency ms: {result.latency_ms:.2f}")


if __name__ == "__main__":
    main()
