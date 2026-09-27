"""Whisper inference kept behind a single lazy-loaded service instance."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError, ModelLoadError

logger = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 16_000
MAX_NEW_TOKENS = 444


@dataclass(frozen=True)
class Transcription:
    text: str
    duration_seconds: float
    inference_seconds: float
    real_time_factor: float
    device: str
    model_dtype: str
    input_dtype: str
    lora_loaded: bool


class WhisperService:
    """Load Whisper on first use; serialize GPU work and reuse model thereafter."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.max_concurrent_whisper < 1:
            raise ConfigurationError("MAX_CONCURRENT_WHISPER must be at least 1")
        self._semaphore = threading.BoundedSemaphore(self.settings.max_concurrent_whisper)
        self._load_lock = threading.Lock()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._lora_loaded = False
        self.model_load_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self, torch: Any) -> str:
        configured = self.settings.whisper_device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if configured.startswith("cuda") and not torch.cuda.is_available():
            raise ConfigurationError("WHISPER_DEVICE requests CUDA, but CUDA is unavailable")
        return configured

    def _validate_lora_path(self) -> Path | None:
        configured = self.settings.whisper_lora_path.strip()
        if not configured:
            return None
        adapter = Path(configured).expanduser()
        if not adapter.is_absolute():
            adapter = (Path.cwd() / adapter).resolve()
        required = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ModelLoadError(
                "WHISPER_LORA_PATH is configured but required adapter files are missing: "
                + ", ".join(missing)
            )
        return adapter

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                self.settings.validate_hf_cache_paths()
                for key, value in (
                    ("HF_HOME", self.settings.hf_home),
                    ("HF_HUB_CACHE", self.settings.hf_hub_cache),
                    ("TRANSFORMERS_CACHE", self.settings.transformers_cache),
                ):
                    if value:
                        os.environ[key] = str(Path(value).expanduser())
                adapter = self._validate_lora_path()
                self._device = self._resolve_device(torch)
                token = self.settings.hf_token or None
                cache_dir = (
                    self.settings.hf_hub_cache
                    or self.settings.hf_home
                    or self.settings.transformers_cache
                    or None
                )
                self._processor = AutoProcessor.from_pretrained(
                    self.settings.whisper_model_id,
                    token=token,
                    cache_dir=cache_dir,
                )
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.settings.whisper_model_id,
                    token=token,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self._device.startswith("cuda") else torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                if adapter is not None:
                    try:
                        from peft import PeftModel

                        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
                    except Exception as exc:
                        raise ModelLoadError(f"Could not load configured Whisper LoRA adapter: {exc}") from exc
                    self._lora_loaded = True
                self._model = model.to(self._device)
                self._model.eval()
                self.model_load_seconds = time.perf_counter() - started
                logger.info(
                    "Whisper loaded model=%s device=%s lora=%s load_seconds=%.3f",
                    self.settings.whisper_model_id,
                    self._device,
                    self._lora_loaded,
                    self.model_load_seconds,
                )
            except (ConfigurationError, ModelLoadError):
                raise
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                if any(term in lowered for term in ("401", "403", "gated", "unauthorized", "forbidden")):
                    code = "HF_AUTH_ERROR" if self.settings.hf_token else "MODEL_ACCESS_ERROR"
                    raise ModelLoadError(f"{code}: cannot access {self.settings.whisper_model_id}: {message}") from exc
                raise ModelLoadError(f"Unable to load Whisper model {self.settings.whisper_model_id}: {message}") from exc

    @staticmethod
    def load_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
        """Decode any supported audio file, converting it to 16 kHz mono."""
        try:
            import librosa

            audio, _ = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ConfigurationError(f"Could not decode/resample audio to 16 kHz mono: {exc}") from exc
        if audio.size == 0:
            raise ConfigurationError("Audio contains no samples")
        return np.asarray(audio, dtype=np.float32), TARGET_SAMPLE_RATE

    def transcribe(self, audio_path: str | Path) -> Transcription:
        audio, sample_rate = self.load_audio(audio_path)
        return self.transcribe_array(audio, sample_rate)

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ConfigurationError(f"Audio must be resampled to {TARGET_SAMPLE_RATE} Hz before inference")
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            raise ConfigurationError("Audio contains no samples")
        self.load()
        import torch

        duration = mono.size / sample_rate
        with self._semaphore, torch.inference_mode():
            inputs = self._processor(mono, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            model_parameter = next(self._model.parameters())
            model_device = model_parameter.device
            model_dtype = model_parameter.dtype
            input_features = inputs["input_features"].to(
                device=model_device,
                dtype=model_dtype,
            )
            logger.info(
                "Whisper generation tensors model_device=%s model_dtype=%s "
                "input_features_device=%s input_features_dtype=%s",
                model_device,
                model_dtype,
                input_features.device,
                input_features.dtype,
            )
            if input_features.device != model_device or input_features.dtype != model_dtype:
                raise RuntimeError(
                    "Whisper input/model mismatch after conversion: "
                    f"input_features=({input_features.device}, {input_features.dtype}), "
                    f"model=({model_device}, {model_dtype})"
                )
            if self._device and self._device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            generated = self._model.generate(
                input_features,
                max_new_tokens=MAX_NEW_TOKENS,
                language="telugu",
                task="transcribe",
            )
            inference_seconds = time.perf_counter() - start
            text = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return Transcription(
            text=text,
            duration_seconds=duration,
            inference_seconds=inference_seconds,
            real_time_factor=inference_seconds / duration,
            device=self._device or "unknown",
            model_dtype=str(model_dtype),
            input_dtype=str(input_features.dtype),
            lora_loaded=self._lora_loaded,
        )

    def peak_gpu_memory_mb(self) -> float | None:
        if not self._device or not self._device.startswith("cuda"):
            return None
        import torch

        return torch.cuda.max_memory_allocated() / (1024 * 1024)


_service: WhisperService | None = None
_service_lock = threading.Lock()


def get_whisper_service() -> WhisperService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = WhisperService()
    return _service
