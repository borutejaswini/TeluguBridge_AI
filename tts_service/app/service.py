from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.errors import TtsError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Synthesis:
    audio_path: Path
    sample_rate: int
    duration_seconds: float
    latency_ms: float
    device: str
    model_id: str


class IndicParlerService:
    """Keep one CPU Parler model loaded for the lifetime of this service."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._description_tokenizer: Any | None = None
        self._torch: Any | None = None
        self.load_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            if self.settings.device != "cpu":
                raise TtsError("INVALID_DEVICE", "Indic Parler-TTS must remain on CPU", 500)
            try:
                import torch
                from parler_tts import ParlerTTSForConditionalGeneration
                from transformers import AutoTokenizer

                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    self.settings.model_id,
                    token=self.settings.hf_token or None,
                    torch_dtype=torch.float32,
                    attn_implementation="eager",
                ).to("cpu")
                tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, token=self.settings.hf_token or None)
                description_tokenizer = AutoTokenizer.from_pretrained(
                    model.config.text_encoder._name_or_path, token=self.settings.hf_token or None
                )
                model.eval()
                self._model = model
                self._tokenizer = tokenizer
                self._description_tokenizer = description_tokenizer
                self._torch = torch
                self.load_seconds = time.perf_counter() - started
                logger.info("Indic Parler-TTS loaded model=%s device=cpu load_seconds=%.3f", self.settings.model_id, self.load_seconds)
            except TtsError:
                raise
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                lowered = detail.lower()
                code = "TTS_MODEL_ACCESS_ERROR" if any(term in lowered for term in ("gated", "401", "unauthorized", "restricted")) else "TTS_MODEL_LOAD_FAILED"
                raise TtsError(code, f"Could not load {self.settings.model_id}: {detail}", 503) from exc

    def synthesize(self, text: str) -> Synthesis:
        if not isinstance(text, str) or not text.strip():
            raise TtsError("EMPTY_TEXT", "text must not be empty", 422)
        clean_text = text.strip()
        if len(clean_text) > self.settings.max_text_characters:
            raise TtsError("TEXT_TOO_LONG", f"text exceeds {self.settings.max_text_characters} characters", 422)
        self.load()
        assert self._model is not None and self._torch is not None
        try:
            with self._lock, self._torch.inference_mode():
                description = self._description_tokenizer(self.settings.voice_description, return_tensors="pt")
                prompt = self._tokenizer(clean_text, return_tensors="pt")
                started = time.perf_counter()
                generated = self._model.generate(
                    input_ids=description.input_ids,
                    prompt_input_ids=prompt.input_ids,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                audio = generated.detach().cpu().numpy().squeeze()
                if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all() or float(np.max(np.abs(audio))) <= 0:
                    raise TtsError("TTS_EMPTY_AUDIO", "TTS model returned empty or invalid audio", 500)
                sample_rate = int(self._model.config.sampling_rate)
                output_dir = self.settings.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"telugu-{uuid.uuid4().hex}.wav"
                import soundfile as sf

                sf.write(str(output_path), audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
                duration = audio.size / sample_rate
                if duration <= 0 or not output_path.is_file() or output_path.stat().st_size <= 44:
                    output_path.unlink(missing_ok=True)
                    raise TtsError("TTS_INVALID_AUDIO", "Generated WAV failed the non-empty audio check", 500)
                return Synthesis(output_path, sample_rate, duration, latency_ms, "cpu", self.settings.model_id)
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError("TTS_INFERENCE_FAILED", f"Indic Parler-TTS inference failed: {type(exc).__name__}: {exc}", 500) from exc


service = IndicParlerService()
