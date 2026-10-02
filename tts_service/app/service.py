from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

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


class GttsTtsService:
    """Google Text-to-Speech (gTTS) service for Telugu — instant CPU synthesis."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self.load_seconds: float | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            started = time.perf_counter()
            try:
                import gtts  # noqa: F401 — validate import on startup
                self._loaded = True
                self.load_seconds = time.perf_counter() - started
                logger.info("gTTS service ready load_seconds=%.3f", self.load_seconds)
            except ImportError as exc:
                raise TtsError(
                    "TTS_MODEL_LOAD_FAILED",
                    "gTTS is not installed. Run: pip install gTTS",
                    503,
                ) from exc

    def synthesize(self, text: str) -> Synthesis:
        if not isinstance(text, str) or not text.strip():
            raise TtsError("EMPTY_TEXT", "text must not be empty", 422)
        clean_text = text.strip()
        if len(clean_text) > self.settings.max_text_characters:
            raise TtsError(
                "TEXT_TOO_LONG",
                f"text exceeds {self.settings.max_text_characters} characters",
                422,
            )
        self.load()
        try:
            from gtts import gTTS
            import soundfile as sf
            import numpy as np

            started = time.perf_counter()

            # Synthesise Telugu speech using gTTS (uses Google TTS API, lang=te)
            tts_obj = gTTS(text=clean_text, lang=self.settings.gtts_lang, slow=self.settings.gtts_slow)

            output_dir = self.settings.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # gTTS writes MP3; we convert it to WAV for the existing pipeline
            mp3_path = output_dir / f"telugu-{uuid.uuid4().hex}.mp3"
            wav_path = output_dir / f"telugu-{uuid.uuid4().hex}.wav"

            with self._lock:
                tts_obj.save(str(mp3_path))

            # Convert MP3 → WAV using pydub (falls back to soundfile/librosa)
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_mp3(str(mp3_path))
                audio_segment = audio_segment.set_frame_rate(22050).set_channels(1)
                audio_segment.export(str(wav_path), format="wav")
            except Exception:
                # Fallback: use soundfile with scipy or librosa
                try:
                    import librosa
                    y, sr = librosa.load(str(mp3_path), sr=22050, mono=True)
                    sf.write(str(wav_path), y, sr, format="WAV", subtype="PCM_16")
                except Exception as exc2:
                    mp3_path.unlink(missing_ok=True)
                    raise TtsError(
                        "TTS_CONVERSION_FAILED",
                        f"MP3 → WAV conversion failed: {exc2}",
                        500,
                    ) from exc2
            finally:
                mp3_path.unlink(missing_ok=True)

            latency_ms = (time.perf_counter() - started) * 1000

            # Validate output
            info = sf.info(str(wav_path))
            duration = info.duration
            sample_rate = info.samplerate

            if duration <= 0 or not wav_path.is_file() or wav_path.stat().st_size <= 44:
                wav_path.unlink(missing_ok=True)
                raise TtsError("TTS_INVALID_AUDIO", "Generated WAV is empty or invalid", 500)

            logger.info(
                "gTTS synthesized Telugu audio duration=%.2fs latency_ms=%.0f",
                duration,
                latency_ms,
            )
            return Synthesis(wav_path, sample_rate, duration, latency_ms, "cpu", "gtts/te")

        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(
                "TTS_INFERENCE_FAILED",
                f"gTTS synthesis failed: {type(exc).__name__}: {exc}",
                500,
            ) from exc


service = GttsTtsService()
