from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from app.config import Settings, get_settings
from app.errors import TranslationServiceError

logger = logging.getLogger(__name__)

# Protect number/quantity spans so NLLB cannot silently rewrite them. Telugu and
# ASCII digits are supported; the matched unit is restored exactly as provided.
_QUANTITY_RE = re.compile(
    r"(?<!\w)(?:[0-9०-९౦-౯]+(?:[.,][0-9०-९౦-౯]+)?)(?:\s*(?:%|‰|"
    r"kg|kgs|g|gm|gms|mg|ml|l|litre|litres|liter|liters|"
    r"కిలోగ్రాములు|కిలోలు|గ్రాములు|గ్రా|మిల్లీగ్రాములు|మి\.గ్రా|"
    r"మిల్లీలీటర్లు|మి\.లీ|లీటర్లు|లీటరు|ఎకరాలు|ఎకరం|హెక్టార్లు|హెక్టారు|"
    r"రోజులు|రోజు|డిగ్రీలు|డిగ్రీ|వారాలు|వారం|నెలలు|నెల|క్వింటాళ్లు|క్వింటాల్)(?=$|[^\w]))?",
    re.IGNORECASE,
)
_SENTINEL_PREFIX = "TBQZX"


@dataclass(frozen=True)
class TranslationResult:
    translation: str
    latency_ms: float
    input_language: str
    output_language: str
    device: str
    model_id: str


class NllbTranslationService:
    def __init__(
        self,
        settings: Settings | None = None,
        loader: Callable[[Settings], tuple[Any, Any]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if self.settings.device.lower() != "cpu":
            raise TranslationServiceError(
                "CONFIGURATION_ERROR", "NLLB_DEVICE must be cpu to preserve GPU capacity for Whisper", 500
            )
        self._loader = loader or self._load_huggingface
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._translate_lock = threading.Lock()
        self.load_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @staticmethod
    def _load_huggingface(settings: Settings) -> tuple[Any, Any]:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        token = settings.hf_token or None
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            src_lang=settings.source_lang,
            token=token,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.model_id,
            token=token,
            use_safetensors=True,
        )
        return tokenizer, model

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                tokenizer, model = self._loader(self.settings)
                tokenizer.src_lang = self.settings.source_lang
                model.to("cpu")
                model.eval()
                target_id = tokenizer.convert_tokens_to_ids(self.settings.target_lang)
                unknown_id = getattr(tokenizer, "unk_token_id", None)
                if target_id is None or target_id == unknown_id:
                    raise TranslationServiceError(
                        "MODEL_CONFIGURATION_ERROR",
                        f"Target language token {self.settings.target_lang!r} is missing from the tokenizer",
                        500,
                    )
                self._tokenizer = tokenizer
                self._model = model
                self.load_seconds = time.perf_counter() - started
                logger.info(
                    "NLLB loaded model=%s device=cpu source=%s target=%s load_seconds=%.3f",
                    self.settings.model_id,
                    self.settings.source_lang,
                    self.settings.target_lang,
                    self.load_seconds,
                )
            except TranslationServiceError:
                raise
            except Exception as exc:
                message = str(exc)
                lowered = message.lower()
                if any(term in lowered for term in ("401", "403", "gated", "unauthorized", "forbidden")):
                    code = "HF_AUTH_ERROR" if self.settings.hf_token else "MODEL_ACCESS_ERROR"
                else:
                    code = "MODEL_LOAD_ERROR"
                raise TranslationServiceError(
                    code,
                    f"Could not load NLLB model {self.settings.model_id}: {message}",
                    503,
                ) from exc

    @staticmethod
    def _protect_quantities(text: str) -> tuple[str, list[tuple[str, str]]]:
        protected: list[tuple[str, str]] = []

        def replace(match: re.Match[str]) -> str:
            marker = f"{_SENTINEL_PREFIX}{len(protected):04d}ZX"
            protected.append((marker, match.group(0)))
            return marker

        return _QUANTITY_RE.sub(replace, text), protected

    @staticmethod
    def _restore_quantities(text: str, protected: list[tuple[str, str]]) -> str:
        restored = text
        for marker, original in protected:
            if marker.lower() not in restored.lower():
                raise TranslationServiceError(
                    "TRANSLATION_INTEGRITY_ERROR",
                    "NLLB output did not preserve a protected number or quantity; refusing to alter it silently",
                    502,
                )
            restored = re.sub(re.escape(marker), lambda _: original, restored, flags=re.IGNORECASE)
        return restored

    def translate(self, text: str) -> TranslationResult:
        if not isinstance(text, str) or not text.strip():
            raise TranslationServiceError("INVALID_INPUT", "text must be a non-empty string", 422)
        if len(text) > 20_000:
            raise TranslationServiceError("INVALID_INPUT", "text exceeds the 20000 character limit", 413)
        if not self.is_loaded:
            raise TranslationServiceError("SERVICE_NOT_READY", "NLLB model has not been loaded", 503)

        protected_text, quantities = self._protect_quantities(text.strip())
        started = time.perf_counter()
        with self._translate_lock, torch.inference_mode():
            encoded = self._tokenizer(
                protected_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.max_input_tokens,
            )
            encoded = {key: value.to("cpu") for key, value in encoded.items()}
            target_id = self._tokenizer.convert_tokens_to_ids(self.settings.target_lang)
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=self.settings.max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        translation = self._restore_quantities(decoded, quantities)
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "NLLB translation complete device=cpu input_language=%s output_language=%s latency_ms=%.2f",
            self.settings.source_lang,
            self.settings.target_lang,
            latency_ms,
        )
        return TranslationResult(
            translation=translation,
            latency_ms=latency_ms,
            input_language=self.settings.source_lang,
            output_language=self.settings.target_lang,
            device="cpu",
            model_id=self.settings.model_id,
        )
