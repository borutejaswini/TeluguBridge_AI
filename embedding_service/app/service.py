from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.config import Settings, get_settings
from app.errors import EmbeddingServiceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    model_id: str
    device: str
    dimension: int
    latency_ms: float


class BgeEmbeddingService:
    def __init__(
        self,
        settings: Settings | None = None,
        loader: Callable[[Settings], Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if self.settings.device.lower() != "cpu":
            raise EmbeddingServiceError(
                "CONFIGURATION_ERROR", "BGE_DEVICE must be cpu; the GPU is reserved for Whisper", 500
            )
        if self.settings.batch_size < 1 or self.settings.max_length < 1:
            raise EmbeddingServiceError("CONFIGURATION_ERROR", "batch size and max length must be positive", 500)
        self._loader = loader or self._load_sentence_transformer
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self.load_seconds: float | None = None
        self.dimension: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _load_sentence_transformer(settings: Settings) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            settings.model_id,
            device=settings.device,
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                model = self._loader(self.settings)
                # SentenceTransformer 5.x sets truncation length on the model; its
                # encode() method rejects max_length as an extra keyword argument.
                model.max_seq_length = self.settings.max_length
                dimension = int(model.get_embedding_dimension())
                if dimension != 1024:
                    raise EmbeddingServiceError(
                        "MODEL_CONFIGURATION_ERROR",
                        f"Expected BGE-M3 dense dimension 1024, got {dimension}",
                        503,
                    )
                self._model = model
                self.dimension = dimension
                self.load_seconds = time.perf_counter() - started
                logger.info(
                    "BGE-M3 loaded model=%s device=%s dimension=%d load_seconds=%.3f",
                    self.settings.model_id,
                    self.settings.device,
                    dimension,
                    self.load_seconds,
                )
            except EmbeddingServiceError:
                raise
            except Exception as exc:
                raise EmbeddingServiceError(
                    "MODEL_LOAD_ERROR", f"Could not load BGE model {self.settings.model_id}: {exc}", 503
                ) from exc

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not isinstance(texts, list) or not texts:
            raise EmbeddingServiceError("INVALID_INPUT", "texts must be a non-empty list", 422)
        if len(texts) > 32:
            raise EmbeddingServiceError("INVALID_INPUT", "at most 32 texts may be embedded per request", 413)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingServiceError("INVALID_INPUT", "each text must be a non-empty string", 422)
        if any(len(text) > 40_000 for text in texts):
            raise EmbeddingServiceError("INVALID_INPUT", "each text must be at most 40000 characters", 413)
        if not self.is_loaded:
            raise EmbeddingServiceError("SERVICE_NOT_READY", "BGE-M3 model has not been loaded", 503)

        started = time.perf_counter()
        try:
            with self._encode_lock:
                vectors = self._model.encode(
                    [text.strip() for text in texts],
                    batch_size=self.settings.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            array = np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingServiceError("EMBEDDING_ERROR", f"BGE-M3 embedding failed: {exc}", 500) from exc

        expected = (len(texts), self.dimension)
        if array.shape != expected or not np.isfinite(array).all():
            raise EmbeddingServiceError(
                "EMBEDDING_OUTPUT_ERROR", f"Expected finite embedding matrix {expected}, got {array.shape}", 500
            )
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info("BGE-M3 embedded count=%d device=cpu latency_ms=%.2f", len(texts), latency_ms)
        return EmbeddingResult(
            embeddings=array.tolist(),
            model_id=self.settings.model_id,
            device=self.settings.device,
            dimension=self.dimension,
            latency_ms=latency_ms,
        )
