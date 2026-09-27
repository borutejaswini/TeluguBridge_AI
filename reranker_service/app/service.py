from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from reranker_service.app.config import RERANKER_MODEL_ID, Settings
from reranker_service.app.errors import RerankerServiceError

logger = logging.getLogger(__name__)


class BgeRerankerService:
    def __init__(self, settings: Settings | None = None, loader: Callable[[Settings], Any] | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if self.settings.model_id != RERANKER_MODEL_ID:
            raise RerankerServiceError("CONFIGURATION_ERROR", f"Reranker model must be {RERANKER_MODEL_ID}", 500)
        if self.settings.device != "cpu":
            raise RerankerServiceError("CONFIGURATION_ERROR", "RERANKER_DEVICE must be cpu; the GPU is reserved for Whisper", 500)
        if self.settings.batch_size < 1:
            raise RerankerServiceError("CONFIGURATION_ERROR", "RERANKER_BATCH_SIZE must be positive", 500)
        self._loader = loader or self._load_cross_encoder
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._score_lock = threading.Lock()
        self.load_seconds: float | None = None

    @staticmethod
    def _load_cross_encoder(settings: Settings) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.model_id, device=settings.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                model = self._loader(self.settings)
                if hasattr(model, "eval"):
                    model.eval()
                self._model = model
                self.load_seconds = time.perf_counter() - started
                logger.info(
                    "BGE reranker loaded model=%s device=cpu load_seconds=%.3f",
                    self.settings.model_id,
                    self.load_seconds,
                )
            except RerankerServiceError:
                raise
            except Exception as exc:
                raise RerankerServiceError(
                    "RERANKER_MODEL_LOAD_FAILED", f"Could not load {self.settings.model_id}: {exc}", 503
                ) from exc

    def score(self, query: str, candidates: list[str]) -> dict[str, object]:
        if not isinstance(query, str) or not query.strip():
            raise RerankerServiceError("INVALID_INPUT", "query must be non-empty", 422)
        if not isinstance(candidates, list) or not candidates or len(candidates) > 32:
            raise RerankerServiceError("INVALID_INPUT", "candidates must contain 1 to 32 texts", 422)
        if any(not isinstance(text, str) or not text.strip() for text in candidates):
            raise RerankerServiceError("INVALID_INPUT", "each candidate must be non-empty text", 422)
        if not self.is_loaded:
            raise RerankerServiceError("SERVICE_NOT_READY", "BGE reranker model has not been loaded", 503)
        try:
            import torch

            pairs = [(query.strip(), text) for text in candidates]
            started = time.perf_counter()
            with self._score_lock:
                scores = self._model.predict(
                    pairs,
                    batch_size=self.settings.batch_size,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Sigmoid(),
                    convert_to_numpy=True,
                )
            latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            raise RerankerServiceError("RERANKER_INFERENCE_FAILED", f"Reranker scoring failed: {exc}", 500) from exc
        try:
            array = np.asarray(scores, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RerankerServiceError("INVALID_RERANKER_RESULT", f"Reranker returned invalid scores: {exc}", 500) from exc
        if array.shape != (len(candidates),) or not np.isfinite(array).all():
            raise RerankerServiceError(
                "INVALID_RERANKER_RESULT",
                f"Expected {len(candidates)} finite reranker scores, got shape {array.shape}",
                500,
            )
        return {
            "scores": [float(score) for score in array],
            "model": self.settings.model_id,
            "device": self.settings.device,
            "latency_ms": latency_ms,
        }

