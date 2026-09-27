from __future__ import annotations

import numpy as np
import pytest

from reranker_service.app.config import RERANKER_MODEL_ID, Settings
from reranker_service.app.errors import RerankerServiceError
from reranker_service.app.service import BgeRerankerService


class FakeCrossEncoder:
    def __init__(self, scores=None, error=None):
        self.scores = scores if scores is not None else [0.2, 0.8]
        self.error = error
        self.calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        if self.error:
            raise self.error
        return np.asarray(self.scores)


def test_reranker_config_defaults_exact_model_and_cpu():
    settings = Settings()
    assert settings.model_id == RERANKER_MODEL_ID == "BAAI/bge-reranker-v2-m3"
    assert settings.device == "cpu"
    with pytest.raises(RerankerServiceError, match="must be cpu"):
        BgeRerankerService(Settings(device="cuda"), loader=lambda _: FakeCrossEncoder())


def test_load_is_once_and_scores_each_query_candidate_pair():
    model = FakeCrossEncoder([0.2, 0.8])
    loader_calls = []
    service = BgeRerankerService(Settings(), loader=lambda settings: loader_calls.append(settings.model_id) or model)
    service.load()
    first_load_seconds = service.load_seconds
    service.load()
    result = service.score("rice pest symptoms", ["chunk a", "chunk b"])
    assert loader_calls == ["BAAI/bge-reranker-v2-m3"]
    assert model.eval_called
    assert model.calls[0][0] == [("rice pest symptoms", "chunk a"), ("rice pest symptoms", "chunk b")]
    assert model.calls[0][1]["show_progress_bar"] is False
    assert result["scores"] == [0.2, 0.8]
    assert result["device"] == "cpu"
    assert result["model"] == RERANKER_MODEL_ID
    assert result["latency_ms"] >= 0
    assert service.load_seconds == first_load_seconds


def test_bad_score_count_nonfinite_and_model_failure_are_structured():
    service = BgeRerankerService(Settings(), loader=lambda _: FakeCrossEncoder([0.1]))
    service.load()
    with pytest.raises(RerankerServiceError) as mismatch:
        service.score("query", ["a", "b"])
    assert mismatch.value.code == "INVALID_RERANKER_RESULT"

    service = BgeRerankerService(Settings(), loader=lambda _: FakeCrossEncoder([0.1, float("nan")]))
    service.load()
    with pytest.raises(RerankerServiceError) as nonfinite:
        service.score("query", ["a", "b"])
    assert nonfinite.value.code == "INVALID_RERANKER_RESULT"

    service = BgeRerankerService(Settings(), loader=lambda _: FakeCrossEncoder(error=RuntimeError("model error")))
    service.load()
    with pytest.raises(RerankerServiceError) as failure:
        service.score("query", ["a", "b"])
    assert failure.value.code == "RERANKER_INFERENCE_FAILED"


def test_load_failure_and_invalid_input_are_structured():
    service = BgeRerankerService(Settings(), loader=lambda _: (_ for _ in ()).throw(RuntimeError("load error")))
    with pytest.raises(RerankerServiceError) as load_error:
        service.load()
    assert load_error.value.code == "RERANKER_MODEL_LOAD_FAILED"

    ready = BgeRerankerService(Settings(), loader=lambda _: FakeCrossEncoder())
    ready.load()
    with pytest.raises(RerankerServiceError) as empty:
        ready.score(" ", ["candidate"])
    assert empty.value.code == "INVALID_INPUT"

