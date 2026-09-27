from __future__ import annotations

from fastapi.testclient import TestClient

from reranker_service.app import main
from reranker_service.app.errors import RerankerServiceError


class FakeService:
    is_loaded = True
    load_seconds = 0.1
    settings = type("Settings", (), {"model_id": "BAAI/bge-reranker-v2-m3", "device": "cpu"})()

    def load(self):
        return None

    def score(self, query, candidates):
        return {"scores": [0.7 for _ in candidates], "model": self.settings.model_id, "device": "cpu", "latency_ms": 2.0}


def test_reranker_api_and_health(monkeypatch):
    monkeypatch.setattr(main, "reranker", FakeService())
    with TestClient(main.app) as client:
        health = client.get("/health")
        response = client.post("/rerank", json={"query": "rice blast symptoms", "candidates": ["text one", "text two"]})
    assert health.status_code == 200
    assert health.json()["loaded"] is True
    assert response.status_code == 200
    assert response.json()["scores"] == [0.7, 0.7]
    assert response.json()["count"] == 2


def test_reranker_api_returns_structured_error(monkeypatch):
    class Broken(FakeService):
        def score(self, query, candidates):
            raise RerankerServiceError("RERANKER_INFERENCE_FAILED", "expected", 500)

    monkeypatch.setattr(main, "reranker", Broken())
    with TestClient(main.app) as client:
        response = client.post("/rerank", json={"query": "q", "candidates": ["text"]})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "RERANKER_INFERENCE_FAILED"
