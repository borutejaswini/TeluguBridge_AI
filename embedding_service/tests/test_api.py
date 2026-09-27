from fastapi.testclient import TestClient

from app.main import app, embedder


class MockModel:
    def get_embedding_dimension(self):
        return 1024

    def encode(self, texts, **kwargs):
        return [[0.25] * 1024 for _ in texts]


def test_health_and_embed_api(monkeypatch):
    monkeypatch.setattr(embedder, "_loader", lambda _: MockModel())
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post("/embed", json={"texts": ["రైతు పత్తి పంటను పరిశీలిస్తున్నాడు"]})
    assert health.status_code == 200
    assert health.json()["loaded"] is True
    assert response.status_code == 200
    assert response.json()["device"] == "cpu"
    assert response.json()["dimension"] == 1024
    assert len(response.json()["embeddings"][0]) == 1024
