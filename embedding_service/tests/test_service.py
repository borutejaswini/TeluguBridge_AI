import numpy as np
import pytest

from app.config import Settings
from app.errors import EmbeddingServiceError
from app.service import BgeEmbeddingService


class MockModel:
    def get_embedding_dimension(self):
        return 1024

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.tile(np.array([[1.0] + [0.0] * 1023], dtype=np.float32), (len(texts), 1))


def test_load_once_and_embed_dense_vectors():
    calls = []
    service = BgeEmbeddingService(Settings(), loader=lambda _: calls.append(1) or MockModel())
    service.load()
    service.load()
    result = service.embed(["పత్తి పంటలో తెగులు ఎలా నియంత్రించాలి?", "Cotton crop disease control"])
    assert len(calls) == 1
    assert service.is_loaded
    assert result.dimension == 1024
    assert result.device == "cpu"
    assert len(result.embeddings) == 2
    assert len(result.embeddings[0]) == 1024
    assert result.embeddings[0][0] == 1.0


def test_rejects_empty_text():
    service = BgeEmbeddingService(Settings(), loader=lambda _: MockModel())
    service.load()
    with pytest.raises(EmbeddingServiceError, match="non-empty"):
        service.embed([" "])


def test_rejects_non_cpu_device():
    with pytest.raises(EmbeddingServiceError, match="must be cpu"):
        BgeEmbeddingService(Settings(device="cuda"), loader=lambda _: MockModel())


def test_refuses_unexpected_embedding_shape():
    class WrongModel(MockModel):
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 3), dtype=np.float32)

    service = BgeEmbeddingService(Settings(), loader=lambda _: WrongModel())
    service.load()
    with pytest.raises(EmbeddingServiceError, match="Expected finite embedding matrix"):
        service.embed(["tomato wilt"])
