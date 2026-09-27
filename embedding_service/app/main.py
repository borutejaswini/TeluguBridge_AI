from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.errors import EmbeddingServiceError
from app.service import BgeEmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
embedder = BgeEmbeddingService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    embedder.load()
    yield


app = FastAPI(title="BGE-M3 Embedding Service", version="0.1.0", lifespan=lifespan)


@app.exception_handler(EmbeddingServiceError)
async def embedding_error_handler(_: Request, exc: EmbeddingServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if embedder.is_loaded else "not_ready",
        "service": "bge_m3",
        "model": settings.model_id,
        "loaded": embedder.is_loaded,
        "device": settings.device,
        "dimension": embedder.dimension,
        "model_load_seconds": embedder.load_seconds,
    }


@app.post("/embed")
def embed(request: EmbedRequest) -> dict[str, object]:
    result = embedder.embed(request.texts)
    return {
        "embeddings": result.embeddings,
        "model": result.model_id,
        "device": result.device,
        "dimension": result.dimension,
        "count": len(result.embeddings),
        "latency_ms": round(result.latency_ms, 2),
    }

