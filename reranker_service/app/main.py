from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from reranker_service.app.config import Settings
from reranker_service.app.errors import RerankerServiceError
from reranker_service.app.service import BgeRerankerService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = Settings.from_env()
reranker = BgeRerankerService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    reranker.load()
    yield


app = FastAPI(title="BGE Reranker Service", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RerankerServiceError)
async def reranker_error_handler(_: Request, exc: RerankerServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.as_dict())


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8_000)
    candidates: list[str] = Field(min_length=1, max_length=32)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if reranker.is_loaded else "not_ready",
        "service": "bge_reranker",
        "model": settings.model_id,
        "loaded": reranker.is_loaded,
        "device": settings.device,
        "model_load_seconds": reranker.load_seconds,
    }


@app.post("/rerank")
def rerank(request: RerankRequest) -> dict[str, object]:
    result = reranker.score(request.query, request.candidates)
    return {**result, "count": len(result["scores"])}

