from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.errors import TranslationServiceError
from app.service import NllbTranslationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
translator = NllbTranslationService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    translator.load()
    yield


app = FastAPI(title="IndicMT Telugu-English", version="0.1.0", lifespan=lifespan)


@app.exception_handler(TranslationServiceError)
async def translation_error_handler(_: Request, exc: TranslationServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if translator.is_loaded else "not_ready",
        "service": "indicmt",
        "model": settings.model_id,
        "loaded": translator.is_loaded,
        "device": settings.device,
        "input_language": settings.source_lang,
        "output_language": settings.target_lang,
        "model_load_seconds": translator.load_seconds,
    }


@app.post("/translate")
def translate(request: TranslateRequest) -> dict[str, object]:
    result = translator.translate(request.text)
    return {
        "translation": result.translation,
        "model": result.model_id,
        "device": result.device,
        "input_language": result.input_language,
        "output_language": result.output_language,
        "latency_ms": round(result.latency_ms, 2),
    }

