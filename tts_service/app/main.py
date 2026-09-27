from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.errors import TtsError
from app.service import service

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(title="Indic Parler-TTS", version="1.0.0", lifespan=lifespan)


@app.exception_handler(TtsError)
async def tts_error_handler(_: Request, exc: TtsError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if service.is_loaded else "not_ready",
        "service": "indic-parler-tts",
        "model": settings.model_id,
        "loaded": service.is_loaded,
        "device": settings.device,
        "sample_rate": int(service._model.config.sampling_rate) if service.is_loaded else None,
        "model_load_seconds": service.load_seconds,
    }


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest) -> dict[str, Any]:
    result = service.synthesize(request.text)
    try:
        info = sf.info(str(result.audio_path))
        valid_wav = info.format == "WAV" and info.frames > 0 and info.duration > 0 and info.samplerate == result.sample_rate
    except Exception as exc:
        result.audio_path.unlink(missing_ok=True)
        raise TtsError("TTS_INVALID_AUDIO", f"Generated audio could not be read as WAV: {exc}", 500) from exc
    if not valid_wav:
        result.audio_path.unlink(missing_ok=True)
        raise TtsError("TTS_INVALID_AUDIO", "Generated audio does not contain a valid, non-empty WAV stream", 500)
    return {
        "audio_filename": result.audio_path.name,
        "audio_url": f"/audio/{result.audio_path.name}",
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": round(info.duration, 3),
        "latency_ms": round(result.latency_ms, 2),
        "device": result.device,
        "model": result.model_id,
    }


@app.get("/audio/{filename}")
def get_audio(filename: str):
    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise TtsError("AUDIO_NOT_FOUND", "Requested audio file was not found", 404)
    path = settings.output_dir / filename
    if not path.is_file():
        raise TtsError("AUDIO_NOT_FOUND", "Requested audio file was not found", 404)
    from fastapi.responses import FileResponse

    return FileResponse(path, media_type="audio/wav", filename=filename)
