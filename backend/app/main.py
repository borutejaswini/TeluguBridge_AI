import time
import uuid
from pathlib import Path

import httpx
import soundfile as sf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.services.rag import RagService, rag_service
from app.services.whisper import get_whisper_service

settings = get_settings()
configure_logging(settings.log_level)
app = FastAPI(title="TeluguBridge AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "telugubridge-api"}


@app.post("/api/rag/query")
def rag_query(request: dict) -> dict:
    query = request.get("query") if isinstance(request, dict) else None
    if not isinstance(query, str) or not query.strip():
        from app.core.errors import AppError

        raise AppError("EMPTY_QUERY", "query must be a non-empty string", 422)
    return rag_service.query(query)


@app.post("/api/pipeline/run")
def pipeline_run(file: UploadFile = File(...)) -> dict:
    from app.core.errors import AppError
    from app.services.whisper import WhisperService

    pipeline_started = time.perf_counter()
    if not file.filename or Path(file.filename).suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
        raise AppError("INVALID_AUDIO", "Upload a supported audio file (WAV, MP3, M4A, FLAC, or OGG)", 415)
    root = Path(__file__).resolve().parents[2]
    temp_dir = root / "data" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload-{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
    audio_dir = root / "outputs" / "audio"
    try:
        with temp_path.open("wb") as output:
            total = 0
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    raise AppError("AUDIO_TOO_LARGE", "Audio uploads are limited to 50 MiB", 413)
                output.write(chunk)
        audio_started = time.perf_counter()
        try:
            audio_info = sf.info(str(temp_path))
            if audio_info.frames < 1 or audio_info.duration <= 0:
                raise ValueError("audio has no frames")
        except Exception as exc:
            raise AppError("AUDIO_DECODE_FAILED", f"Audio could not be read: {type(exc).__name__}: {exc}", 422) from exc
        try:
            audio_array, processed_sample_rate = WhisperService.load_audio(temp_path)
        except Exception as exc:
            raise AppError("AUDIO_DECODE_FAILED", f"Whisper audio preprocessing failed: {type(exc).__name__}: {exc}", 422) from exc
        audio_load_ms = (time.perf_counter() - audio_started) * 1000

        whisper_service = get_whisper_service()
        whisper_started = time.perf_counter()
        try:
            transcription = whisper_service.transcribe_array(audio_array, processed_sample_rate)
        except Exception as exc:
            if isinstance(exc, AppError):
                raise AppError("WHISPER_FAILED", exc.message, exc.status_code) from exc
            raise AppError("WHISPER_FAILED", f"Whisper stage failed: {type(exc).__name__}: {exc}", 503) from exc
        whisper_wall_ms = (time.perf_counter() - whisper_started) * 1000
        whisper_ms = transcription.inference_seconds * 1000
        if not transcription.text.strip():
            raise AppError("WHISPER_EMPTY_TRANSCRIPT", "Whisper returned an empty transcription", 422)

        rag = RagService(settings)
        rag_result = rag.query(transcription.text)
        if rag_result["status"] == "insufficient_evidence":
            raise AppError("INSUFFICIENT_AGRICULTURE_EVIDENCE", rag_result["answer_telugu"], 422)

        tts_started = time.perf_counter()
        try:
            response = httpx.post(
                f"{settings.tts_url.rstrip('/')}/synthesize",
                json={"text": rag_result["answer_telugu"]},
                timeout=settings.tts_timeout_seconds,
            )
            response.raise_for_status()
            tts_result = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("error", {}).get("message", "TTS service returned an error")
            except ValueError:
                detail = "TTS service returned an error"
            raise AppError("TTS_FAILED", f"TTS stage failed: {detail}", 502) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError("TTS_UNAVAILABLE", f"TTS stage unavailable: {type(exc).__name__}: {exc}", 503) from exc
        tts_ms = (time.perf_counter() - tts_started) * 1000

        audio_filename = tts_result.get("audio_filename")
        if not isinstance(audio_filename, str) or Path(audio_filename).name != audio_filename or not audio_filename.endswith(".wav"):
            raise AppError("TTS_INVALID_RESPONSE", "TTS returned an invalid audio filename", 502)
        generated_audio = audio_dir / audio_filename
        if not generated_audio.is_file():
            raise AppError("TTS_AUDIO_MISSING", "TTS reported success but the generated WAV does not exist", 502)
        try:
            generated_info = sf.info(str(generated_audio))
            if generated_info.format != "WAV" or generated_info.frames <= 0 or generated_info.duration <= 0:
                raise ValueError("WAV contains no valid audio frames")
        except Exception as exc:
            raise AppError("TTS_AUDIO_INVALID", f"Generated WAV failed validation: {type(exc).__name__}: {exc}", 502) from exc

        timings = {
            "audio_loading_ms": round(audio_load_ms, 2),
            "whisper_ms": round(whisper_ms, 2),
            "whisper_stage_wall_ms": round(whisper_wall_ms, 2),
            "whisper_model_load_ms": round((whisper_service.model_load_seconds or 0) * 1000, 2),
            **{f"{key}_ms": round(float(rag_result["timings_ms"][key]), 2) for key in ("nllb", "embedding", "faiss", "reranker", "gemini")},
            "tts_ms": round(tts_ms, 2),
        }
        timings["total_pipeline_ms"] = round((time.perf_counter() - pipeline_started) * 1000, 2)
        return {
            "status": "success",
            "audio_filename": audio_filename,
            "audio_url": f"/api/audio/{audio_filename}",
            "transcript_telugu": transcription.text,
            "translated_query": rag_result["translated_query"],
            "answer_telugu": rag_result["answer_telugu"],
            "sources": rag_result["sources"],
            "retrieved_context": rag_result["retrieved_context"],
            "input_audio": {
                "format": audio_info.format,
                "sample_rate": audio_info.samplerate,
                "processed_sample_rate": processed_sample_rate,
                "channels": audio_info.channels,
                "duration_seconds": audio_info.duration,
            },
            "timings": timings,
            "whisper": {
                "model": settings.whisper_model_id,
                "lora_loaded": transcription.lora_loaded,
                "device": transcription.device,
                "model_dtype": transcription.model_dtype,
                "input_dtype": transcription.input_dtype,
                "duration_seconds": transcription.duration_seconds,
                "inference_seconds": transcription.inference_seconds,
                "rtf": transcription.real_time_factor,
                "peak_gpu_memory_mb": whisper_service.peak_gpu_memory_mb(),
            },
            "tts": {
                "model": settings.tts_model_id,
                "sample_rate": generated_info.samplerate,
                "duration_seconds": generated_info.duration,
                "device": "cpu",
            },
        }
    finally:
        temp_path.unlink(missing_ok=True)
        try:
            file.file.close()
        except Exception:
            pass


@app.get("/api/audio/{filename}")
def get_pipeline_audio(filename: str) -> FileResponse:
    from app.core.errors import AppError

    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise AppError("AUDIO_NOT_FOUND", "Generated audio was not found", 404)
    path = Path(__file__).resolve().parents[2] / "outputs" / "audio" / filename
    if not path.is_file():
        raise AppError("AUDIO_NOT_FOUND", "Generated audio was not found", 404)
    return FileResponse(path, media_type="audio/wav", filename=filename)
