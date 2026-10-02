import time
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.services.rag import RagService, rag_service
from app.services.tts_jobs import get_tts_job, submit_tts_job
from app.services.whisper import get_whisper_service

from contextlib import asynccontextmanager
import threading

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-warm Whisper in a background thread so first user request is instant
    whisper = get_whisper_service()
    threading.Thread(target=whisper.load, daemon=True, name="whisper-preloader").start()
    yield


app = FastAPI(title="TeluguBridge AI", version="0.1.0", lifespan=lifespan)
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
    # Accept either 'query' or 'question' key for flexibility
    query = request.get("query") or request.get("question") if isinstance(request, dict) else None
    if not isinstance(query, str) or not query.strip():
        from app.core.errors import AppError
        raise AppError("EMPTY_QUERY", "query must be a non-empty string", 422)
    return rag_service.query(query)


@app.post("/api/pipeline/run")
def pipeline_run(file: UploadFile = File(...)) -> dict:
    from app.core.errors import AppError
    from app.services.whisper import WhisperService

    pipeline_started = time.perf_counter()
    if not file.filename or Path(file.filename).suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac", ".opus"}:
        raise AppError("INVALID_AUDIO", "Upload a supported audio file (WAV, MP3, M4A, FLAC, OGG, WebM, MP4, AAC, or OPUS)", 415)
    root = Path(__file__).resolve().parents[2]
    temp_dir = root / "data" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload-{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
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
            audio_array, processed_sample_rate = WhisperService.load_audio(temp_path)
        except Exception as exc:
            raise AppError("AUDIO_DECODE_FAILED", f"Whisper audio preprocessing failed: {type(exc).__name__}: {exc}", 422) from exc
        if audio_array.size < 1:
            raise AppError("AUDIO_DECODE_FAILED", "Audio contains no samples", 422)
        # soundfile cannot inspect every browser/container format (for example
        # WebM/Opus). Decoding above is the authoritative check; use metadata
        # when libsndfile supports the format and otherwise report decoded data.
        try:
            audio_info = sf.info(str(temp_path))
            original_sample_rate = audio_info.samplerate
            original_channels = audio_info.channels
            audio_format = audio_info.format
        except Exception:
            original_sample_rate = None
            original_channels = None
            audio_format = Path(file.filename).suffix.lstrip(".").upper()
        duration_seconds = audio_array.size / processed_sample_rate
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

        tts_queue_started = time.perf_counter()
        audio_job_id = submit_tts_job(rag_result["answer_telugu"], settings)
        tts_queue_ms = (time.perf_counter() - tts_queue_started) * 1000

        timings = {
            "audio_loading_ms": round(audio_load_ms, 2),
            "whisper_ms": round(whisper_ms, 2),
            "whisper_stage_wall_ms": round(whisper_wall_ms, 2),
            "whisper_model_load_ms": round((whisper_service.model_load_seconds or 0) * 1000, 2),
            **{f"{key}_ms": round(float(rag_result["timings_ms"].get(key, 0)), 2) for key in ("nllb", "answer_translation", "embedding", "faiss", "reranker", "llm")},
            "tts_queue_ms": round(tts_queue_ms, 2),
        }
        timings["total_pipeline_ms"] = round((time.perf_counter() - pipeline_started) * 1000, 2)
        return {
            "status": "success",
            "audio_filename": None,
            "audio_url": None,
            "audio_job_id": audio_job_id,
            "audio_status": "queued",
            "transcript_telugu": transcription.text,
            "translated_query": rag_result["translated_query"],
            "answer_telugu": rag_result["answer_telugu"],
            "answer_english": rag_result["answer_english"],
            "answer_status": rag_result["status"],
            "answer_grounded": rag_result["answer_grounded"],
            "answer_basis": rag_result["answer_basis"],
            "llm_provider": rag_result["llm_provider"],
            "sources": rag_result["sources"],
            "retrieved_context": rag_result["retrieved_context"],
            "input_audio": {
                "format": audio_format,
                "sample_rate": original_sample_rate,
                "processed_sample_rate": processed_sample_rate,
                "channels": original_channels,
                "duration_seconds": duration_seconds,
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
                "device": "cpu",
                "status": "queued",
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


@app.get("/api/audio-jobs/{job_id}")
def get_pipeline_audio_job(job_id: str) -> dict:
    job = get_tts_job(job_id)
    if job is None:
        raise AppError("AUDIO_JOB_NOT_FOUND", "Audio generation job was not found or has expired", 404)
    return {"job_id": job_id, **job}
