from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import uuid
from typing import Any

import httpx
import soundfile as sf

from app.core.config import Settings, get_settings


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="telugubridge-tts")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _run_job(job_id: str, text: str, settings: Settings) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["updated_at"] = time.time()
    try:
        response = httpx.post(
            f"{settings.tts_url.rstrip('/')}/synthesize",
            json={"text": text},
            timeout=httpx.Timeout(max(settings.tts_timeout_seconds, 3600.0), connect=15.0),
        )
        response.raise_for_status()
        result = response.json()
        filename = result.get("audio_filename") if isinstance(result, dict) else None
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".wav"):
            raise ValueError("TTS returned an invalid audio filename")

        audio_path = Path(__file__).resolve().parents[3] / "outputs" / "audio" / filename
        if not audio_path.is_file():
            raise ValueError("TTS reported success but the generated WAV does not exist")
        info = sf.info(str(audio_path))
        if info.format != "WAV" or info.frames <= 0 or info.duration <= 0:
            raise ValueError("TTS generated an invalid WAV file")

        with _jobs_lock:
            _jobs[job_id].update(
                status="complete",
                audio_url=f"/api/audio/{filename}",
                duration_seconds=info.duration,
                updated_at=time.time(),
            )
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                error = body.get("error")
                detail = error.get("message") if isinstance(error, dict) else body.get("detail")
            else:
                detail = None
            detail = detail or exc.response.text[:500]
        except ValueError:
            detail = exc.response.text[:500]
        error = f"TTS returned HTTP {exc.response.status_code}: {detail}"
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=error, updated_at=time.time())
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update(
                status="failed",
                error=f"TTS failed: {type(exc).__name__}: {exc}",
                updated_at=time.time(),
            )


def submit_tts_job(text: str, settings: Settings | None = None) -> str:
    config = settings or get_settings()
    job_id = uuid.uuid4().hex
    now = time.time()
    with _jobs_lock:
        # Keep only recent job state in memory; generated WAVs remain available.
        for old_id in [
            key for key, job in _jobs.items()
            if job["status"] in {"complete", "failed"} and now - job["updated_at"] > 3600
        ]:
            _jobs.pop(old_id, None)
        _jobs[job_id] = {"status": "queued", "updated_at": now}
    _executor.submit(_run_job, job_id, text, config)
    return job_id


def get_tts_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None
