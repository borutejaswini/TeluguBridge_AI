# TeluguBridge AI

TeluguBridge AI is an agriculture-focused Telugu voice assistant. Its intended path is:

```text
Telugu recording
  → Whisper Large V3 Turbo + Chunk-01 LoRA
  → NLLB Telugu-to-English query
  → BGE-M3 embedding + FAISS top-10
  → BGE reranker top-5
  → grounded Gemini answer written directly in Telugu
  → Indic Parler-TTS WAV
```

## Current verification state

Verified components include Phases 1–7, a real `final.wav` Whisper + LoRA run, and a separate real Indic Parler-TTS synthesis. Gemini/RAG and the full voice-to-audio path have not passed end-to-end verification. The current 11.71 GiB system ran out of available RAM/pagefile when all CPU models were resident; TTS loaded and synthesized correctly after the other model services were stopped. A real Gemini call was blocked by the automatic approval review because it would send the query and retrieved `paddy.pdf` text to Google. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for exact results and timings.

## Models and environments

| Component | Model | Environment/device |
|---|---|---|
| Speech input | `openai/whisper-large-v3-turbo` + `models/after_chunk_01/` | `backend/.venv`, Python 3.12.10, PyTorch 2.11.0+cu130, CUDA / RTX 2050 |
| Telugu → English | `facebook/nllb-200-distilled-600M` | `indicmt_service/.venv`, CPU; `tel_Telu` → `eng_Latn` |
| Embedding | `BAAI/bge-m3` | `embedding_service/.venv`, CPU, 1,024 dimensions |
| Vector search | FAISS `IndexFlatIP` | `vector_store/.venv`, CPU |
| Reranking | `BAAI/bge-reranker-v2-m3` | `embedding_service/.venv`, CPU |
| Answer generation | `gemini-3.8-flash` | External Google API; key stays on the backend |
| Speech output | `ai4bharat/indic-parler-tts` | `tts_service/.venv`, CPU, separate from Whisper |

IndicTrans2 is not part of the project pipeline. Gemini generates the final answer directly in Telugu.

## Requirements and setup

- Python 3.12.10 and Node.js (the frontend was built with Node 24.11.0).
- Keep the verified Whisper environment in `backend/.venv` unchanged. The TTS service has its own environment because the official Parler-TTS package requires Transformers 4.46.1; NLLB, BGE-M3, and FAISS also remain isolated.
- Configure the root `.env` from `.env.example`. Set `GEMINI_API_KEY` and `GEMINI_MODEL=gemini-3.8-flash`. An optional `HF_TOKEN` is read locally for Hugging Face model access. Do not put credentials in source, frontend files, or logs. `.env` is ignored by Git.
- Keep Hugging Face cache variables empty to use the user’s normal cache, or configure valid local paths in `.env`. Model files do not belong under `models/after_chunk_01/`; that folder contains only the custom LoRA.
- Install frontend dependencies from `frontend/` with `npm ci` after setup. Environment setup scripts are in `indicmt_service/`, `embedding_service/`, `vector_store/`, `ingestion_service/`, and `tts_service/`. Do not recreate or change the verified Whisper/PyTorch environment as part of other service setup.

## Services and ports

| Service | Port | Health/route |
|---|---:|---|
| Backend API | 8000 | `GET /health` |
| NLLB | 8001 | `GET /health`, `POST /translate` |
| Indic Parler-TTS | 8002 | `GET /health`, `POST /synthesize` |
| BGE reranker | 8003 | `GET /health`, `POST /rerank` |
| FAISS retrieval | 8004 | `GET /health`, `POST /retrieve` |
| BGE-M3 | 8005 | `GET /health`, `POST /embed` |
| React frontend | 5173 | Vite page |

BGE-M3 was moved to port 8005 because it previously occupied the required TTS port 8002. Its URL and retrieval client configuration were updated with it.

## Start and stop

From the project folder:

```bat
start.bat
```

Startup checks model readiness and writes logs under `data/logs/`. It reports incomplete startup with a nonzero exit status. It starts BGE-M3, NLLB, reranker, FAISS retrieval, TTS, backend, and frontend. The current machine may not have enough memory/pagefile to keep all CPU models loaded together; the current full stack is therefore not marked ready.

Stop only the project listeners:

```bat
stop.bat
```

Shutdown checks each configured port and its expected project command before stopping it. It does not terminate unrelated Python or Node processes.

## API and frontend

- `POST http://127.0.0.1:8000/api/rag/query` accepts `{"query":"తెలుగు ప్రశ్న"}` and returns the Telugu answer, translated query, retrieved evidence and source/page/scores, timing data, and status.
- `POST http://127.0.0.1:8000/api/pipeline/run` accepts multipart form data with an audio field named `file`; it returns the transcript, translation, answer, source information, timings, and generated audio path.
- `GET http://127.0.0.1:8000/api/audio/{filename}` serves the verified generated WAV. TTS output is stored in `outputs/audio/`.
- The React page at `http://127.0.0.1:5173` uploads audio, shows the transcript and answer/source details, and plays returned audio. The Gemini key is never sent to the browser.

## Verify a real audio component

The supplied primary audio is `tests/audio/final.wav` (PCM16 WAV, 48 kHz, stereo, 5.077 seconds). `tests/audio/test_telugu.wav` is also present (PCM16 WAV, 48 kHz, stereo, 9.536 seconds). To run the real Whisper + LoRA check:

```bat
backend\.venv\Scripts\python.exe scripts\test_whisper_lora.py tests\audio\final.wav
```

The full endpoint is intended to run the complete path above. Do not treat a component smoke test, a mocked test, or an HTTP process check as a successful full pipeline. The current `final.wav` run produced a real Telugu Whisper transcript, but the subsequent NLLB output was semantically unreliable; Gemini and final combined audio remain unverified.

## Device placement and security

Whisper is the only model assigned to the RTX 2050. NLLB, BGE-M3, FAISS, reranker, and Parler-TTS are CPU-only. Do not load all heavy models on the 4 GiB GPU.

The real Gemini key belongs only in root `.env`; it must not appear in Git-tracked files, Python/React code, tests, logs, or API responses. The backend alone calls Gemini. Retrieved PDF text is sent to the configured external model only when a Gemini request is authorized and made.
