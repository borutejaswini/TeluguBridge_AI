# IndicMT Telugu → English service

Standalone CPU-only translation microservice for Phase 3. It uses `facebook/nllb-200-distilled-600M`, Telugu source code `tel_Telu`, and English target code `eng_Latn`. Its virtual environment is isolated at `indicmt_service\.venv`; do not install its packages into `backend\.venv`.

## Setup and run

1. Run `setup.bat` from this directory with Python 3.12 available.
2. Configure `.env` from `.env.example` if needed.
3. Run `start.bat`. First startup downloads and loads NLLB once on CPU.
4. Check `http://127.0.0.1:8001/health`.

Translate with `POST http://127.0.0.1:8001/translate` and JSON body `{"text":"తెలుగు వాక్యం"}`. The response includes translation, model, device, language codes, and latency. NLLB runs direct tokenizer/model generation with an English forced beginning token. Numeric quantities and recognized units are protected and restored verbatim; if NLLB drops a protected marker, the service returns a structured integrity error instead of silently changing the quantity.

Run the real model check inside this service environment with `python scripts\test_translation.py`. Run mock-based unit tests with `.venv\Scripts\python.exe -m pytest tests`. Tests do not download or fabricate model files.
