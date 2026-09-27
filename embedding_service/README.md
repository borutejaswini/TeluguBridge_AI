# BGE-M3 embedding service

Phase 4 uses the official `BAAI/bge-m3` model for normalized 1,024-dimensional dense embeddings. The service runs on CPU so the RTX 2050 remains available to Whisper. Its Python environment is isolated from `backend/.venv` and `indicmt_service/.venv`.

Create the environment from this directory with `setup.bat`, copy `.env.example` to `.env` to customize settings, and start it with `start.bat`. Model weights use the normal Hugging Face user cache unless the user's environment configures another cache; weights are not stored under this service or project model directories.

The model loads once during API startup. `GET /health` reports readiness and load time. `POST /embed` accepts `{"texts": ["..."]}` and returns dense embeddings, model, device, dimension, and latency. Each batch is capped at 32 items; input tokenization is limited to 8,192 tokens per text.

Run mocked tests with `..\\indicmt_service\\.venv\\Scripts\\python.exe -m pytest tests` after installing the development requirements, or use this environment's Python. Run a real model smoke test with `python scripts/test_embedding.py` after dependencies and model weights are available.

