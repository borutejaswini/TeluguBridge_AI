# CPU BGE reranker

Runs the configured `BAAI/bge-reranker-v2-m3` CrossEncoder on CPU. It uses the existing `embedding_service/.venv` ML runtime and the normal Hugging Face cache; no reranker dependency is installed into the Whisper environment. The model loads once at process startup and is reused by every request.

From this folder, run `start.bat`; `/health` reports model readiness/load time and `POST /rerank` scores query/candidate pairs independently. The service returns one sigmoid-scaled relevance score per candidate plus inference latency. Reranking failures are structured and never trigger an implicit FAISS-only fallback.

