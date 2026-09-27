# Phase 6 vector store

This package builds and queries an exact, persistent FAISS `IndexFlatIP` from Phase 5 knowledge chunks. It uses normalized 1,024-dimensional dense vectors from the existing `BAAI/bge-m3` CPU embedding service. It does not contain or reload the embedding model.

## Setup

Run `setup.bat` in this folder to create the isolated Python 3.12 FAISS environment. Start the existing BGE-M3 service separately from `embedding_service/` and wait for `/health` to report `loaded: true`. Its existing process loads the model once and serves both indexing and later queries.

## Build and retrieve

From the repository root, run:

```powershell
vector_store\.venv\Scripts\python.exe -m vector_store.build
vector_store\.venv\Scripts\python.exe -m vector_store.smoke_retrieval
```

The default source is the verified Phase 5 JSONL. Defaults can be overridden with `VECTOR_STORE_SOURCE_JSONL`, `VECTOR_STORE_DIR`, `BGE_EMBEDDING_URL`, and `INITIAL_TOP_K`, or with command-line options. The persistent store is written to `data/vector_store/` as `index.faiss`, `metadata.json`, and `store_manifest.json`.

The FAISS ID at position *i* always maps to metadata record *i*. Metadata records remain in the source JSONL order after filtering to `KNOWLEDGE`; no random shuffle or approximate index is used. The loader checks file presence, metadata JSON/schema, checksum, unique IDs, exact vector/metadata count, dimension, inner-product index type, and normalized stored vectors before search.

The BGE service must be running at build/retrieval time. A missing service or failed model call produces a structured `EMBEDDING_FAILURE`; FAISS and metadata errors are also returned as structured errors.

