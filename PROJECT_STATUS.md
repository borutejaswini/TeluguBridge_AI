# Project Status

Updated: 2026-09-27

## Phase 1 — Foundation

- [x] Project structure and configuration foundation
- [x] `.env.example`, `.gitignore`, requirements split, README
- [x] Structured errors and logging setup
- [x] FastAPI health endpoint
- [x] Windows setup/start/stop scripts
- [x] Test scaffold
- [x] Phase 1 dependencies installed in local Python 3.13 virtual environment
- [x] Foundation tests: 2 passed
- [x] API health smoke check: HTTP 200

Status: passed in Python 3.13.2. Target Python 3.12 is unavailable on this machine.

## Phase 2 — Whisper standalone service

- [x] Lazy, reusable Transformers Whisper service
- [x] 16 kHz mono decode/resample path
- [x] 444 token generation cap and Telugu transcription task
- [x] Optional PEFT LoRA with required-file validation and no silent fallback
- [x] Configurable HF cache/auth settings and structured load errors
- [x] Standalone benchmark and LoRA benchmark scripts
- [x] Unit coverage for sample rate, token cap, and missing adapter behavior
- [x] Whisper unit/foundation tests: 5 passed
- [x] Real model/audio smoke test and GPU memory measurement (see successful run below)

Status: Phase 2 passed after the actual CUDA Whisper + Chunk-01 LoRA smoke test completed and returned a Telugu transcription. The earlier environment/cache/dtype failures below are retained as troubleshooting history.

### Environment reconciliation and real smoke test — 2026-09-26

- [x] LoRA directory detected at `models/after_chunk_01/`
- [x] `adapter_config.json` detected (1,207 bytes)
- [x] `adapter_model.safetensors` detected (13,129,824 bytes)
- [x] Recording detected at `tests/audio/test_telugu.wav` (1,830,990 bytes)
- [x] WAV decoded for metadata: 48,000 Hz, stereo, 457,728 frames, 9.536 seconds
- [x] NVIDIA driver detects NVIDIA GeForce RTX 2050 with 4,096 MiB VRAM; driver reports CUDA 13.0
- [x] `backend/.venv` Python: 3.12.10
- [x] PyTorch: 2.11.0+cu130
- [x] `torch.cuda.is_available()`: True
- [x] PyTorch CUDA version: 13.0
- [x] GPU: NVIDIA GeForce RTX 2050; PyTorch reports 4,294,443,008 bytes VRAM (~4 GiB)
- [x] Transformers: 5.17.0
- [x] PEFT: 0.21.0
- [x] Accelerate: 1.15.0
- [x] librosa: 0.11.0
- [x] soundfile: 0.14.0
- [x] NumPy: 2.5.3
- [x] Base backend packages installed: FastAPI 0.141.1, Uvicorn 0.54.0, pydantic-settings 2.15.0, httpx 0.28.1
- [x] Foundation/Whisper unit checks: 5 passed using workspace-local pytest temporary storage
- [x] Phase 2 benchmark now reports runtime, device/model, input and processed sample rates, audio duration, transcription, timing, RTF, and peak GPU memory
- [x] Real Whisper + Chunk-01 LoRA inference: generation completed and returned a Telugu transcription

Cache configuration resolution:

- No project `.env` exists; `.env.example` leaves `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` empty.
- The stale `HF_HOME=D:\HF_Models` was present only in the execution process environment; User and Machine environment values were empty.
- Removed the process-level override for the smoke-test command. Hugging Face resolved its normal cache to `C:\Users\LOHITH\.cache\huggingface`; that directory and its `hub` subdirectory already existed.
- The exact Whisper model `openai/whisper-large-v3-turbo` downloaded and loaded from the default cache. Chunk-01 LoRA was loaded.

Exact inference failure:

```text
RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same
```

The recording had been decoded from 48,000 Hz stereo and resampled to 16,000 Hz mono. The fix reads the loaded model's first parameter dtype/device and converts only `input_features` to match; it logs and validates both tensor/model dtype and device immediately before `generate()`. CUDA remained enabled, and generation remained under `torch.inference_mode()` with `max_new_tokens=444`, Telugu, and transcribe settings. The benchmark also configures UTF-8 stdout so the Telugu result can be reported in Windows consoles.

Successful real smoke test, rerun with `scripts/test_whisper_lora.py tests/audio/test_telugu.wav`:

- Model: `openai/whisper-large-v3-turbo` — loaded from Hugging Face cache
- LoRA: Chunk-01 loaded (`models/after_chunk_01`)
- Python: 3.12.10
- PyTorch: 2.11.0+cu130
- CUDA: available; PyTorch CUDA 13.0
- GPU/device: NVIDIA GeForce RTX 2050, `cuda:0`
- Model dtype / input dtype: `torch.float16` / `torch.float16`
- Recording: 9.536 seconds, original 48,000 Hz stereo; processed 16,000 Hz mono
- Telugu transcription: `యవసాయం చేసి మనందరికి ఆహారాలని అందించే వెక్తిని రైతు అంటారు దేశానికి వెన్నమక రైతు రైతే రాజువని పెద్దలు అంటారు`
- Inference time: 5.147 seconds
- Real-time factor: 0.54
- Peak allocated GPU memory: 1,623.35 MiB
- Model load time: 26.53 seconds

Phase 2 is fully passed. No synthetic audio or model files were created. Phase 3 has not been started.

## Phase 3 — Telugu → English NLLB translation

- [x] Separate `indicmt_service/` source structure, API, configuration, setup/run scripts, and documentation created
- [x] Exact model/languages configured: `facebook/nllb-200-distilled-600M`, `tel_Telu` → `eng_Latn`
- [x] Service designed for one-time CPU model load at FastAPI startup and direct tokenizer/model generation
- [x] Numeric quantities and recognized units protected and restored with a structured integrity error if NLLB drops a marker
- [x] Separate `indicmt_service/.venv` created with Python 3.12.10, CPU PyTorch 2.11.0+cpu, Transformers 5.17.0, SentencePiece 0.2.2, FastAPI 0.141.1, and supporting runtime packages
- [x] CPU PyTorch minimal import probe: `2.11.0+cpu`
- [x] CPU PyTorch CUDA probe: `False` (expected for CPU wheel)
- [x] Mock unit/API tests: 6 passed
- [x] Real NLLB model download, load, and Telugu → English translation smoke test passed
- [x] Real FastAPI startup, `GET /health`, and `POST /translate` smoke test passed

### PyTorch CPU import investigation

The first pytest collection had failed with Windows fatal exception `0x8007000e` in `platform._wmi_query`, called by `platform.uname()` during PyTorch DLL initialization. This did not reproduce on the requested minimal probes: both `import torch; print(torch.__version__)` and `import torch; print(torch.cuda.is_available())` passed. A resource snapshot during investigation showed 11,996 MiB total physical memory, 1,485 MiB free physical memory, 33,297 MiB total virtual memory, and 5,086 MiB free virtual memory; no Python process was present during that snapshot. This is consistent with a transient Windows WMI/resource exhaustion during the earlier import, but the original transient condition could not be reproduced or attributed to a specific conflicting process. No package downgrade/change was made in response. The CPU PyTorch wheel remained `2.11.0+cpu`.

### Verification results

The real smoke test used `scripts/test_translation.py` with input `ఈ రోజు రైతు పత్తి పంటను పరిశీలిస్తున్నాడు.`:

- Model: `facebook/nllb-200-distilled-600M`, downloaded and loaded successfully
- Device: CPU (`torch 2.11.0+cpu`; CUDA unavailable as intended in this service)
- Input language: `tel_Telu`
- Output language: `eng_Latn`
- Cold model load including initial download: 915.446 seconds
- Translation latency on first generation: 11,703.67 ms
- English output: `Today the farmer is observing the cotton crop.`

The real FastAPI smoke test (`scripts/test_api_smoke.py`) also loaded the cached model, reported `/health` as `ok`, and returned HTTP 200 for `/translate`:

- Cached model load at service startup: 22.321 seconds
- Translation latency: 4,136.55 ms
- Output: `Today the farmer is observing the cotton crop.`

The mocked integrity test confirms numeric/unit spans such as `5 కిలోలు` are preserved exactly; if the model omits a protected span, the service returns `TRANSLATION_INTEGRITY_ERROR` instead of silently changing the quantity. Crop/disease terminology quality beyond the smoke phrase has not been broadly evaluated.

Phase 3 is fully verified. No IndicTrans2 or English → Telugu translation was implemented.

## Later phases

BGE-M3 dense embedding service (Phase 4) is implemented in `embedding_service/` with a separate Python 3.12.10 CPU environment, `BAAI/bge-m3`, FastAPI `/health` and `/embed`, one-time startup loading, normalized 1,024-dimensional outputs, structured errors, and mocked tests. Installed runtime: PyTorch 2.11.0+cpu, Transformers 5.17.0, Sentence Transformers 5.7.0.

### Phase 4 real verification — 2026-09-26

- [x] Inspected only the BGE-M3 Hugging Face cache; C: had 79.85 GiB free before the retry.
- [x] Downloaded the real BGE-M3 checkpoint to the normal Hugging Face cache; cached `pytorch_model.bin` is 2,271,145,830 bytes. Removed only two stale `.incomplete` artifacts under `models--BAAI--bge-m3`; Whisper and NLLB caches were not touched.
- [x] Python: 3.12.10; PyTorch: 2.11.0+cpu; Transformers: 5.17.0; Sentence Transformers: 5.7.0.
- [x] Real BGE-M3 model loaded and inference ran on CPU.
- [x] Real Telugu agriculture input: `పత్తి పంటలో ఆకుమచ్చ తెగులు కనిపిస్తే ఎకరానికి 2 లీటర్ల మందు వేయాలా?`
- [x] Dense embedding dimension: 1,024; vector norm: 1.0.
- [x] Cached model load time: 37.905 seconds; embedding latency: 2,609.00 ms; end-to-end smoke runtime: 40.516 seconds.
- [x] Existing BGE-M3 unit and API tests: 5 passed (one upstream Starlette/httpx deprecation warning).

The initial real inference attempt exposed that Sentence Transformers 5.7.0 does not accept `max_length` in `encode()`. The service now applies the configured length through `model.max_seq_length`, and a rerun produced a real embedding. The smoke script also uses UTF-8 stdout for Telugu output. Phase 4 is fully verified.

## Phase 5 — Agriculture PDF/OCR ingestion

### Verification — 2026-09-26

- [x] Ran the Phase 5 unit suite from `ingestion_service/`: `18 passed` using `pytest -q -p no:cacheprovider tests --basetemp .pytest-tmp`.
- [x] Processed the real, user-provided `knowledge_base/paddy.pdf`; no synthetic PDF or content was used.
- [x] Discovery: 1 PDF; source filename/path: `paddy.pdf`; source document SHA-256/document ID: `d1d3a9a2be5898c7236631d42550028f5acbea306e4311cdb26c7953881a58ea`.
- [x] Source has 7 pages and 13,691 extracted text characters. Every page was classified `KNOWLEDGE` and extracted with PyMuPDF native `text`; OCR was not used on any page.
- [x] Per-page extracted character counts: p.1 2,302; p.2 2,086; p.3 2,020; p.4 1,718; p.5 1,820; p.6 2,336; p.7 1,409.
- [x] Produced 22 chunks with metadata for chunk/document ID, source filename and relative path, page number, section, extraction method, OCR usage, classification, and text. IDs are unique and deterministic.
- [x] Parent sections persist across page breaks: `Brown Plant Hopper (BPH)` continues onto p.2; `Cultural IPM Practices` continues onto p.6; `Disease Record 1: Sheath Blight` continues onto p.7. Crop pest/disease names, contextual text, and pesticide quantities/dosages are retained in extracted chunks.
- [x] Repeated ingestion with identical inputs and settings. JSONL and manifest SHA-256 hashes were byte-for-byte identical across the two runs.
- [x] Output JSONL: `data/processed_documents/d1d3a9a2be5898c7236631d42550028f5acbea306e4311cdb26c7953881a58ea.jsonl` (SHA-256 `0FD22F46DAF26732CF06708535D9D46D8C3AE01DA550B66E7431AF8A51B35B5E`).
- [x] Output manifest: `data/processed_documents/d1d3a9a2be5898c7236631d42550028f5acbea306e4311cdb26c7953881a58ea.manifest.json` (SHA-256 `0ABE4AEE554BE21ADC18505CD12CBF9F7454FF6860E411B9D9DFC9829ED27F06`).

Section/chunk audit (page → section → chunk ID):

| Page | Section | Chunk ID |
|---|---|---|
| 1 | Stem Borer | `d1d3a9a2be58-p0001-c001-fd80058a1a30fddffaf28f79` |
| 1 | Brown Plant Hopper (BPH) | `d1d3a9a2be58-p0001-c002-d80186354e5612efd9d626a3` |
| 2 | Brown Plant Hopper (BPH), continuation | `d1d3a9a2be58-p0002-c001-e2c4264c932e964d11cc11a5` |
| 2 | Pest Record 1: Gall Midge | `d1d3a9a2be58-p0002-c002-ed63270f3729e457cd3bb386` |
| 3 | Pest Record 2: Rice Hispa | `d1d3a9a2be58-p0003-c001-e57793ec46112119aafd04d8` |
| 3 | Pest Record 3: Whorl Maggot / Rice Fly | `d1d3a9a2be58-p0003-c002-75ab9e2c6cc056edb002ce9c` |
| 3 | Pest Record 4: Leaf Folder | `d1d3a9a2be58-p0003-c003-5fa4f39e9a286e3dd461c383` |
| 4 | Pest Record 5: Thrips | `d1d3a9a2be58-p0004-c001-716435e816e7e2011a11f2a5` |
| 4 | Pest Record 6: Leaf Mites (Red Spider Mite) | `d1d3a9a2be58-p0004-c002-bf4dafca3d7b2a2ea72321c4` |
| 4 | Pest Record 7: Panicle Mite | `d1d3a9a2be58-p0004-c003-3d3cb5a6aea0a7b3db610aac` |
| 4 | Pest Record 8: Cutworm / Leaf-Cutting Caterpillar | `d1d3a9a2be58-p0004-c004-af2b613b77f599e5a94177ea` |
| 5 | Pest Record 8: Cutworm / Leaf-Cutting Caterpillar, continuation | `d1d3a9a2be58-p0005-c001-30ea11ada6e2eb10c78a13b5` |
| 5 | Pest Record 9: Armyworm / Head-Cutting Cutworm | `d1d3a9a2be58-p0005-c002-6228dde3e0bdafdae7e12ced` |
| 5 | Pest Record 10: Rice Earhead Bug (Gundhi Bug) | `d1d3a9a2be58-p0005-c003-6592c57cc6cbc5f993eb372c` |
| 5 | 3. General Pesticide & Granule Spray Guidelines | `d1d3a9a2be58-p0005-c004-7ed00945c2330eabfef44696` |
| 5 | Cultural IPM Practices | `d1d3a9a2be58-p0005-c005-5ab803e66090b9bcc102d726` |
| 6 | Cultural IPM Practices, continuation | `d1d3a9a2be58-p0006-c001-426fc1df4f86a3edd4fb35a5` |
| 6 | Crop Resistant Varieties Reference | `d1d3a9a2be58-p0006-c002-a73400995947790e611be211` |
| 6 | Rice Blast | `d1d3a9a2be58-p0006-c003-9310cf3e3f0280b25935e37f` |
| 6 | Disease Record 1: Sheath Blight | `d1d3a9a2be58-p0006-c004-f68b630511289552a0a3ace5` |
| 7 | Disease Record 1: Sheath Blight, continuation | `d1d3a9a2be58-p0007-c001-ce1f44e021230f98c1012c59` |
| 7 | Disease Record 2: Bacterial Leaf Blight (BLB) | `d1d3a9a2be58-p0007-c002-6b654cfa5f9c5762017a247f` |

OCR limitation: all pages in this real PDF had sufficient embedded text, so OCR was not needed or exercised. Tesseract is not installed/configured in this environment; scanned/image-only PDFs that require OCR need Tesseract and the corresponding OCR language data installed/configured. This did not affect `paddy.pdf` or its output.

Phase 5 is passed for the real `paddy.pdf` ingestion and the tested native-text path.

## Phase 6 — BGE-M3 + FAISS vector retrieval

### Verification — 2026-09-26

- [x] Added a separate `vector_store/` package for configuration, BGE service client, metadata normalization/validation, FAISS persistence, retrieval, build command, and real smoke checks.
- [x] Reused the existing Phase 4 `BAAI/bge-m3` service through its `/embed` endpoint. Model loaded once in the existing service process and remained on CPU; no BGE, Whisper, or NLLB source/environment was changed.
- [x] FAISS CPU dependency is pinned to `faiss-cpu==1.13.2` in the new isolated `vector_store/.venv` (Python 3.12.10; NumPy 2.5.3). The existing embedding environment stayed at PyTorch 2.11.0+cpu, Transformers 5.17.0, Sentence Transformers 5.7.0.
- [x] Built from the real Phase 5 JSONL with actual BGE-M3 inference: model `BAAI/bge-m3`, device CPU, embedding dimension 1,024, normalized float32 vectors.
- [x] Persistent index type: exact `IndexFlatIP` (inner product equals cosine similarity for normalized vectors); 22 vectors; 22 metadata rows; 22 unique chunk IDs.
- [x] Reloaded the persisted index and metadata successfully. Verified `index.ntotal == len(metadata) == 22`, dimension 1,024, exact source JSONL metadata ordering, and vector norms in `[0.99999988, 1.00000012]`.
- [x] Metadata retains chunk/document IDs, source filename/relative path, page, section, extraction method, OCR flag, classification, and full chunk text. Loader verifies schema, checksums, dimension, metric/index type, count, unique IDs, and stored-vector normalization.
- [x] Rebuilt from identical source/configuration. `index.faiss`, `metadata.json`, and `store_manifest.json` had identical SHA-256 hashes across the builds.
- [x] Phase 6 unit and real integration tests: **13 passed** (9 FAISS/validation unit tests and 4 real BGE-M3 retrieval checks). FAISS emitted three upstream SWIG deprecation warnings; no test failures.
- [x] Regression tests: existing BGE-M3 tests **5 passed** (one existing Starlette/httpx deprecation warning); Phase 5 ingestion tests **18 passed**.
- [x] Actual BGE-M3 service startup reported a 33.257-second model load. One actual 22-chunk embedding request took 33.732 seconds.

Persisted files:

- FAISS index: `data/vector_store/index.faiss` (SHA-256 `ef18680604e8e4219488aa4f706e7b28086245993c42edb70d0af9ef376d0349`).
- Ordered metadata: `data/vector_store/metadata.json` (SHA-256 `7297fc10510c67804aa028c1d65c20344a58e4fb0466db3a53294e114888164e`).
- Manifest: `data/vector_store/store_manifest.json` (SHA-256 `332a37e92947729d18cc17a0c36dc11f948d7e3c69a4c3ebd8127184be1974cf`).
- Manifest records source JSONL SHA-256 `0fd22f46daf26732cf06708535d9d46d8c3ae01da550b66e7431af8a51b35b5e`, document ID, source ordering, model, dimension, normalization, index type, counts, and metadata/index checksums.

Real English queries used `top_k=10`; the following are actual top-ranked results from the persistent store (text excerpts are from returned chunks):

| Query | Chunk ID | Score | Source | Retrieved text excerpt |
|---|---|---:|---|---|
| What are the symptoms of stem borer in rice? | `d1d3a9a2be58-p0001-c001-fd80058a1a30fddffaf28f79` | 0.706224 | p.1 · Stem Borer | “Dark brown, straw, or yellowish female moths have a distinct black spot… Egg masses… appear as small brown clumps on leaf tips or stems.” |
| What are the management practices for brown plant hopper? | `d1d3a9a2be58-p0001-c002-d80186354e5612efd9d626a3` | 0.603851 | p.1 · Brown Plant Hopper (BPH) | “Nymphs and adult hoppers remain at the base of tillers near the water surface to suck sap, turning the crop light yellow.” |
| same query | `d1d3a9a2be58-p0002-c001-e2c4264c932e964d11cc11a5` | 0.533548 | p.2 · Brown Plant Hopper (BPH) | Returned continuation includes BPH favorable factors and management details. |
| What pesticide dosage information is given for rice pests? | `d1d3a9a2be58-p0003-c001-e57793ec46112119aafd04d8` | 0.659756 | p.3 · Pest Record 2: Rice Hispa | Rice Hispa pest record with ETL and recommended pesticide dosage content. |
| same query | `d1d3a9a2be58-p0005-c004-7ed00945c2330eabfef44696` | 0.605690 | p.5 · General Pesticide & Granule Spray Guidelines | “Mix recommended granules with 25 kg of dry sand per acre… All liquid spray dosages are specified per liter of water… 150 to 200 liters… per acre.” |
| What are the diseases affecting rice? | `d1d3a9a2be58-p0006-c003-9310cf3e3f0280b25935e37f` | 0.585202 | p.6 · Rice Blast | “Fungal pathogen *Pyricularia oryzae*… Spindle-shaped spots form on leaves… margins turn dark brown or black…” |

For the diseases query, the top 10 also included Sheath Blight (p.6) and Bacterial Leaf Blight (p.7). The top-K tests check that relevant sections appear somewhere in the results, without requiring a fixed rank-1 chunk.

The existing process-level `HF_HOME=D:\HF_Models` override initially prevented the BGE service from starting because `D:` was unavailable. User/Machine environment settings were empty. The service was launched with that process-only override cleared and reused the existing Hugging Face cache; no project or system cache configuration was changed.

Phase 6 is **PASSED**. The Phase 2 Whisper + LoRA smoke test is verified as above. Full pipeline/end-to-end behavior remains unverified.

## Phase 7 — Retrieval Reranking

### Recovery verification — 2026-09-27

- [x] Reused the existing `embedding_service/.venv`: Python 3.12.10, PyTorch 2.11.0+cpu, Transformers 5.17.0, Sentence Transformers 5.7.0, and huggingface_hub 1.33.0. No package versions changed.
- [x] Kept `BAAI/bge-reranker-v2-m3` on CPU; no changes to Whisper, NLLB, BGE-M3, Phase 5 data, or the Phase 6 FAISS index/metadata.
- [x] Downloaded the real model with `huggingface_hub.snapshot_download` into the normal user cache. Snapshot: `C:/Users/LOHITH/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. `model.safetensors`: **2,271,071,852 bytes**, matching Hugging Face metadata.
- [x] Root cause of earlier incomplete downloads: the default 10-second Hub file-read timeout expired during the large weight transfer; subsequent retries encountered transient DNS resolution failures (`[Errno 11001] getaddrinfo failed`). Retried from the existing cache with process-only `HF_HUB_DOWNLOAD_TIMEOUT=120` and `HF_HUB_DISABLE_XET=1`. The inherited stale `HF_HOME=D:\HF_Models` was not used; the service launcher and download process cleared it, resolving the cache to the normal C: user cache. No global environment variables were changed.
- [x] Loaded the actual CrossEncoder through the reranker service loader on CPU. Direct load: **57.578 s**, followed by actual query/document scoring (**7,361.28 ms**, score `0.9443972`). Service startup load: **39.079 s**.
- [x] `GET http://127.0.0.1:8003/health`: HTTP 200, `loaded=true`, expected model, CPU. Real `POST /rerank` returned scores `[0.9299711, 0.0381207]` for two agriculture candidates in **2,014.93 ms**.
- [x] Ran five real paddy.pdf queries through existing BGE-M3 → FAISS top-10 → reranker → top-5. All returned 10 FAISS candidates and five final results. Full real comparison, scores, ranks, timings, and source metadata: `data/vector_store/reranking_smoke_report.json`.
- [x] Actual metadata fields remain attached to reranked records: chunk/document ID, source filename/path, page, section, extraction method, classification, text, FAISS score, and reranker score. Five real integration tests verify IDs, source/page/section, preserved FAISS scores, reranker ordering, and timings.
- [x] FAISS-only versus reranked example for “What are the symptoms of stem borer in rice?”: FAISS top five were Stem Borer, Rice Blast, Rice Earhead Bug, Rice Hispa, Brown Plant Hopper. Reranked top five were Stem Borer, Rice Blast, Brown Plant Hopper, Rice Hispa, Whorl Maggot/Rice Fly. Brown Plant Hopper moved from FAISS rank 5 to reranker rank 3; Rice Earhead Bug left the top five. No claim is made that reranking always improves relevance.
- [x] Five-query mean latency: query embedding **724.82 ms**, FAISS search **1.64 ms**, reranker **19,058.83 ms**, total **19,785.29 ms**. Timings are per-query and exclude model startup.
- [x] Phase 6 + Phase 7 unit and real integration tests: **24 passed**, three upstream FAISS SWIG deprecation warnings.
- [x] Phase 5 regressions: **18 passed**; reranker service unit/API: **6 passed**; BGE-M3: **5 passed**; Whisper/foundation: **5 passed**; NLLB: **6 passed**. Starlette/httpx emitted its existing TestClient deprecation warning in the relevant service suites.

Status: **PASSED**. Real model loading, CPU inference, HTTP readiness/scoring, paddy.pdf top-10→top-5 reranking, metadata, score ordering, timing, and regression conditions were verified.

## Phase 8 — Grounded Gemini Generation

Status: **IMPLEMENTED; REAL GEMINI VERIFICATION BLOCKED**.

- Added `POST /api/rag/query`: Telugu → English NLLB, local BGE-M3/FAISS top-10 and BGE reranker top-5, then a grounded Telugu Gemini request using the configured `gemini-3.8-flash` model.
- Retrieved chunks are serialized as evidence data; the system instruction says to treat them as data, ignore instructions inside the evidence, use only evidence for factual agricultural claims, and refuse unsupported answers in Telugu.
- Response includes original/translated query, answer, retrieved text, source/page metadata, FAISS and reranker scores, per-stage timings, and status. Weak retrieval evidence returns a fixed safe Telugu insufficiency response without calling Gemini.
- `GEMINI_API_KEY` is read from root `.env`, is not logged or returned, and `.env` is ignored by Git. A source/log scan found no copy of its configured value elsewhere.
- Mock/service-contract and structured API validation tests pass (see final test summary below).
- **No real Gemini request was sent.** Automatic approval review rejected the attempted call because it would send the Telugu query and retrieved `paddy.pdf` text to Google’s external API, and said authorization to implement the integration did not clearly authorize exporting that local derived data. No Gemini answer or grounding result is claimed.

## TTS, backend orchestration, frontend, and startup — 2026-09-27

### Indic Parler-TTS

- Added an isolated `tts_service/.venv` using Python 3.12.10, CPU PyTorch 2.11.0, Transformers 4.46.1 (the upstream Parler-TTS compatibility requirement), and the official Parler-TTS package. The verified Whisper environment was not changed.
- Added a one-time CPU model loader, `/health`, `/synthesize`, and audio retrieval route. WAVs are written under `outputs/audio/` and checked for readable WAV format, sample rate, channels, frames, and positive duration.
- **Standalone real inference PASSED.** `ai4bharat/indic-parler-tts` loaded on CPU in **39.95 s**; real Telugu synthesis took **144,676.52 ms** and produced [outputs/audio/telugu-2dc5b28f5498482f98df45762664ade6.wav](outputs/audio/telugu-2dc5b28f5498482f98df45762664ade6.wav) — 44,100 Hz, mono, 7.581 seconds, 668,716 bytes. The service health endpoint reported `loaded=true`; the generated bytes were fetched and reopened as WAV.
- **Combined all-service startup is BLOCKED by system memory/pagefile capacity.** With BGE-M3, NLLB, and the reranker resident, TTS model weight loading failed with Windows `OSError: The paging file is too small for this operation to complete. (os error 1455)`. At that point physical memory was 11.71 GiB total with about 0.4 GiB free. Later, with free memory falling to 0.05 GiB, the reranker connection reset and the retrieval process exited. The same TTS model worked after shutting down the other CPU models.

### Final recording — actual component runs

- `tests/audio/final.wav`: readable PCM16 WAV, stereo, 48,000 Hz, 5.077 seconds, 974,922 bytes. `tests/audio/test_telugu.wav`: readable PCM16 WAV, stereo, 48,000 Hz, 9.536 seconds, 1,830,990 bytes.
- Ran the real `scripts/test_whisper_lora.py tests/audio/final.wav`. **Whisper + Chunk-01 LoRA PASSED:** Python 3.12.10; PyTorch 2.11.0+cu130; CUDA 13.0; NVIDIA GeForce RTX 2050; model `openai/whisper-large-v3-turbo`; adapter loaded; CUDA `float16` model and input; model load 110.886 s; inference 3.225 s; RTF 0.635; peak GPU memory 1,623.35 MiB.
- Actual transcript: `వరిశేనులో ఆకు ముడత పురుగుని ఎలాని వారించాది`.
- Ran actual NLLB CPU inference on that transcript: model load 30.893 s, translation 7,861.63 ms, languages `tel_Telu` → `eng_Latn`. Output: `In the valley of the shadow of death, a leaf-shaped worm was planted.` This translation is semantically unreliable and is recorded as returned; it was not corrected or presented as a valid agriculture query.
- No complete final-audio request ran through BGE-M3 → FAISS → reranker → Gemini → Parler-TTS. A local retrieval attempt hit `RERANKER_INFERENCE_FAILED` with `[WinError 10054] An existing connection was forcibly closed by the remote host` while system memory was exhausted. The Gemini outbound request was separately blocked by approval review. The standalone TTS sample above is not output from `final.wav` or from Gemini.

### Backend, frontend, services, and tests

- Backend now exposes `/api/rag/query`, `/api/pipeline/run` (audio upload through Whisper → retrieval/RAG → TTS), and `/api/audio/{filename}`. Errors identify the failing stage and return structured JSON. Whisper remains lazy-loaded once and on CUDA; other model services use CPU.
- Added a minimal React/Vite upload, status, transcript, answer, source, timing, and audio playback interface. `npm install` completed with zero reported vulnerabilities; `npm run build` passed. The clean-start attempt served the frontend successfully at `http://127.0.0.1:5173/` after Vite initialized.
- Resolved the real 8002 port conflict by keeping TTS at 8002 and moving BGE-M3 to 8005; added FAISS retrieval on 8004. The other requested ports remain backend 8000, NLLB 8001, reranker 8003, frontend 5173. Service URLs and the example env file were updated consistently.
- `start.bat` attempted a clean service startup. BGE-M3, NLLB, reranker, retrieval, backend, and Vite each reached readiness at some point, but the combined stack did not remain healthy under current memory limits and TTS could not load alongside the other CPU models. The script returned a blocked/nonzero status instead of claiming success.
- `stop.bat` was exercised and stopped only listeners whose configured port and command line matched this project. All project listeners were shut down afterward.
- The Gemini and pipeline unit/API tests use mocks only for unit boundaries; no mocked response is counted as a real model result.

### Final test summary

| Group | Result |
|---|---|
| Backend, Whisper/foundation, RAG and API validation | **10 passed** (one upstream Starlette/httpx deprecation warning) |
| NLLB unit/API | **6 passed** (one upstream deprecation warning) |
| BGE-M3 unit/API | **5 passed** (one upstream deprecation warning) |
| Reranker service unit/API | **6 passed** (one upstream deprecation warning) |
| Phase 5 ingestion | **18 passed** |
| Vector-store/FAISS unit and HTTP contract | **16 passed, 9 skipped** (real-service cases require resident services) |
| TTS validation unit tests | **2 passed** |
| Frontend production build | **PASSED** |
| Real final.wav Whisper + LoRA | **PASSED** |
| Real final-transcript NLLB | **RAN; translation semantically unreliable** |
| Real standalone Indic Parler-TTS synthesis | **PASSED** |
| Real Gemini positive/negative grounding calls | **BLOCKED before request by automatic approval review** |
| Clean all-service startup / complete final.wav pipeline | **BLOCKED by Windows paging-file error 1455 and low available RAM** |

### Current overall state

Phase 7 remains **PASSED** based on its earlier real model verification. Gemini RAG and orchestration are implemented but not fully verified; TTS is verified as a standalone service. The end-to-end product is **BLOCKED**, not passed. The actual final WAV reached Whisper and NLLB, but the NLLB output was unreliable, the concurrent CPU model stack exhausted available memory, and no external Gemini request was made. No complete Telugu answer audio from `final.wav` exists.
