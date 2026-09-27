from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed_documents" / (
    "d1d3a9a2be5898c7236631d42550028f5acbea306e4311cdb26c7953881a58ea.jsonl"
)
DEFAULT_STORE = PROJECT_ROOT / "data" / "vector_store"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_project_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {
                "VECTOR_STORE_SOURCE_JSONL", "VECTOR_STORE_DIR", "BGE_EMBEDDING_URL", "INITIAL_TOP_K",
                "TOP_K", "RERANKER_ENABLED", "RERANKER_MODEL_ID", "RERANKER_DEVICE", "RERANKER_URL",
            }:
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except OSError:
        return


@dataclass(frozen=True)
class Settings:
    source_jsonl: Path = DEFAULT_SOURCE
    store_dir: Path = DEFAULT_STORE
    embedding_url: str = "http://127.0.0.1:8005"
    reranker_url: str = "http://127.0.0.1:8003"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    normalized: bool = True
    initial_top_k: int = 10
    final_top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_enabled: bool = True
    request_timeout_seconds: float = 600.0

    @classmethod
    def from_env(cls) -> "Settings":
        _load_project_env()
        top_k = int(os.getenv("INITIAL_TOP_K", "10"))
        final_top_k = int(os.getenv("TOP_K", "5"))
        if top_k < 1 or final_top_k < 1:
            raise ValueError("INITIAL_TOP_K and TOP_K must be positive")
        model = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
        device = os.getenv("RERANKER_DEVICE", "cpu").lower()
        if model != "BAAI/bge-reranker-v2-m3" or device != "cpu":
            raise ValueError("RERANKER_MODEL_ID must be BAAI/bge-reranker-v2-m3 and RERANKER_DEVICE must be cpu")
        return cls(
            source_jsonl=_path_from_env("VECTOR_STORE_SOURCE_JSONL", DEFAULT_SOURCE),
            store_dir=_path_from_env("VECTOR_STORE_DIR", DEFAULT_STORE),
            embedding_url=os.getenv("BGE_EMBEDDING_URL", "http://127.0.0.1:8005").rstrip("/"),
            reranker_url=os.getenv("RERANKER_URL", "http://127.0.0.1:8003").rstrip("/"),
            initial_top_k=top_k,
            final_top_k=final_top_k,
            reranker_model=model,
            reranker_device=device,
            reranker_enabled=os.getenv("RERANKER_ENABLED", "true").strip().lower() in {"true", "1", "yes"},
        )
