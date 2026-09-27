from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


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
            if key.strip() in {"RERANKER_MODEL_ID", "RERANKER_DEVICE", "RERANKER_ENABLED", "TOP_K", "INITIAL_TOP_K", "RERANKER_URL"}:
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except OSError:
        return


@dataclass(frozen=True)
class Settings:
    model_id: str = RERANKER_MODEL_ID
    device: str = "cpu"
    enabled: bool = False
    batch_size: int = 8
    host: str = "127.0.0.1"
    port: int = 8003

    @classmethod
    def from_env(cls) -> "Settings":
        _load_project_env()
        return cls(
            model_id=os.getenv("RERANKER_MODEL_ID", RERANKER_MODEL_ID),
            device=os.getenv("RERANKER_DEVICE", "cpu").lower(),
            enabled=os.getenv("RERANKER_ENABLED", "false").strip().lower() in {"true", "1", "yes"},
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
        )

