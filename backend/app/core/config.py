from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"
    max_concurrent_whisper: int = 1

    whisper_device: str = "cpu"
    whisper_model_id: str = "openai/whisper-large-v3-turbo"
    whisper_lora_path: str = "./models/after_chunk_01"
    nllb_model_id: str = "facebook/nllb-200-distilled-600M"
    nllb_device: str = "cpu"
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = False
    reranker_device: str = "cpu"
    llm_provider: str = "groq"
    gemini_model: str = "gemini-3.8-flash"
    import os

    GCP_API_KEY = os.getenv("GCP_API_KEY")
    groq_model: str = "openai/gpt-oss-120b"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    tts_model_id: str = "gtts/te"
    tts_url: str = "http://127.0.0.1:8002"
    nllb_url: str = "http://127.0.0.1:8001"
    retrieval_url: str = "http://127.0.0.1:8004"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta"
    groq_url: str = "https://api.groq.com/openai/v1"
    tts_timeout_seconds: float = 900

    hf_home: str = ""
    hf_hub_cache: str = ""
    transformers_cache: str = ""
    hf_token: str = ""
    knowledge_base_dir: str = "./knowledge_base"
    vector_store_dir: str = "./data/vector_store"
    top_k: int = 5
    service_timeout_seconds: float = 30

    model_config = SettingsConfigDict(
        env_file=[str(Path(__file__).resolve().parents[2] / ".env"), ".env"],
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def validate_hf_cache_paths(self) -> None:
        for label, value in (("HF_HOME", self.hf_home), ("HF_HUB_CACHE", self.hf_hub_cache), ("TRANSFORMERS_CACHE", self.transformers_cache)):
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.exists() or not path.is_dir():
                raise ValueError(f"{label} is configured but does not exist as a directory: {path}")
            if not _is_writable(path):
                raise ValueError(f"{label} is configured but is not writable: {path}")


def _is_writable(path: Path) -> bool:
    try:
        probe = path / ".telugubridge_write_check"
        probe.touch(exist_ok=False)
        probe.unlink()
        return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
