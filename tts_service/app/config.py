from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "gtts/te"          # Google TTS, Telugu language
    device: str = "cpu"                # gTTS uses Google API (no local GPU)
    host: str = "127.0.0.1"
    port: int = 8002
    output_dir: Path = Path("outputs/audio")
    max_text_characters: int = 4000    # gTTS supports up to ~5000 chars
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    gtts_lang: str = "te"              # Telugu language code for gTTS
    gtts_slow: bool = False            # Normal speed

    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
