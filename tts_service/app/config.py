from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "ai4bharat/indic-parler-tts"
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8002
    output_dir: Path = Path("outputs/audio")
    max_text_characters: int = 2000
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    voice_description: str = (
        "A clear, natural Telugu-speaking voice speaks at a moderate pace with a warm, helpful tone. "
        "The recording is clean and close with very clear audio."
    )

    model_config = SettingsConfigDict(env_prefix="TTS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
