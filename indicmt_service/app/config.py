from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "facebook/nllb-200-distilled-600M"
    device: str = "cpu"
    source_lang: str = "tel_Telu"
    target_lang: str = "eng_Latn"
    max_input_tokens: int = 512
    max_new_tokens: int = 256
    host: str = "127.0.0.1"
    port: int = 8001
    hf_token: str = ""

    model_config = SettingsConfigDict(
        env_prefix="NLLB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

