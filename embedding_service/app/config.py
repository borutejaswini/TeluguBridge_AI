from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_id: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 1
    max_length: int = 8192
    host: str = "127.0.0.1"
    port: int = 8005

    model_config = SettingsConfigDict(
        env_prefix="BGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
