from dataclasses import dataclass


@dataclass
class AppError(Exception):
    """An error safe to serialize in the API response."""

    code: str
    message: str
    status_code: int = 500

    def __str__(self) -> str:
        return self.message


class ConfigurationError(AppError):
    def __init__(self, message: str):
        super().__init__("CONFIGURATION_ERROR", message, 500)


class ModelLoadError(AppError):
    def __init__(self, message: str):
        super().__init__("MODEL_LOAD_ERROR", message, 503)

