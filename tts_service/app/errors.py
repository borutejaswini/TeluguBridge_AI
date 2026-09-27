from dataclasses import dataclass


@dataclass
class TtsError(Exception):
    code: str
    message: str
    status_code: int = 500

    def __str__(self) -> str:
        return self.message
