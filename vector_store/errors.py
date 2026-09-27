from __future__ import annotations


class VectorStoreError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def as_dict(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}

