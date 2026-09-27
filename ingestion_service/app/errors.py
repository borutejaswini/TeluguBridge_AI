class IngestionError(Exception):
    def __init__(self, code: str, message: str, *, source: str | None = None, page: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.source = source
        self.page = page

    def as_dict(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "source": self.source,
                "page_number": self.page,
            }
        }

