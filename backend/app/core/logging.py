import logging
import os


def configure_logging(level: str | None = None) -> None:
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

