from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import Settings
from app.core.errors import ConfigurationError, ModelLoadError


def test_model_defaults_are_exact_and_conservative():
    settings = Settings(_env_file=None)
    assert settings.whisper_model_id == "openai/whisper-large-v3-turbo"
    assert settings.max_concurrent_whisper == 1
    assert settings.nllb_device == "cpu"
    assert settings.embedding_device == "cpu"


def test_errors_have_structured_codes():
    assert ConfigurationError("bad config").code == "CONFIGURATION_ERROR"
    assert ModelLoadError("bad model").code == "MODEL_LOAD_ERROR"

