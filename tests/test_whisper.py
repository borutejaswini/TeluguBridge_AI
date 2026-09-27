from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import Settings
from app.core.errors import ConfigurationError, ModelLoadError
from app.services.whisper import MAX_NEW_TOKENS, TARGET_SAMPLE_RATE, WhisperService


def test_safe_generation_limit_and_sample_rate():
    assert MAX_NEW_TOKENS == 444
    assert TARGET_SAMPLE_RATE == 16_000


def test_requires_resampled_mono_audio():
    service = WhisperService(Settings(_env_file=None, whisper_lora_path=""))
    with pytest.raises(ConfigurationError, match="resampled"):
        service.transcribe_array(np.zeros(100), 48_000)


def test_configured_missing_lora_fails_without_fallback(tmp_path):
    service = WhisperService(Settings(_env_file=None, whisper_lora_path=str(tmp_path / "adapter")))
    # Validation occurs before attempting any model download.
    with pytest.raises(ModelLoadError, match="adapter files are missing"):
        service._validate_lora_path()



