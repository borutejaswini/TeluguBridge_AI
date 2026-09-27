import pytest

from app.config import Settings
from app.errors import TtsError
from app.service import IndicParlerService


def test_tts_rejects_empty_text_before_loading_model():
    service = IndicParlerService(Settings())
    with pytest.raises(TtsError) as error:
        service.synthesize("   ")
    assert error.value.code == "EMPTY_TEXT"
    assert not service.is_loaded


def test_tts_enforces_cpu_device():
    service = IndicParlerService(Settings(device="cuda"))
    with pytest.raises(TtsError) as error:
        service.load()
    assert error.value.code == "INVALID_DEVICE"
    assert not service.is_loaded
