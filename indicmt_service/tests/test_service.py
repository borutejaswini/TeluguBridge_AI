from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from app.config import Settings
from app.errors import TranslationServiceError
from app.service import NllbTranslationService


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    unk_token_id = 0
    src_lang = None

    def __init__(self):
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return {"tel_Telu": 10, "eng_Latn": 20}.get(token, 0)

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return FakeBatch(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, output, skip_special_tokens=True):
        return ["The crop received TBQZX0000ZX today."]


class FakeModel:
    def __init__(self):
        self.generate_calls = []
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[4, 5]])


def make_service():
    tokenizer, model = FakeTokenizer(), FakeModel()
    service = NllbTranslationService(
        Settings(_env_file=None),
        loader=lambda _: (tokenizer, model),
    )
    return service, tokenizer, model


def test_load_is_once_cpu_and_uses_exact_model_configuration():
    service, tokenizer, model = make_service()
    service.load()
    first_load_time = service.load_seconds
    service.load()
    assert service.is_loaded
    assert service.settings.model_id == "facebook/nllb-200-distilled-600M"
    assert service.settings.source_lang == "tel_Telu"
    assert service.settings.target_lang == "eng_Latn"
    assert tokenizer.src_lang == "tel_Telu"
    assert model.to_device == "cpu"
    assert model.eval_called
    assert service.load_seconds == first_load_time


def test_translation_forces_english_and_restores_quantities():
    service, tokenizer, model = make_service()
    service.load()
    result = service.translate("పంటకు 5 కిలోలు ఎరువు వాడారు")
    assert result.translation == "The crop received 5 కిలోలు today."
    assert tokenizer.calls[0][0] == "పంటకు TBQZX0000ZX ఎరువు వాడారు"
    assert model.generate_calls[0]["forced_bos_token_id"] == 20
    assert model.generate_calls[0]["max_new_tokens"] == service.settings.max_new_tokens
    assert result.device == "cpu"
    assert result.input_language == "tel_Telu"
    assert result.output_language == "eng_Latn"


def test_missing_quantity_marker_returns_structured_integrity_error():
    service, tokenizer, _ = make_service()
    service.load()
    tokenizer.batch_decode = lambda *args, **kwargs: ["The crop received fertilizer today."]
    with pytest.raises(TranslationServiceError) as error:
        service.translate("పంటకు 5 కిలోలు ఎరువు వాడారు")
    assert error.value.code == "TRANSLATION_INTEGRITY_ERROR"


def test_empty_text_is_a_structured_error():
    service, _, _ = make_service()
    service.load()
    with pytest.raises(TranslationServiceError) as error:
        service.translate("  ")
    assert error.value.code == "INVALID_INPUT"

