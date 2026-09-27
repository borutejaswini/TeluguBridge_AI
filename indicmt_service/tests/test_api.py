import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from test_service import make_service


def test_health_and_translate_routes_with_mocked_model(monkeypatch):
    main = importlib.import_module("app.main")
    service, _, _ = make_service()
    monkeypatch.setattr(main, "translator", service)

    with TestClient(main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["loaded"] is True
        assert health.json()["device"] == "cpu"

        response = client.post("/translate", json={"text": "పంటకు 5 కిలోలు ఎరువు వాడారు"})
        assert response.status_code == 200
        assert response.json()["translation"] == "The crop received 5 కిలోలు today."
        assert response.json()["input_language"] == "tel_Telu"
        assert response.json()["output_language"] == "eng_Latn"


def test_translate_route_returns_structured_input_error(monkeypatch):
    main = importlib.import_module("app.main")
    service, _, _ = make_service()
    service.load()
    monkeypatch.setattr(main, "translator", service)

    with TestClient(main.app) as client:
        response = client.post("/translate", json={"text": "  "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"
