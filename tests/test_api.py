import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api import main


class FakeModelService:
    def load(self):
        pass

    def is_ready(self):
        return True

    def predict(self, image_bytes):
        return b"generated-png", 12.5


def test_health_reports_model_readiness(monkeypatch):
    monkeypatch.setattr(main, "model_service", FakeModelService())

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_rejects_non_image_uploads(monkeypatch):
    monkeypatch.setattr(main, "model_service", FakeModelService())

    with TestClient(main.app) as client:
        response = client.post("/predict", files={"file": ("note.txt", b"text", "text/plain")})

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


def test_predict_streams_model_png_and_timing_header(monkeypatch):
    monkeypatch.setattr(main, "model_service", FakeModelService())
    image = Image.new("RGB", (2, 2))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    with TestClient(main.app) as client:
        response = client.post(
            "/predict", files={"file": ("satellite.png", image_bytes.getvalue(), "image/png")}
        )

    assert response.status_code == 200
    assert response.content == b"generated-png"
    assert response.headers["x-inference-time-ms"] == "12.5"
