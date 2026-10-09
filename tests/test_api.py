import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.inference.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "healthy"
    assert "fire" in json_resp["classes"]


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    json_resp = response.json()
    assert "project" in json_resp
    assert "data" in json_resp


def test_predict_endpoint():
    # Create dummy RGB image in memory
    img = Image.new("RGB", (224, 224), color=(255, 100, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    files = {"file": ("test_fire.jpg", img_byte_arr, "image/jpeg")}
    response = client.post("/predict?model_type=transfer_learning", files=files)

    assert response.status_code == 200
    json_resp = response.json()
    assert "prediction" in json_resp
    assert json_resp["prediction"] in ["fire", "smoke"]
    assert "confidence" in json_resp
    assert "class_probabilities" in json_resp
