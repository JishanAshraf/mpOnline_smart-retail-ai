"""Pytest suite for verifying all FastAPI endpoints using TestClient.

Tests status codes, response JSON schema structures, and security input validations.
"""

import io
import os
import sys
import pytest

# Bootstrapping sys.path for test execution across all working directories
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tests_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_dummy_image_bytes() -> bytes:
    """Helper function to create a minimal 1x1 black JPEG image in memory."""
    try:
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded_img = cv2.imencode(".jpg", img)
        return encoded_img.tobytes()
    except Exception:
        # Fallback dummy JPEG bytes if cv2 image generation is bypassed
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"


def test_root_health_check():
    """Verify root UI endpoint and API status endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    api_resp = client.get("/api/status")
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert data["status"] == "online"


def test_recognize_face_endpoint():
    """Verify POST /recognize-face accepts image upload and returns valid response shape."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test_face.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/recognize-face", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "customer_id" in data
    assert "status" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_recognize_face_invalid_file_type():
    """Verify POST /recognize-face rejects non-image files with 400."""
    files = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/recognize-face", files=files)
    assert response.status_code == 400


def test_classify_product_endpoint():
    """Verify POST /classify-product accepts image upload and returns valid response shape."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test_product.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/classify-product", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_analyze_sentiment_endpoint():
    """Verify POST /analyze-sentiment returns sentiment label and confidence score."""
    payload = {"text": "I really love shopping at this store, excellent customer support!"}
    response = client.post("/analyze-sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert "sentiment" in data
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert "confidence" in data


def test_analyze_sentiment_validation():
    """Verify POST /analyze-sentiment rejects empty or oversized strings with 400."""
    # Empty string
    resp_empty = client.post("/analyze-sentiment", json={"text": "   "})
    assert resp_empty.status_code == 400

    # Oversized string
    resp_oversized = client.post("/analyze-sentiment", json={"text": "a" * 6000})
    assert resp_oversized.status_code == 400


def test_chatbot_endpoint():
    """Verify POST /chatbot returns intent classification and reply message."""
    payload = {"message": "Where is my order status?", "user_id": "usr_99"}
    response = client.post("/chatbot", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "intent" in data
    assert "match_type" in data


def test_chatbot_validation():
    """Verify POST /chatbot rejects empty or oversized messages with 400."""
    resp_empty = client.post("/chatbot", json={"message": ""})
    assert resp_empty.status_code == 400

    resp_oversized = client.post("/chatbot", json={"message": "x" * 2000})
    assert resp_oversized.status_code == 400


def test_dashboard_stats_endpoint():
    """Verify GET /dashboard/stats returns aggregate metrics JSON."""
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_visits" in data
    assert "sentiment_counts" in data
    assert "top_intents" in data
    assert isinstance(data["top_intents"], list)
