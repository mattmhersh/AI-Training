# tests/test_api.py

import pytest
from fastapi.testclient import TestClient

import sys

sys.path.append("../Runpod2")
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_calculate_cost():
    payload = {"hours": 10, "gpu_type": "RTX-4090"}
    response = client.post("/calculate-cost", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_cost"] == 6.90
    assert data["rate_per_hour"] == 0.69


def test_invalid_gpu_type():
    payload = {"hours": 10, "gpu_type": "INVALID-GPU"}
    response = client.post("/calculate-cost", json=payload)
    assert "error" in response.json()
