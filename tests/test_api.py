"""
FastAPI integration tests checking GET /health and POST /route endpoints.
"""

import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.router_engine import BaseRouter


class DummyRouter(BaseRouter):
    def predict_probabilities(self, text: str) -> Dict[str, float]:
        lower = text.lower()
        if "delete" in lower or "update" in lower or "insert" in lower:
            return {"READ": 0.05, "WRITE": 0.90, "GENERAL_CHAT": 0.05}
        elif "show" in lower or "find" in lower or "search" in lower:
            return {"READ": 0.85, "WRITE": 0.05, "GENERAL_CHAT": 0.10}
        else:
            return {"READ": 0.10, "WRITE": 0.10, "GENERAL_CHAT": 0.80}

    def is_ready(self) -> bool:
        return True


@pytest.fixture
def client(monkeypatch):
    from app.main import engine
    monkeypatch.setattr(engine, "router", DummyRouter())
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "backend" in data
    assert data["model_loaded"] is True
    assert data["confidence_threshold"] == 0.60
    assert data["write_threshold"] == 0.40


def test_route_read_endpoint(client):
    payload = {
        "prompt": "Show me customer orders from June",
        "chat_history": []
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "READ"
    assert data["confidence"] > 0.60
    assert "READ" in data["probabilities"]
    assert "WRITE" in data["probabilities"]
    assert "GENERAL_CHAT" in data["probabilities"]
    assert data["latency_ms"] >= 0.0


def test_route_write_endpoint(client):
    payload = {
        "prompt": "Delete user #1094 from database"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "WRITE"
    assert data["confidence"] >= 0.40


def test_route_chat_endpoint(client):
    payload = {
        "prompt": "Tell me a story about a dragon"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "GENERAL_CHAT"


def test_route_empty_prompt_rejected(client):
    payload = {
        "prompt": "   "
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 422
