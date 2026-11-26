"""
FastAPI integration tests checking GET /health and POST /route endpoints.
Validates:
- Correct JSON schema: route, raw_prediction, confidence, probabilities, fallback_triggered
- 3 classes: GENERAL_CHAT, DB_READ, DB_WRITE
- Handling of optional chat_history
- Rejection of empty/whitespace prompts with 422
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient
import app.main as main_module
from app.main import app
from app.router_engine import RouterEngine


class MockApiRouter:
    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        lower = prompt.lower()
        if "delete" in lower or "update" in lower or "insert" in lower:
            return {"GENERAL_CHAT": 0.04, "DB_READ": 0.05, "DB_WRITE": 0.91}
        elif "show" in lower or "find" in lower or "list" in lower:
            return {"GENERAL_CHAT": 0.08, "DB_READ": 0.88, "DB_WRITE": 0.04}
        else:
            return {"GENERAL_CHAT": 0.85, "DB_READ": 0.10, "DB_WRITE": 0.05}

    def is_ready(self) -> bool:
        return True


@pytest.fixture
def client(monkeypatch):
    test_engine = RouterEngine(router=MockApiRouter())
    monkeypatch.setattr(main_module, "engine", test_engine)
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
    assert "GENERAL_CHAT" in data["classes"]
    assert "DB_READ" in data["classes"]
    assert "DB_WRITE" in data["classes"]


def test_route_read_endpoint(client):
    payload = {
        "prompt": "Show me all active pursuits with deal value greater than $1,000,000.",
        "chat_history": []
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "DB_READ"
    assert data["raw_prediction"] == "DB_READ"
    assert data["confidence"] >= 0.60
    assert "GENERAL_CHAT" in data["probabilities"]
    assert "DB_READ" in data["probabilities"]
    assert "DB_WRITE" in data["probabilities"]
    assert data["fallback_triggered"] is False


def test_route_write_endpoint(client):
    payload = {
        "prompt": "Delete row 42 immediately",
        "chat_history": None
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "DB_WRITE"
    assert data["raw_prediction"] == "DB_WRITE"
    assert data["confidence"] == 0.91
    assert data["fallback_triggered"] is False
    assert data["probabilities"]["DB_WRITE"] == 0.91


def test_route_chat_endpoint(client):
    payload = {
        "prompt": "How does indexing speed up a SELECT query?"
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "GENERAL_CHAT"
    assert data["raw_prediction"] == "GENERAL_CHAT"
    assert data["confidence"] >= 0.60
    assert data["fallback_triggered"] is False


def test_route_empty_prompt_validation(client):
    payload = {"prompt": "    "}
    response = client.post("/route", json=payload)
    assert response.status_code == 422


def test_route_with_chat_history_endpoint(client):
    payload = {
        "prompt": "Yes, delete it now",
        "chat_history": [
            {"role": "user", "content": "Check lead #402"},
            {"role": "assistant", "content": "Lead #402 is inactive. Want me to delete it?"},
        ],
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "DB_WRITE"
    assert data["confidence"] == 0.91


def test_standalone_ollama_client(monkeypatch):
    import io
    import json
    import math
    import urllib.request
    from app.ollama_client import classify_intent

    mock_payload = {
        "message": {"role": "assistant", "content": "READ"},
        "logprobs": [
            {
                "token": "READ",
                "logprob": math.log(0.50),
                "top_logprobs": [
                    {"token": "READ", "logprob": math.log(0.50)},
                    {"token": "WRITE", "logprob": math.log(0.44)},
                    {"token": "CHAT", "logprob": math.log(0.06)},
                ],
            }
        ],
    }

    class FakeHttpResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=15.0: FakeHttpResp(json.dumps(mock_payload).encode("utf-8")),
    )
    res = classify_intent("Check and update lead #402", version="v2")
    assert res["model"] == "intent-router-qwen2.5-0.5b-instruct:v2"
    assert res["raw_prediction"] == "DB_READ"
    assert res["route"] == "DB_WRITE"
    assert res["fallback_triggered"] is True
