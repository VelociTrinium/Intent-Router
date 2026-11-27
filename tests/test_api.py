"""
FastAPI integration tests checking GET /health and POST /route endpoints.
Validates:
- Correct JSON schema: route, raw_prediction, confidence, probabilities, fallback_triggered
- 4 classes: GENERAL_CHAT, DB_READ, DB_WRITE, CLARIFICATION_REQUIRED
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
        if "delete" in lower or lower in {"update it", "update kevin's phone number"}:
            return {
                "GENERAL_CHAT": 0.03,
                "DB_READ": 0.04,
                "DB_WRITE": 0.08,
                "CLARIFICATION_REQUIRED": 0.85,
            }
        elif ("update" in lower or "create" in lower) and (chat_history or " to " in lower or "in india" in lower):
            return {
                "GENERAL_CHAT": 0.03,
                "DB_READ": 0.04,
                "DB_WRITE": 0.90,
                "CLARIFICATION_REQUIRED": 0.03,
            }
        elif "show" in lower or "find" in lower or "list" in lower:
            return {
                "GENERAL_CHAT": 0.06,
                "DB_READ": 0.87,
                "DB_WRITE": 0.04,
                "CLARIFICATION_REQUIRED": 0.03,
            }
        else:
            return {
                "GENERAL_CHAT": 0.85,
                "DB_READ": 0.08,
                "DB_WRITE": 0.04,
                "CLARIFICATION_REQUIRED": 0.03,
            }

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
    assert data["clarify_threshold"] == 0.40
    assert data["fallback_route"] == "CLARIFICATION_REQUIRED"
    assert set(data["classes"]) == {"GENERAL_CHAT", "DB_READ", "DB_WRITE", "CLARIFICATION_REQUIRED"}


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
    assert set(data["probabilities"].keys()) == {"GENERAL_CHAT", "DB_READ", "DB_WRITE", "CLARIFICATION_REQUIRED"}
    assert data["fallback_triggered"] is False


def test_route_write_endpoint(client):
    payload = {
        "prompt": "Update Kevin Sauer's mobile at Microsoft to +1 214 555 8721",
        "chat_history": None
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "DB_WRITE"
    assert data["raw_prediction"] == "DB_WRITE"
    assert data["confidence"] == 0.90
    assert data["fallback_triggered"] is False
    assert data["probabilities"]["DB_WRITE"] == 0.90


def test_route_clarification_endpoint(client):
    payload = {
        "prompt": "Update Kevin's phone number",
        "chat_history": None
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "CLARIFICATION_REQUIRED"
    assert data["raw_prediction"] == "CLARIFICATION_REQUIRED"
    assert data["confidence"] == 0.85
    assert data["fallback_triggered"] is False


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
        "prompt": "Update status to qualified",
        "chat_history": [
            {"role": "user", "content": "Check lead #402"},
            {"role": "assistant", "content": "Lead #402 is open. What status should I set?"},
        ],
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "DB_WRITE"
    assert data["confidence"] == 0.90


def test_standalone_ollama_client(monkeypatch):
    import io
    import json
    import math
    import urllib.request
    from app.ollama_client import classify_intent

    # 1. Mixed READ+WRITE when ASK is low -> DB_WRITE triggers
    mock_payload_write = {
        "message": {"role": "assistant", "content": "READ"},
        "logprobs": [
            {
                "token": "READ",
                "logprob": math.log(0.50),
                "top_logprobs": [
                    {"token": "READ", "logprob": math.log(0.50)},
                    {"token": "WRITE", "logprob": math.log(0.44)},
                    {"token": "CHAT", "logprob": math.log(0.03)},
                    {"token": "ASK", "logprob": math.log(0.03)},
                ],
            }
        ],
    }

    # 2. Ambiguous / destructive write where ASK >= WRITE -> CLARIFICATION_REQUIRED wins
    mock_payload_ask = {
        "message": {"role": "assistant", "content": "ASK"},
        "logprobs": [
            {
                "token": "ASK",
                "logprob": math.log(0.52),
                "top_logprobs": [
                    {"token": "ASK", "logprob": math.log(0.52)},
                    {"token": "WRITE", "logprob": math.log(0.43)},
                    {"token": "READ", "logprob": math.log(0.03)},
                    {"token": "CHAT", "logprob": math.log(0.02)},
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
        lambda req, timeout=15.0: FakeHttpResp(json.dumps(mock_payload_write).encode("utf-8")),
    )
    res = classify_intent("Check lead #402 and update status to qualified", version="v2")
    assert res["model"] == "intent-router-qwen2.5-0.5b-instruct:v2"
    assert res["raw_prediction"] == "DB_READ"
    assert res["route"] == "DB_WRITE"
    assert res["fallback_triggered"] is True

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=15.0: FakeHttpResp(json.dumps(mock_payload_ask).encode("utf-8")),
    )
    res_ask = classify_intent("Delete Microsoft from CRM", version="v2")
    assert res_ask["raw_prediction"] == "CLARIFICATION_REQUIRED"
    assert res_ask["route"] == "CLARIFICATION_REQUIRED"
    assert res_ask["fallback_triggered"] is False

    # 3. Destructive bulk write wrongly emitted as WRITE -> intercepted to CLARIFICATION_REQUIRED
    mock_payload_bulk_write = {
        "message": {"role": "assistant", "content": "WRITE"},
        "logprobs": [
            {
                "token": "WRITE",
                "logprob": math.log(0.97),
                "top_logprobs": [
                    {"token": "WRITE", "logprob": math.log(0.97)},
                    {"token": "CHAT", "logprob": math.log(0.01)},
                    {"token": "READ", "logprob": math.log(0.01)},
                    {"token": "ASK", "logprob": math.log(0.01)},
                ],
            }
        ],
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=15.0: FakeHttpResp(json.dumps(mock_payload_bulk_write).encode("utf-8")),
    )
    res_bulk = classify_intent("Override every active deal stage status to 'Terminated'", version="v2")
    assert res_bulk["raw_prediction"] == "DB_WRITE"
    assert res_bulk["route"] == "CLARIFICATION_REQUIRED"
    assert res_bulk["fallback_triggered"] is True


def test_review_logging_respects_config_flag(tmp_path, monkeypatch):
    import json
    from app.config import settings

    log_file = tmp_path / "review.jsonl"
    monkeypatch.setattr(settings, "REVIEW_LOG_PATH", str(log_file))
    monkeypatch.setattr(settings, "REVIEW_CONFIDENCE_THRESHOLD", 0.95)
    monkeypatch.setattr(settings, "ENABLE_REVIEW_LOGGING", False)

    test_engine = RouterEngine(router=MockApiRouter())
    monkeypatch.setattr(main_module, "engine", test_engine)
    tc = TestClient(app)

    r1 = tc.post("/route", json={"prompt": "Show all companies in China"})
    assert r1.status_code == 200
    assert not log_file.exists()

    monkeypatch.setattr(settings, "ENABLE_REVIEW_LOGGING", True)
    r2 = tc.post("/route", json={"prompt": "Show all companies in China"})
    assert r2.status_code == 200
    assert log_file.exists()
    logged = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert logged["prompt"] == "Show all companies in China"
    assert logged["route"] == "DB_READ"
