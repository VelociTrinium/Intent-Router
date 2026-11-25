"""
Unit tests for routing rules, guardrails, chat_history formatting,
and Ollama single-token logprobs probability extraction.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from app.intent_spec import (
    build_ollama_messages,
    extract_ollama_token_probabilities,
    format_text_with_history,
)
from app.router_engine import RouterEngine, BaseRouter, OllamaRouter
from app.schemas import RouteResponse


class MockRouter(BaseRouter):
    """Deterministic mock router for testing routing guardrail logic."""
    def __init__(self, probabilities: Dict[str, float]):
        self._probabilities = probabilities

    def set_probabilities(self, probabilities: Dict[str, float]):
        self._probabilities = probabilities

    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        return self._probabilities

    def is_ready(self) -> bool:
        return True


@pytest.fixture
def test_engine():
    engine = RouterEngine(backend_override="baseline")
    mock_router = MockRouter({
        "GENERAL_CHAT": 0.34,
        "DB_READ": 0.33,
        "DB_WRITE": 0.33
    })
    engine.router = mock_router
    return engine, mock_router


def test_write_threshold_triggered(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({"GENERAL_CHAT": 0.13, "DB_READ": 0.45, "DB_WRITE": 0.42})

    result: RouteResponse = engine.route("Update customer profile record")
    assert result.route == "DB_WRITE"
    assert result.raw_prediction == "DB_READ"
    assert result.fallback_triggered is True
    assert result.confidence == 0.42


def test_fallback_to_general_chat_when_uncertain(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({"GENERAL_CHAT": 0.20, "DB_READ": 0.55, "DB_WRITE": 0.25})

    result: RouteResponse = engine.route("Something ambiguous or unclear")
    assert result.route == "GENERAL_CHAT"
    assert result.raw_prediction == "DB_READ"
    assert result.fallback_triggered is True


def test_high_confidence_db_read(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({"GENERAL_CHAT": 0.05, "DB_READ": 0.88, "DB_WRITE": 0.07})

    result: RouteResponse = engine.route("Show all companies in China")
    assert result.route == "DB_READ"
    assert result.raw_prediction == "DB_READ"
    assert result.confidence == 0.88
    assert result.fallback_triggered is False


def test_high_confidence_general_chat(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({"GENERAL_CHAT": 0.92, "DB_READ": 0.05, "DB_WRITE": 0.03})

    result: RouteResponse = engine.route("How does indexing speed up a query?")
    assert result.route == "GENERAL_CHAT"
    assert result.raw_prediction == "GENERAL_CHAT"
    assert result.confidence == 0.92
    assert result.fallback_triggered is False


def test_high_confidence_db_write(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({"GENERAL_CHAT": 0.02, "DB_READ": 0.03, "DB_WRITE": 0.95})

    result: RouteResponse = engine.route("Delete row 42")
    assert result.route == "DB_WRITE"
    assert result.raw_prediction == "DB_WRITE"
    assert result.confidence == 0.95
    assert result.fallback_triggered is False


def test_chat_history_optional_input(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({"GENERAL_CHAT": 0.10, "DB_READ": 0.80, "DB_WRITE": 0.10})

    history = [
        {"role": "user", "content": "I am looking for property listings."},
        {"role": "assistant", "content": "What area are you interested in?"}
    ]
    result: RouteResponse = engine.route("BKC Mumbai", chat_history=history)
    assert result.route == "DB_READ"
    assert result.confidence == 0.80


def test_format_text_with_history():
    assert format_text_with_history("Show all deals") == "Show all deals"
    formatted = format_text_with_history(
        "And in Singapore?",
        chat_history=[
            {"role": "user", "content": "List properties in Mumbai"},
            {"role": "assistant", "content": "Found 12 properties in Mumbai."},
        ],
    )
    assert "User: List properties in Mumbai" in formatted
    assert "Assistant: Found 12 properties in Mumbai." in formatted
    assert formatted.endswith("User: And in Singapore?")


def test_build_ollama_messages():
    msgs = build_ollama_messages(
        prompt="Yes, delete it",
        chat_history=[
            {"role": "user", "content": "Delete lead #402"},
            {"role": "assistant", "content": "Are you sure?"},
        ],
        include_system=False,
    )
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "Yes, delete it"


def test_extract_ollama_token_probabilities_write_trigger():
    # READ=0.52, WRITE=0.45, CHAT=0.03 -> raw is READ, but WRITE >= 0.40 guardrail triggers DB_WRITE
    mock_resp = {
        "message": {"role": "assistant", "content": "READ"},
        "logprobs": [
            {
                "token": "READ",
                "logprob": math.log(0.52),
                "top_logprobs": [
                    {"token": "READ", "logprob": math.log(0.52)},
                    {"token": "WRITE", "logprob": math.log(0.45)},
                    {"token": "CHAT", "logprob": math.log(0.03)},
                ],
            }
        ],
    }
    probs = extract_ollama_token_probabilities(mock_resp)
    assert probs is not None
    assert abs(probs["DB_READ"] - 0.52) < 0.01
    assert abs(probs["DB_WRITE"] - 0.45) < 0.01
    assert abs(probs["GENERAL_CHAT"] - 0.03) < 0.01

    engine = RouterEngine(backend_override="ollama", ollama_model_override="intent-router")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return mock_resp

    class FakeClient:
        def post(self, *args, **kwargs):
            return FakeResp()

    assert isinstance(engine.router, OllamaRouter)
    engine.router._client = FakeClient()
    res = engine.route("Check and update lead #402")
    assert res.raw_prediction == "DB_READ"
    assert res.route == "DB_WRITE"
    assert res.fallback_triggered is True


def test_build_dataset_custom_priority_and_formats(tmp_path):
    import json
    from scripts.build_dataset import load_custom_file, build_splits

    json_list = tmp_path / "examples.json"
    json_list.write_text(json.dumps([
        {"prompt": "What is our biggest deal in Miami?", "label": "DB_READ", "split": "test"},
        {"text": "Update lead 10", "intent": "WRITE", "chat_history": [{"role": "user", "content": "Show lead 10"}, {"role": "assistant", "content": "Lead 10 is open."}]},
    ]), encoding="utf-8")

    json_dict = tmp_path / "grouped.json"
    json_dict.write_text(json.dumps({
        "GENERAL_CHAT": ["Hello Mira"],
        "DB_WRITE": ["Archive old deal #99"],
    }), encoding="utf-8")

    custom = load_custom_file(json_list) + load_custom_file(json_dict)
    for c in custom:
        c["_is_custom"] = True

    base = [
        {"text": "What is our biggest deal in Miami?", "label": "GENERAL_CHAT", "_is_custom": False},
        {"text": "List all companies in China", "label": "DB_READ", "_is_custom": False},
    ]

    train_s, val_s, test_s, stats = build_splits(custom, base, custom_train_repeat=2)
    assert stats["custom_overrode_synthetic"] == 1
    miami_rows = [r for r in test_s if r["text"] == "What is our biggest deal in Miami?"]
    assert len(miami_rows) == 1
    assert miami_rows[0]["label"] == "DB_READ"
    assert not any(r["text"] == "What is our biggest deal in Miami?" for r in train_s)


def test_export_template_detection(tmp_path):
    import json
    from scripts.export_ollama_gguf import detect_template_family

    d1 = tmp_path / "llama"
    d1.mkdir()
    (d1 / "tokenizer_config.json").write_text('{"eos_token": "<|eot_id|>"}', encoding="utf-8")
    assert detect_template_family(d1) == "llama3"

    d2 = tmp_path / "qwen"
    d2.mkdir()
    (d2 / "training_metadata.json").write_text(json.dumps({"chat_family": "chatml"}), encoding="utf-8")
    assert detect_template_family(d2) == "chatml"
