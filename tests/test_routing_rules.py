"""
Unit tests for 4-class routing rules, guardrails, chat_history formatting,
and Ollama single-token (CHAT/READ/WRITE/ASK) logprobs probability extraction.
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
    OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
    build_ollama_messages,
    extract_ollama_token_probabilities,
)
from app.router_engine import OllamaRouter, RouterEngine
from app.schemas import RouteResponse


class MockRouter:
    """Deterministic mock router for testing 4-class routing guardrail logic."""
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
    mock_router = MockRouter({
        "GENERAL_CHAT": 0.25,
        "DB_READ": 0.25,
        "DB_WRITE": 0.25,
        "CLARIFICATION_REQUIRED": 0.25,
    })
    engine = RouterEngine(router=mock_router)
    return engine, mock_router


def test_write_threshold_triggered_over_read(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.08,
        "DB_READ": 0.45,
        "DB_WRITE": 0.42,
        "CLARIFICATION_REQUIRED": 0.05,
    })

    result: RouteResponse = engine.route("Find HSBC contact and update phone to +1 555 0192")
    assert result.route == "DB_WRITE"
    assert result.raw_prediction == "DB_READ"
    assert result.fallback_triggered is True
    assert result.confidence == 0.42


def test_clarification_beats_write_when_stronger(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.04,
        "DB_READ": 0.06,
        "DB_WRITE": 0.42,
        "CLARIFICATION_REQUIRED": 0.48,
    })

    result: RouteResponse = engine.route("Delete Microsoft from the CRM")
    assert result.route == "CLARIFICATION_REQUIRED"
    assert result.raw_prediction == "CLARIFICATION_REQUIRED"
    assert result.fallback_triggered is False
    assert result.confidence == 0.48


def test_clarification_beats_write_on_threshold_tie(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.05,
        "DB_READ": 0.05,
        "DB_WRITE": 0.45,
        "CLARIFICATION_REQUIRED": 0.45,
    })

    result: RouteResponse = engine.route("Update Kevin's phone number")
    assert result.route == "CLARIFICATION_REQUIRED"
    assert result.confidence == 0.45


def test_write_wins_when_strictly_higher_than_clarify(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.03,
        "DB_READ": 0.05,
        "DB_WRITE": 0.50,
        "CLARIFICATION_REQUIRED": 0.42,
    })

    result: RouteResponse = engine.route("Update Kevin Sauer's mobile number to +1 214 555 8721")
    assert result.route == "DB_WRITE"
    assert result.raw_prediction == "DB_WRITE"
    assert result.fallback_triggered is False
    assert result.confidence == 0.50


def test_fallback_to_clarification_required_when_uncertain(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.20,
        "DB_READ": 0.55,
        "DB_WRITE": 0.15,
        "CLARIFICATION_REQUIRED": 0.10,
    })

    result: RouteResponse = engine.route("Something ambiguous or unclear")
    assert result.route == "CLARIFICATION_REQUIRED"
    assert result.raw_prediction == "DB_READ"
    assert result.fallback_triggered is True


def test_high_confidence_db_read(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.04,
        "DB_READ": 0.88,
        "DB_WRITE": 0.05,
        "CLARIFICATION_REQUIRED": 0.03,
    })

    result: RouteResponse = engine.route("Show all companies in China")
    assert result.route == "DB_READ"
    assert result.raw_prediction == "DB_READ"
    assert result.confidence == 0.88
    assert result.fallback_triggered is False


def test_high_confidence_general_chat(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.90,
        "DB_READ": 0.05,
        "DB_WRITE": 0.03,
        "CLARIFICATION_REQUIRED": 0.02,
    })

    result: RouteResponse = engine.route("How does indexing speed up a query?")
    assert result.route == "GENERAL_CHAT"
    assert result.raw_prediction == "GENERAL_CHAT"
    assert result.confidence == 0.90
    assert result.fallback_triggered is False


def test_high_confidence_db_write(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.02,
        "DB_READ": 0.03,
        "DB_WRITE": 0.92,
        "CLARIFICATION_REQUIRED": 0.03,
    })

    result: RouteResponse = engine.route("Create company Apex Digital Solutions in India and assign to me")
    assert result.route == "DB_WRITE"
    assert result.raw_prediction == "DB_WRITE"
    assert result.confidence == 0.92
    assert result.fallback_triggered is False


def test_high_confidence_clarification_required(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.02,
        "DB_READ": 0.04,
        "DB_WRITE": 0.05,
        "CLARIFICATION_REQUIRED": 0.89,
    })

    result: RouteResponse = engine.route("Update it")
    assert result.route == "CLARIFICATION_REQUIRED"
    assert result.raw_prediction == "CLARIFICATION_REQUIRED"
    assert result.confidence == 0.89
    assert result.fallback_triggered is False


def test_chat_history_optional_input(test_engine):
    engine, mock = test_engine
    mock.set_probabilities({
        "GENERAL_CHAT": 0.05,
        "DB_READ": 0.85,
        "DB_WRITE": 0.05,
        "CLARIFICATION_REQUIRED": 0.05,
    })

    history = [
        {"role": "user", "content": "I am looking for property listings."},
        {"role": "assistant", "content": "What area are you interested in?"}
    ]
    result: RouteResponse = engine.route("BKC Mumbai", chat_history=history)
    assert result.route == "DB_READ"
    assert result.confidence == 0.85


def test_build_ollama_messages():
    msgs = build_ollama_messages(
        prompt="Yes, set deal value to USD 2,000,000",
        chat_history=[
            {"role": "user", "content": "Update pursuit #402"},
            {"role": "assistant", "content": "What deal value would you like to set?"},
        ],
        include_system=True,
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "ASK" in msgs[0]["content"]
    assert "ASK" in OLLAMA_CLASSIFIER_SYSTEM_PROMPT
    assert msgs[-1]["content"] == "Yes, set deal value to USD 2,000,000"


def test_extract_ollama_token_probabilities_write_trigger():
    # READ=0.50, WRITE=0.44, CHAT=0.03, ASK=0.03 -> READ raw, WRITE >= 0.40 wins over low ASK
    mock_resp = {
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
    probs = extract_ollama_token_probabilities(mock_resp)
    assert probs is not None
    assert abs(probs["DB_READ"] - 0.50) < 0.01
    assert abs(probs["DB_WRITE"] - 0.44) < 0.01
    assert abs(probs["GENERAL_CHAT"] - 0.03) < 0.01
    assert abs(probs["CLARIFICATION_REQUIRED"] - 0.03) < 0.01

    engine = RouterEngine(ollama_model_override="intent-router")

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
    res = engine.route("Check and update lead #402 status to qualified")
    assert res.raw_prediction == "DB_READ"
    assert res.route == "DB_WRITE"
    assert res.fallback_triggered is True


def test_extract_ollama_token_probabilities_ask_beats_write():
    mock_resp = {
        "message": {"role": "assistant", "content": "ASK"},
        "logprobs": [
            {
                "token": "ASK",
                "logprob": math.log(0.51),
                "top_logprobs": [
                    {"token": "ASK", "logprob": math.log(0.51)},
                    {"token": "WRITE", "logprob": math.log(0.43)},
                    {"token": "READ", "logprob": math.log(0.04)},
                    {"token": "CHAT", "logprob": math.log(0.02)},
                ],
            }
        ],
    }
    probs = extract_ollama_token_probabilities(mock_resp)
    assert probs is not None
    assert abs(probs["CLARIFICATION_REQUIRED"] - 0.51) < 0.01

    engine = RouterEngine(ollama_model_override="intent-router")

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
    res = engine.route("Delete Microsoft from CRM")
    assert res.route == "CLARIFICATION_REQUIRED"
    assert res.raw_prediction == "CLARIFICATION_REQUIRED"
    assert res.fallback_triggered is False


def test_extract_ollama_token_probabilities_text_fallback_ask():
    mock_resp = {"message": {"role": "assistant", "content": "ASK"}, "logprobs": []}
    probs = extract_ollama_token_probabilities(mock_resp)
    assert probs is not None
    assert probs["CLARIFICATION_REQUIRED"] == 0.91
    assert abs(sum(probs.values()) - 1.0) < 0.001


def test_build_dataset_custom_priority_and_formats(tmp_path):
    import json
    from scripts.build_dataset import load_custom_file, build_splits

    json_list = tmp_path / "examples.json"
    json_list.write_text(json.dumps([
        {"prompt": "What is our biggest deal in Miami?", "label": "DB_READ", "split": "test"},
        {"text": "Update lead 10 to qualified", "intent": "WRITE", "chat_history": [{"role": "user", "content": "Show lead 10"}, {"role": "assistant", "content": "Lead 10 is open."}]},
        {"prompt": "Update that deal value", "intent": "ASK"},
    ]), encoding="utf-8")

    json_dict = tmp_path / "grouped.json"
    json_dict.write_text(json.dumps({
        "GENERAL_CHAT": ["Hello Mira"],
        "CLARIFICATION_REQUIRED": ["Delete row 42"],
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
    all_rows = train_s + val_s + test_s
    assert any(r["text"] == "Update that deal value" and r["label"] == "CLARIFICATION_REQUIRED" for r in all_rows)
    assert any(r["text"] == "Delete row 42" and r["label"] == "CLARIFICATION_REQUIRED" for r in all_rows)


def test_write_modelfile_chatml(tmp_path):
    from scripts.export_ollama_gguf import write_modelfile

    modelfile_path = tmp_path / "Modelfile"
    write_modelfile(modelfile_path, "intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf")
    content = modelfile_path.read_text(encoding="utf-8")
    assert "FROM ./intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf" in content
    assert "<|im_start|>" in content
    assert "ASK:" in content
    assert "PARAMETER num_predict 1" in content


def test_ollama_router_raises_on_unparseable_response():
    engine = RouterEngine(ollama_model_override="intent-router")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"role": "assistant", "content": "INVALID"}, "logprobs": []}

    class FakeClient:
        def post(self, *args, **kwargs):
            return FakeResp()

    assert isinstance(engine.router, OllamaRouter)
    engine.router._client = FakeClient()
    with pytest.raises(RuntimeError, match="Could not parse"):
        engine.route("Delete row 42")


def test_resolve_ollama_ref_and_windows_safe_stem():
    from scripts.export_ollama_gguf import resolve_ollama_ref

    ref, stem = resolve_ollama_ref("intent-router-qwen2.5-0.5b-instruct", "v1")
    assert ref == "intent-router-qwen2.5-0.5b-instruct:v1"
    assert stem == "intent-router-qwen2.5-0.5b-instruct-v1"
    assert ":" not in stem

    ref2, stem2 = resolve_ollama_ref("intent-router-qwen2.5-0.5b-instruct:2.0", "v9")
    assert ref2 == "intent-router-qwen2.5-0.5b-instruct:2.0"
    assert stem2 == "intent-router-qwen2.5-0.5b-instruct-2.0"

    ref3, stem3 = resolve_ollama_ref("intent-router-qwen2.5-0.5b-instruct", "")
    assert ref3 == "intent-router-qwen2.5-0.5b-instruct"
    assert stem3 == "intent-router-qwen2.5-0.5b-instruct"


def test_push_to_hf_dry_run(tmp_path, monkeypatch):
    import sys
    from scripts import push_to_hf

    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf").write_bytes(b"GGUF")
    (gguf_dir / "Modelfile").write_text("FROM ./intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["push_to_hf.py", "--repo-id", "user/test-gguf", "--gguf-dir", str(gguf_dir), "--tag", "v1.0.0", "--dry-run"],
    )
    push_to_hf.main()
    assert (gguf_dir / "README.md").exists()
    assert "user/test-gguf" in (gguf_dir / "README.md").read_text(encoding="utf-8")


def test_push_to_hf_creates_tag(tmp_path, monkeypatch):
    import sys
    import huggingface_hub
    from scripts import push_to_hf

    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf").write_bytes(b"GGUF")
    (gguf_dir / "Modelfile").write_text("FROM ./intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf\n", encoding="utf-8")

    calls = {"upload": [], "tag": []}

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, repo_id, repo_type="model", private=False, exist_ok=False):
            calls["repo"] = (repo_id, repo_type, private, exist_ok)

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="model", commit_message=""):
            calls["upload"].append((path_in_repo, repo_id))

        def create_tag(self, repo_id, *, tag, tag_message=None, repo_type=None, exist_ok=False):
            calls["tag"].append((repo_id, tag, tag_message, repo_type, exist_ok))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "push_to_hf.py",
            "--repo-id", "user/test-gguf",
            "--gguf-dir", str(gguf_dir),
            "--hf-token", "hf_test_token_value",
            "--tag", "v1.0.0",
        ],
    )
    push_to_hf.main()
    assert ("user/test-gguf", "v1.0.0", "Release v1.0.0", "model", True) in calls["tag"]


def test_cleanup_build_artifacts_dry_run_and_execute(tmp_path, monkeypatch):
    from scripts import cleanup

    fake_repo = tmp_path / "repo"
    (fake_repo / "checkpoints" / "run1").mkdir(parents=True)
    (fake_repo / "checkpoints" / "run1" / "model.safetensors").write_bytes(b"123456")
    (fake_repo / "models" / "ollama_lora").mkdir(parents=True)
    (fake_repo / "models" / "ollama_lora" / "adapter.bin").write_bytes(b"abcd")
    (fake_repo / "models" / "ollama_merged").mkdir(parents=True)
    (fake_repo / "models" / "ollama_merged" / "model.safetensors").write_bytes(b"xyz")
    (fake_repo / "models" / "ollama_gguf").mkdir(parents=True)
    (fake_repo / "models" / "ollama_gguf" / "v2.gguf").write_bytes(b"gguf")
    (fake_repo / "models" / "hf_base" / "Qwen2.5-0.5B-Instruct").mkdir(parents=True)
    (fake_repo / "models" / "hf_base" / "Qwen2.5-0.5B-Instruct" / "config.json").write_text("{}")
    (fake_repo / "data" / "processed").mkdir(parents=True)
    (fake_repo / "data" / "processed" / ".gitkeep").write_text("")
    (fake_repo / "data" / "processed" / "train.jsonl").write_text("{}\n")

    monkeypatch.setattr(cleanup, "BASE_DIR", fake_repo)

    # Dry-run preserves everything
    monkeypatch.setattr(sys, "argv", ["cleanup.py", "--dry-run"])
    cleanup.main()
    assert (fake_repo / "models" / "ollama_gguf" / "v2.gguf").exists()
    assert (fake_repo / "models" / "ollama_merged").exists()

    # Execute default cleanup removes build dirs while keeping hf_base and data
    monkeypatch.setattr(sys, "argv", ["cleanup.py", "--yes"])
    cleanup.main()
    assert not (fake_repo / "checkpoints").exists()
    assert not (fake_repo / "models" / "ollama_lora").exists()
    assert not (fake_repo / "models" / "ollama_merged").exists()
    assert not (fake_repo / "models" / "ollama_gguf").exists()
    assert (fake_repo / "models" / "hf_base" / "Qwen2.5-0.5B-Instruct" / "config.json").exists()
    assert (fake_repo / "data" / "processed" / "train.jsonl").exists()
    assert (fake_repo / "data" / "processed" / ".gitkeep").exists()
