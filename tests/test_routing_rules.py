"""
Unit tests for routing rules:
- Fallback-to-GENERAL_CHAT when max confidence < CONFIDENCE_THRESHOLD (0.60)
- Write-threshold logic when WRITE probability >= WRITE_THRESHOLD (0.40)
- Normal top-probability predictions
"""

import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.router_engine import RouterEngine, BaseRouter
from app.schemas import ChatMessage

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class MockRouter(BaseRouter):
    """Predictable mock router for deterministic unit testing of decision rules."""
    def __init__(self, probabilities: Dict[str, float]):
        self._probabilities = probabilities

    def set_probabilities(self, probabilities: Dict[str, float]):
        self._probabilities = probabilities

    def predict_probabilities(self, text: str) -> Dict[str, float]:
        return self._probabilities

    def is_ready(self) -> bool:
        return True


def get_test_engine():
    engine = RouterEngine()
    mock_router = MockRouter({"READ": 0.33, "WRITE": 0.33, "GENERAL_CHAT": 0.34})
    engine.router = mock_router
    return engine, mock_router


if HAS_PYTEST:
    @pytest.fixture
    def test_engine():
        """Initializes a RouterEngine equipped with a MockRouter."""
        return get_test_engine()


def test_write_threshold_triggered(test_engine=None):
    """
    Test that when WRITE probability >= WRITE_THRESHOLD (0.40),
    the router always prioritizes WRITE even if READ has a slightly higher or equal value.
    """
    engine, mock = test_engine or get_test_engine()
    # WRITE is 0.42 (>= 0.40), READ is 0.45
    mock.set_probabilities({"READ": 0.45, "WRITE": 0.42, "GENERAL_CHAT": 0.13})

    result = engine.route("Update customer profile record")
    assert result.intent == "WRITE"
    assert "write_threshold_triggered" in result.routed_by_rule
    assert result.confidence == 0.42


def test_fallback_to_general_chat_when_uncertain(test_engine=None):
    """
    Test that when max confidence < CONFIDENCE_THRESHOLD (0.60)
    and WRITE probability < WRITE_THRESHOLD (0.40),
    the system falls back safely to GENERAL_CHAT.
    """
    engine, mock = test_engine or get_test_engine()
    # No class reaches 0.60, WRITE is 0.25 (< 0.40)
    mock.set_probabilities({"READ": 0.55, "WRITE": 0.25, "GENERAL_CHAT": 0.20})

    result = engine.route("Something vague or ambiguous")
    assert result.intent == "GENERAL_CHAT"
    assert "fallback_confidence_too_low" in result.routed_by_rule
    assert result.raw_intent == "READ"


def test_high_confidence_read(test_engine=None):
    """
    Test high-confidence READ intent (confidence >= 0.60 and WRITE < 0.40).
    """
    engine, mock = test_engine or get_test_engine()
    mock.set_probabilities({"READ": 0.88, "WRITE": 0.05, "GENERAL_CHAT": 0.07})

    result = engine.route("Show me all orders from yesterday")
    assert result.intent == "READ"
    assert result.confidence == 0.88
    assert result.routed_by_rule == "model_prediction"


def test_high_confidence_general_chat(test_engine=None):
    """
    Test high-confidence GENERAL_CHAT intent (confidence >= 0.60 and WRITE < 0.40).
    """
    engine, mock = test_engine or get_test_engine()
    mock.set_probabilities({"READ": 0.05, "WRITE": 0.05, "GENERAL_CHAT": 0.90})

    result = engine.route("Good morning! How are you doing today?")
    assert result.intent == "GENERAL_CHAT"
    assert result.confidence == 0.90
    assert result.routed_by_rule == "model_prediction"


def test_chat_history_inclusion(test_engine=None):
    """
    Test that chat history is concatenated into routing context.
    """
    engine, mock = test_engine or get_test_engine()
    mock.set_probabilities({"READ": 0.10, "WRITE": 0.85, "GENERAL_CHAT": 0.05})

    history = [
        ChatMessage(role="user", content="I want to cancel my subscription."),
        ChatMessage(role="assistant", content="Are you sure you want to proceed?")
    ]
    result = engine.route("Yes, please confirm cancellation", chat_history=history)
    assert result.intent == "WRITE"
    assert result.confidence == 0.85


if __name__ == "__main__":
    print("Running routing rules unit tests directly...")
    test_write_threshold_triggered()
    test_fallback_to_general_chat_when_uncertain()
    test_high_confidence_read()
    test_high_confidence_general_chat()
    test_chat_history_inclusion()
    print("All routing rule unit tests passed successfully!")

