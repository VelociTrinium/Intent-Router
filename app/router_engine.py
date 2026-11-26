"""
Ollama Router Engine for 4-class NLP intent classification.
Uses the fine-tuned 1-token Qwen2.5-0.5B-Instruct Ollama model (CHAT/READ/WRITE/ASK)
with exact token `logprobs` probability calibration and safety guardrails.

Routes:
- GENERAL_CHAT: General conversation, conceptual questions, syntax help, drafting without DB save.
- DB_READ: Clear, specific requests to fetch, search, filter, count, aggregate, or inspect records.
- DB_WRITE: Clear, complete requests to insert, update, log, schedule, or mixed read+write actions.
- CLARIFICATION_REQUIRED: Vague, incomplete, pronoun-dependent, ambiguous, or destructive prompts needing clarification.
"""

import time
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.intent_spec import (
    CLASSES,
    OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
    apply_routing_guardrails,
    build_ollama_messages,
    extract_ollama_token_probabilities,
)
from app.schemas import RouteName, RouteResponse


def normalize_probabilities(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """Ensures all 4 classes exist, are non-negative, and sum to 1.0."""
    out = {}
    total = 0.0
    for c in CLASSES:
        val = max(0.0, float(prob_dict.get(c, 0.0)))
        out[c] = val
        total += val

    if total > 0.0:
        for c in CLASSES:
            out[c] = round(out[c] / total, 4)
    else:
        uniform = round(1.0 / len(CLASSES), 4)
        out = {c: uniform for c in CLASSES}
    return out


class OllamaRouter:
    """
    Ollama runtime client for the fine-tuned 1-token classifier (CHAT/READ/WRITE/ASK)
    with exact token `logprobs` probability extraction and optional chat_history.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_sec: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self._client = httpx.Client(timeout=timeout_sec)

    def is_ready(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            models = [m.get("name", "") for m in r.json().get("models", [])]
            target = self.model_name
            return any(m == target or (":" not in target and m == f"{target}:latest") for m in models)
        except Exception:
            return False

    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        # Fine-tuned Modelfile bakes SYSTEM in; include system explicitly for untuned base models
        include_system = not self.model_name.startswith("intent-router")
        messages = build_ollama_messages(
            prompt=prompt,
            chat_history=chat_history,
            include_system=include_system,
            system_prompt=OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
        )
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "logprobs": True,
            "top_logprobs": 20,
            "options": {
                "temperature": 0.0,
                "num_predict": 1,
            },
        }
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        extracted = extract_ollama_token_probabilities(resp.json())
        if extracted is None:
            raise RuntimeError(
                f"Could not parse CHAT/READ/WRITE/ASK token probabilities from Ollama model '{self.model_name}'."
            )
        return normalize_probabilities(extracted)


class RouterEngine:
    """
    Router engine applying 4-class threshold guardrails:
    1. Clarification Priority: P(CLARIFICATION_REQUIRED) >= CLARIFY_THRESHOLD and >= P(DB_WRITE) -> CLARIFICATION_REQUIRED
    2. Write Sensitivity Guardrail: P(DB_WRITE) >= WRITE_THRESHOLD -> DB_WRITE
    3. Confidence Fallback Guardrail: max(P) < CONFIDENCE_THRESHOLD -> FALLBACK_ROUTE (CLARIFICATION_REQUIRED)
    4. Normal Prediction: Top probability class
    """

    def __init__(
        self,
        ollama_model_override: Optional[str] = None,
        router: Optional[Any] = None,
    ):
        self.backend_name = "ollama"
        self.router = router or OllamaRouter(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=ollama_model_override or settings.OLLAMA_MODEL,
            timeout_sec=settings.OLLAMA_TIMEOUT_SEC,
        )

    def route(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RouteResponse:
        t0 = time.perf_counter()

        probabilities = self.router.predict_probabilities(prompt, chat_history=chat_history)
        final_route, raw_prediction, final_confidence, fallback_triggered = apply_routing_guardrails(
            probabilities=probabilities,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            write_threshold=settings.WRITE_THRESHOLD,
            clarify_threshold=settings.CLARIFY_THRESHOLD,
            fallback_route=settings.FALLBACK_ROUTE,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return RouteResponse(
            route=final_route,  # type: ignore
            raw_prediction=raw_prediction,  # type: ignore
            confidence=round(final_confidence, 4),
            probabilities=probabilities,
            fallback_triggered=fallback_triggered,
            backend=self.backend_name,
            latency_ms=round(elapsed_ms, 2),
        )
