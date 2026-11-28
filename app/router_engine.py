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

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import settings
from app.intent_spec import (
    CLASSES,
    OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
    apply_routing_guardrails,
    build_ollama_messages,
    extract_ollama_token_probabilities,
    normalize_chat_history,
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
        cache_size: Optional[int] = None,
    ):
        self.backend_name = "ollama"
        self.router = router or OllamaRouter(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=ollama_model_override or settings.OLLAMA_MODEL,
            timeout_sec=settings.OLLAMA_TIMEOUT_SEC,
        )
        # Enable LRU cache by default for live OllamaRouter; keep 0 when a mock router is injected unless set
        if cache_size is None:
            self.cache_size = settings.ROUTE_CACHE_SIZE if router is None else 0
        else:
            self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[Tuple[Any, ...], Dict[str, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _make_cache_key(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Any, ...]:
        turns = normalize_chat_history(chat_history, max_turns=6)
        hist_key = tuple((t["role"], t["content"]) for t in turns)
        return (hist_key, prompt.strip())

    def route(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RouteResponse:
        t0 = time.perf_counter()

        probabilities: Optional[Dict[str, float]] = None
        cache_key = self._make_cache_key(prompt, chat_history) if self.cache_size > 0 else None
        if cache_key is not None:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    probabilities = dict(cached)

        if probabilities is None:
            probabilities = self.router.predict_probabilities(prompt, chat_history=chat_history)
            if cache_key is not None:
                with self._cache_lock:
                    self._cache[cache_key] = dict(probabilities)
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)

        final_route, raw_prediction, final_confidence, fallback_triggered = apply_routing_guardrails(
            probabilities=probabilities,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            write_threshold=settings.WRITE_THRESHOLD,
            clarify_threshold=settings.CLARIFY_THRESHOLD,
            fallback_route=settings.FALLBACK_ROUTE,
            prompt=prompt,
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
