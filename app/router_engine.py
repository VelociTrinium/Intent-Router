"""
Unified Router Engine for 3-class NLP intent classification.
Supports:
1. OllamaRouter (Fine-tuned 1-token Ollama classifier with logprobs calibration, or legacy JSON schema)
2. SetFitRouter (Local sentence-transformers backbone, e.g., BAAI/bge-small-en-v1.5)
3. BaselineRouter (TF-IDF + LogisticRegression)
4. Heuristic Fallback (Rule-based safeguard)

Routes:
- GENERAL_CHAT: General conversation, conceptual questions, syntax help, drafting without DB save.
- DB_READ: Fetch, search, filter, count, aggregate, or inspect live/deleted/audit records.
- DB_WRITE: Insert, update, delete, modify, log, schedule, or mixed read+write DB actions.
"""

import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from app.config import settings
from app.schemas import RouteResponse, RouteName
from app.intent_spec import (
    CLASSES,
    OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
    build_ollama_messages,
    extract_ollama_token_probabilities,
    format_text_with_history,
)

logger = logging.getLogger(__name__)


def normalize_probabilities(prob_dict: Dict[str, float]) -> Dict[str, float]:
    """Ensures all 3 classes exist, non-negative, and sum to 1.0."""
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
        out = {"GENERAL_CHAT": 0.3334, "DB_READ": 0.3333, "DB_WRITE": 0.3333}
    return out


def heuristic_fallback(text: str) -> Dict[str, float]:
    """
    Lightweight rule-based fallback distinguishing tricky boundary cases:
    - 'How do I delete a row?' -> GENERAL_CHAT (instructional/syntax)
    - 'Show deleted rows' -> DB_READ (read query)
    - 'Delete row 42' / mixed READ+WRITE -> DB_WRITE (mutation)
    """
    lower = text.strip().lower()

    chat_patterns = [
        "how do i ", "how to ", "how can i ", "explain ", "what is ", "what does ",
        "difference between", "tell me about", "give me an example", "write me an example",
        "sample query", "sql syntax", "can you draft", "draft a ", "template for", "hello", "hi "
    ]
    is_chat = any(p in lower for p in chat_patterns)

    write_keywords = [
        "delete", "remove", "insert", "create", "update", "modify",
        "add", "drop", "alter", "truncate", "schedule a meeting", "log a call",
        "mark activity", "mark deal", "register a new", "archive"
    ]

    read_keywords = [
        "show", "list", "find", "get", "fetch", "search", "lookup", "count",
        "display", "how many", "which accounts", "total closed", "all data",
        "details of", "website of", "primary contact", "inspect"
    ]

    if is_chat and ("query" in lower or "how" in lower or "explain" in lower or "draft" in lower):
        return {"GENERAL_CHAT": 0.85, "DB_READ": 0.10, "DB_WRITE": 0.05}

    if "show deleted" in lower or "list deleted" in lower or "find removed" in lower:
        return {"GENERAL_CHAT": 0.05, "DB_READ": 0.90, "DB_WRITE": 0.05}

    write_score = sum(1 for w in write_keywords if w in lower)
    read_score = sum(1 for r in read_keywords if r in lower)

    # Mixed READ+WRITE prioritizes DB_WRITE
    if write_score > 0:
        return {"GENERAL_CHAT": 0.05, "DB_READ": 0.15 if read_score > 0 else 0.10, "DB_WRITE": 0.80}
    elif read_score > 0:
        return {"GENERAL_CHAT": 0.10, "DB_READ": 0.85, "DB_WRITE": 0.05}
    elif is_chat:
        return {"GENERAL_CHAT": 0.80, "DB_READ": 0.10, "DB_WRITE": 0.10}
    else:
        return {"GENERAL_CHAT": 0.60, "DB_READ": 0.25, "DB_WRITE": 0.15}


class BaseRouter(ABC):
    @abstractmethod
    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Returns normalized probability distribution across CLASSES."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the backend is operational and ready to serve."""
        pass


class SetFitRouter(BaseRouter):
    """
    SetFit few-shot router backed by compact Sentence Transformers
    (e.g., BAAI/bge-small-en-v1.5) with calibrated class probabilities.
    """
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.id_to_label: Optional[Dict[int, str]] = None
        self._load()

    def _load(self):
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            try:
                from setfit import SetFitModel
                try:
                    self.model = SetFitModel.from_pretrained(str(self.model_dir), local_files_only=True)
                except Exception:
                    self.model = SetFitModel.from_pretrained(str(self.model_dir))

                label_map_path = self.model_dir / "label_map.json"
                if label_map_path.exists():
                    with open(label_map_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        raw_map = data.get("id_to_label", {})
                        self.id_to_label = {int(k): v for k, v in raw_map.items()}
                logger.info(f"Loaded SetFit model from {self.model_dir}")
            except Exception as e:
                logger.warning(f"Failed to load SetFit model from {self.model_dir}: {e}")
                self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        if not self.is_ready():
            return heuristic_fallback(prompt)

        input_text = format_text_with_history(prompt, chat_history)

        try:
            probs = self.model.predict_proba([input_text])[0]
            if hasattr(probs, "tolist"):
                probs = probs.tolist()

            prob_dict = {}
            if self.id_to_label:
                for idx, p in enumerate(probs):
                    label = self.id_to_label.get(idx, f"CLASS_{idx}")
                    prob_dict[label] = float(p)
            elif hasattr(self.model, "labels") and self.model.labels:
                for idx, label in enumerate(self.model.labels):
                    prob_dict[label] = float(probs[idx]) if idx < len(probs) else 0.0
            else:
                for idx, c in enumerate(CLASSES):
                    prob_dict[c] = float(probs[idx]) if idx < len(probs) else 0.0

            return normalize_probabilities(prob_dict)
        except Exception as e:
            logger.warning(f"SetFit inference error: {e}. Falling back to heuristic.")
            return heuristic_fallback(prompt)


class BaselineRouter(BaseRouter):
    """
    Lightweight TF-IDF + Logistic Regression baseline router for benchmark comparison.
    """
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                import joblib
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Loaded baseline model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load baseline model: {e}")
                self.pipeline = None

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        if not self.is_ready():
            return heuristic_fallback(prompt)

        input_text = format_text_with_history(prompt, chat_history)
        try:
            classes = list(self.pipeline.classes_)
            probs = self.pipeline.predict_proba([input_text])[0]
            prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
            return normalize_probabilities(prob_dict)
        except Exception as e:
            logger.warning(f"Baseline inference error: {e}. Falling back to heuristic.")
            return heuristic_fallback(prompt)


class OllamaRouter(BaseRouter):
    """
    Ollama runtime backend supporting:
    - 'single_token' mode (default): fine-tuned 1-token classifier (CHAT/READ/WRITE)
      with exact token `logprobs` probability extraction and optional chat_history.
    - 'json_schema' mode: legacy structured JSON probability output for untuned models.
    """
    def __init__(
        self,
        base_url: str,
        model_name: str,
        mode: str = "single_token",
        timeout_sec: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.mode = mode
        self.timeout_sec = timeout_sec
        self._client = httpx.Client(timeout=timeout_sec) if HAS_HTTPX else None

    def is_ready(self) -> bool:
        if not self._client:
            return False
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=2.0)
            if r.status_code != 200:
                return False
            models = [m.get("name", "") for m in r.json().get("models", [])]
            target = self.model_name
            return any(m == target or (":" not in target and m == f"{target}:latest") for m in models)
        except Exception:
            return False

    def _predict_single_token(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        # Fine-tuned Modelfile bakes SYSTEM in; include system explicitly for other models
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
            logger.warning("Could not parse token probabilities from Ollama response. Falling back to heuristic.")
            return heuristic_fallback(prompt)
        return normalize_probabilities(extracted)

    def _predict_json_schema(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        system_instruction = (
            "You are a strict 3-class NLP intent router for a database assistant.\n"
            "Classify the incoming user prompt into:\n"
            "- GENERAL_CHAT: General conversation, conceptual questions, coding/SQL syntax help, "
            "or drafting text without saving to DB.\n"
            "- DB_READ: Requests to fetch, search, filter, count, aggregate, or inspect live/deleted records.\n"
            "- DB_WRITE: Requests to insert, update, delete, log, schedule, or mixed read+write DB actions.\n\n"
            "Output calibrated probabilities summing to 1.0."
        )
        json_schema = {
            "type": "object",
            "properties": {
                "GENERAL_CHAT": {"type": "number"},
                "DB_READ": {"type": "number"},
                "DB_WRITE": {"type": "number"}
            },
            "required": ["GENERAL_CHAT", "DB_READ", "DB_WRITE"]
        }
        messages = build_ollama_messages(
            prompt=prompt,
            chat_history=chat_history,
            include_system=True,
            system_prompt=system_instruction,
        )
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": json_schema,
            "options": {"temperature": 0.0}
        }
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        return normalize_probabilities(parsed)

    def predict_probabilities(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        if not self._client:
            return heuristic_fallback(prompt)

        try:
            if self.mode == "json_schema":
                return self._predict_json_schema(prompt, chat_history)
            return self._predict_single_token(prompt, chat_history)
        except Exception as e:
            logger.warning(f"Ollama routing request failed ({e}). Falling back to heuristic.")
            return heuristic_fallback(prompt)


class RouterEngine:
    """
    Unified router engine applying threshold-based guardrails:
    1. Write Sensitivity Guardrail: probability(DB_WRITE) >= WRITE_THRESHOLD -> DB_WRITE
    2. Confidence Fallback Guardrail: max(probabilities) < CONFIDENCE_THRESHOLD -> FALLBACK_ROUTE (GENERAL_CHAT)
    3. Normal Prediction: Top probability class
    """
    def __init__(
        self,
        backend_override: Optional[str] = None,
        ollama_model_override: Optional[str] = None,
        ollama_mode_override: Optional[str] = None,
    ):
        self.backend_name = backend_override or settings.ROUTER_BACKEND
        self.ollama_model_override = ollama_model_override
        self.ollama_mode_override = ollama_mode_override
        self.router: BaseRouter = self._init_router(self.backend_name)

    def _init_router(self, backend: str) -> BaseRouter:
        backend_lower = backend.lower()
        if backend_lower == "setfit":
            return SetFitRouter(settings.MODEL_PATH)
        elif backend_lower == "ollama":
            return OllamaRouter(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=self.ollama_model_override or settings.OLLAMA_MODEL,
                mode=self.ollama_mode_override or settings.OLLAMA_MODE,
                timeout_sec=settings.OLLAMA_TIMEOUT_SEC,
            )
        elif backend_lower == "baseline":
            return BaselineRouter(settings.BASELINE_MODEL_PATH)
        else:
            raise ValueError(f"Unknown router backend: {backend}. Choose 'setfit', 'ollama', or 'baseline'.")

    def route(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> RouteResponse:
        t0 = time.perf_counter()

        probabilities = self.router.predict_probabilities(prompt, chat_history=chat_history)
        raw_prediction: RouteName = max(probabilities.items(), key=lambda kv: kv[1])[0]  # type: ignore
        raw_confidence = probabilities[raw_prediction]

        write_prob = probabilities.get("DB_WRITE", 0.0)
        fallback_triggered = False

        if write_prob >= settings.WRITE_THRESHOLD:
            final_route: RouteName = "DB_WRITE"
            final_confidence = write_prob
            if raw_prediction != "DB_WRITE":
                fallback_triggered = True
        elif raw_confidence < settings.CONFIDENCE_THRESHOLD:
            final_route: RouteName = settings.FALLBACK_ROUTE  # type: ignore
            final_confidence = probabilities.get(final_route, raw_confidence)
            fallback_triggered = True
        else:
            final_route = raw_prediction
            final_confidence = raw_confidence
            fallback_triggered = False

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return RouteResponse(
            route=final_route,
            raw_prediction=raw_prediction,
            confidence=round(final_confidence, 4),
            probabilities=probabilities,
            fallback_triggered=fallback_triggered,
            backend=self.backend_name,
            latency_ms=round(elapsed_ms, 2)
        )
