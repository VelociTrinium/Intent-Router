"""
Unified Router Engine.
Provides a single interface switching between SetFitRouter, OllamaRouter,
and BaselineRouter, applying threshold-based decision rules:
- Write Sensitivity Rule: probability(WRITE) >= WRITE_THRESHOLD -> WRITE
- Confidence Fallback Rule: max(probabilities) < CONFIDENCE_THRESHOLD -> GENERAL_CHAT
- Model Prediction Rule: Top probability intent
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from app.config import settings
from app.schemas import RouteResponse, ChatMessage, IntentType

CLASSES = ["READ", "WRITE", "GENERAL_CHAT"]


class BaseRouter(ABC):
    @abstractmethod
    def predict_probabilities(self, text: str) -> Dict[str, float]:
        """Returns normalized probability distribution across CLASSES."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if model weights/service are available."""
        pass


class SetFitRouter(BaseRouter):
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.id_to_label = None
        self._load()

    def _load(self):
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            try:
                from setfit import SetFitModel
                self.model = SetFitModel.from_pretrained(str(self.model_dir))
                label_map_path = self.model_dir / "label_map.json"
                if label_map_path.exists():
                    with open(label_map_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        raw_map = data.get("id_to_label", {})
                        self.id_to_label = {int(k): v for k, v in raw_map.items()}
            except Exception as e:
                print(f"[WARN] Failed to load SetFit model from {self.model_dir}: {e}")
                self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_probabilities(self, text: str) -> Dict[str, float]:
        if not self.is_ready():
            # Graceful heuristic fallback if model weights not yet trained
            return heuristic_fallback(text)

        probs = self.model.predict_proba([text])[0]
        # Map probabilities to classes
        prob_dict = {}
        if hasattr(probs, "tolist"):
            probs = probs.tolist()

        if self.id_to_label:
            for idx, p in enumerate(probs):
                label = self.id_to_label.get(idx, f"CLASS_{idx}")
                prob_dict[label] = float(p)
        else:
            for idx, c in enumerate(CLASSES):
                prob_dict[c] = float(probs[idx]) if idx < len(probs) else 0.0

        # Ensure all 3 classes exist
        return normalize_probabilities(prob_dict)


class BaselineRouter(BaseRouter):
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                import joblib
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"[WARN] Failed to load baseline model: {e}")
                self.pipeline = None

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict_probabilities(self, text: str) -> Dict[str, float]:
        if not self.is_ready():
            return heuristic_fallback(text)

        classes = list(self.pipeline.classes_)
        probs = self.pipeline.predict_proba([text])[0]
        prob_dict = {cls_name: float(prob) for cls_name, prob in zip(classes, probs)}
        return normalize_probabilities(prob_dict)


class OllamaRouter(BaseRouter):
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def is_ready(self) -> bool:
        if not HAS_HTTPX:
            return False
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    def predict_probabilities(self, text: str) -> Dict[str, float]:
        if not HAS_HTTPX:
            return heuristic_fallback(text)
        system_prompt = (
            "You are an intent classification routing module. "
            "Given a user prompt, evaluate the probability distribution over 3 classes:\n"
            "- READ: Queries, reading, lookups, search, viewing data.\n"
            "- WRITE: Mutations, creating, updating, deleting, executing actions.\n"
            "- GENERAL_CHAT: Casual conversation, greetings, trivia, open-ended.\n"
            "Respond ONLY with a JSON object with keys 'READ', 'WRITE', 'GENERAL_CHAT' summing to 1.0."
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this prompt: '{text}'"}
            ],
            "stream": False,
            "format": "json"
        }
        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=5.0)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            return normalize_probabilities(parsed)
        except Exception as e:
            print(f"[WARN] Ollama routing request failed: {e}. Using heuristic fallback.")
            return heuristic_fallback(text)


def heuristic_fallback(text: str) -> Dict[str, float]:
    """Lightweight rule-based fallback when trained models are not yet compiled."""
    lower = text.lower()
    write_keywords = ["delete", "remove", "insert", "create", "update", "modify", "add", "drop", "change", "set"]
    read_keywords = ["show", "find", "get", "fetch", "list", "search", "lookup", "what is", "where is", "read", "status"]

    write_hits = sum(1 for w in write_keywords if w in lower)
    read_hits = sum(1 for r in read_keywords if r in lower)

    if write_hits > read_hits:
        return {"READ": 0.15, "WRITE": 0.75, "GENERAL_CHAT": 0.10}
    elif read_hits > write_hits:
        return {"READ": 0.75, "WRITE": 0.10, "GENERAL_CHAT": 0.15}
    else:
        return {"READ": 0.20, "WRITE": 0.15, "GENERAL_CHAT": 0.65}


def normalize_probabilities(prob_dict: Dict[str, float]) -> Dict[str, float]:
    out = {}
    total = 0.0
    for c in CLASSES:
        val = max(0.0, float(prob_dict.get(c, 0.0)))
        out[c] = val
        total += val

    if total > 0:
        for c in CLASSES:
            out[c] = round(out[c] / total, 4)
    else:
        out = {"READ": 0.3333, "WRITE": 0.3333, "GENERAL_CHAT": 0.3334}
    return out


class RouterEngine:
    def __init__(self):
        self.backend_name = settings.ROUTER_BACKEND
        self.router: BaseRouter = self._init_router(self.backend_name)

    def _init_router(self, backend: str) -> BaseRouter:
        if backend == "setfit":
            return SetFitRouter(settings.MODEL_PATH)
        elif backend == "ollama":
            return OllamaRouter(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
        elif backend == "baseline":
            return BaselineRouter(settings.BASELINE_MODEL_PATH)
        else:
            raise ValueError(f"Unknown router backend: {backend}")

    def route(self, prompt: str, chat_history: Optional[List[ChatMessage]] = None) -> RouteResponse:
        t0 = time.perf_counter()

        # Combine conversational context if history provided
        context_text = prompt
        if chat_history:
            recent_turns = chat_history[-3:]
            history_str = " | ".join(f"{m.role}: {m.content}" for m in recent_turns)
            context_text = f"{history_str} | current: {prompt}"

        # Get probability distribution from active backend
        probabilities = self.router.predict_probabilities(context_text)

        # Raw model prediction (highest probability)
        raw_intent = max(probabilities.items(), key=lambda kv: kv[1])[0]
        confidence = probabilities[raw_intent]

        # Apply routing rules
        # Rule 1: WRITE Sensitivity Threshold
        write_prob = probabilities.get("WRITE", 0.0)
        if write_prob >= settings.WRITE_THRESHOLD:
            final_intent: IntentType = "WRITE"
            routed_by_rule = f"write_threshold_triggered (p={write_prob:.3f} >= {settings.WRITE_THRESHOLD})"
            final_confidence = write_prob

        # Rule 2: Confidence Fallback Threshold
        elif confidence < settings.CONFIDENCE_THRESHOLD:
            final_intent: IntentType = "GENERAL_CHAT"
            routed_by_rule = f"fallback_confidence_too_low (max_conf={confidence:.3f} < {settings.CONFIDENCE_THRESHOLD})"
            final_confidence = probabilities.get("GENERAL_CHAT", confidence)

        # Rule 3: Top Prediction
        else:
            final_intent: IntentType = raw_intent  # type: ignore
            routed_by_rule = "model_prediction"
            final_confidence = confidence

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return RouteResponse(
            intent=final_intent,
            confidence=round(final_confidence, 4),
            raw_intent=raw_intent,
            probabilities=probabilities,
            backend=self.backend_name,
            routed_by_rule=routed_by_rule,
            latency_ms=round(elapsed_ms, 2)
        )
