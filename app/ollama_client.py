"""
Standalone, zero-dependency Python client for calling the fine-tuned `intent-router`
Ollama model directly from another project.

Copy this single file into your separate project — uses only Python's standard library
(`urllib.request`, `json`, `math`, `time`).

Usage:
    from ollama_client import classify_intent

    res = classify_intent(
        prompt="Update Kevin's phone number to +1 214 555 8721",
        chat_history=[
            {"role": "user", "content": "Show Kevin Sauer at Microsoft"},
            {"role": "assistant", "content": "Found Kevin Sauer at Microsoft."},
        ],
    )
    print(res["route"], res["confidence"], res["probabilities"])
"""

import json
import math
import time
import urllib.request
from typing import Any, Dict, List, Optional

CLASSES = ("GENERAL_CHAT", "DB_READ", "DB_WRITE", "CLARIFICATION_REQUIRED")
TOKEN_TO_ROUTE = {
    "CHAT": "GENERAL_CHAT",
    "READ": "DB_READ",
    "WRITE": "DB_WRITE",
    "ASK": "CLARIFICATION_REQUIRED",
}


def classify_intent(
    prompt: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    model: str = "intent-router-qwen2.5-0.5b-instruct:v1",
    version: Optional[str] = None,
    base_url: str = "http://127.0.0.1:11434",
    write_threshold: float = 0.40,
    clarify_threshold: float = 0.40,
    confidence_threshold: float = 0.60,
    fallback_route: str = "CLARIFICATION_REQUIRED",
    timeout_sec: float = 15.0,
    max_history_turns: int = 4,
    keep_alive: str = "10m",
) -> Dict[str, Any]:
    """
    Calls Ollama `/api/chat` on the fine-tuned 1-token `intent-router` model (CHAT/READ/WRITE/ASK),
    extracts calibrated 4-class probabilities from token `logprobs`, and applies
    clarification-priority, write-sensitivity, and confidence-fallback guardrails.
    """
    t0 = time.perf_counter()

    if version:
        clean_ver = version.strip().lstrip(":")
        base_name = model.split(":", 1)[0]
        model_ref = f"{base_name}:{clean_ver}" if clean_ver else base_name
    else:
        model_ref = model

    messages: List[Dict[str, str]] = []
    if chat_history:
        for turn in chat_history[-max_history_turns:]:
            if isinstance(turn, dict):
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", "")).strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": model_ref,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "logprobs": True,
        "top_logprobs": 20,
        "options": {
            "temperature": 0.0,
            "num_predict": 1,
        },
    }

    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Extract calibrated 4-class probabilities from first generated token logprobs
    prob_sums: Dict[str, float] = {c: 0.0 for c in CLASSES}
    logprobs = data.get("logprobs") or []
    if logprobs and isinstance(logprobs, list):
        first_step = logprobs[0] or {}
        top_entries = first_step.get("top_logprobs") or []
        if not top_entries and "token" in first_step and "logprob" in first_step:
            top_entries = [{"token": first_step["token"], "logprob": first_step["logprob"]}]
        for entry in top_entries:
            tok = str(entry.get("token", "")).strip().upper()
            lp = entry.get("logprob")
            route = TOKEN_TO_ROUTE.get(tok)
            if route and lp is not None:
                prob_sums[route] += math.exp(float(lp))

    total = sum(prob_sums.values())
    if total > 0.0:
        probabilities = {c: round(prob_sums[c] / total, 4) for c in CLASSES}
    else:
        content = str((data.get("message") or {}).get("content", "")).strip().upper()
        matched = next(
            (route for tok, route in TOKEN_TO_ROUTE.items() if content == tok or content.startswith(tok) or content == route),
            fallback_route,
        )
        other_prob = round((1.0 - 0.91) / (len(CLASSES) - 1), 4)
        probabilities = {c: (0.91 if c == matched else other_prob) for c in CLASSES}

    raw_prediction = max(((c, probabilities[c]) for c in CLASSES), key=lambda kv: kv[1])[0]
    raw_confidence = probabilities[raw_prediction]
    write_prob = probabilities["DB_WRITE"]
    clarify_prob = probabilities["CLARIFICATION_REQUIRED"]

    if clarify_prob >= clarify_threshold and clarify_prob >= write_prob:
        final_route = "CLARIFICATION_REQUIRED"
        final_confidence = clarify_prob
        fallback_triggered = raw_prediction != "CLARIFICATION_REQUIRED"
    elif write_prob >= write_threshold:
        final_route = "DB_WRITE"
        final_confidence = write_prob
        fallback_triggered = raw_prediction != "DB_WRITE"
    elif raw_confidence < confidence_threshold:
        final_route = fallback_route
        final_confidence = probabilities.get(final_route, raw_confidence)
        fallback_triggered = True
    else:
        final_route = raw_prediction
        final_confidence = raw_confidence
        fallback_triggered = False

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "route": final_route,
        "raw_prediction": raw_prediction,
        "confidence": round(final_confidence, 4),
        "probabilities": probabilities,
        "fallback_triggered": fallback_triggered,
        "model": model_ref,
        "latency_ms": elapsed_ms,
    }


if __name__ == "__main__":
    for sample in [
        "How do I delete row 42 in Postgres?",
        "Show deleted rows from the archive table",
        "Update Kevin Sauer's mobile number at Microsoft to +1 214 555 8721.",
        "Update Kevin's phone number",
    ]:
        print(sample, "->", classify_intent(sample))
