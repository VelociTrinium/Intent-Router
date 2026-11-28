"""
Canonical intent specification, dataset loaders, probability extractors,
guardrail decision rules, and conversation formatting helpers shared across
training, evaluation, and serving.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASSES: List[str] = ["GENERAL_CHAT", "DB_READ", "DB_WRITE", "CLARIFICATION_REQUIRED"]

ROUTE_TO_TOKEN: Dict[str, str] = {
    "GENERAL_CHAT": "CHAT",
    "DB_READ": "READ",
    "DB_WRITE": "WRITE",
    "CLARIFICATION_REQUIRED": "ASK",
}

TOKEN_TO_ROUTE: Dict[str, str] = {tok: route for route, tok in ROUTE_TO_TOKEN.items()}

OLLAMA_CLASSIFIER_SYSTEM_PROMPT = (
    "Route the prompt to one token: CHAT, READ, WRITE, or ASK.\n"
    "CHAT: chat, concepts, SQL/sample queries, live-data meta, notices, or drafting without DB actions.\n"
    "READ: clear DB fetch/search/count/audit queries or negative-write guards.\n"
    "WRITE: clear non-destructive DB writes with targets/values, complete read+write, or full inbound inquiries.\n"
    "ASK: missing target/value, unresolved pronouns, read-vs-write ambiguity, or destructive/bulk ops."
)

# Post-model safety net: only triggers when the model/threshold pipeline outputs DB_WRITE
# for an unambiguous bulk/destructive mutation without a negative-write guard.
_DESTRUCTIVE_BULK_WRITE_RE = re.compile(
    r"\b("
    r"delete\s+(?:all|every)\b|drop\s+(?:the\s+)?(?:\w+\s+){0,2}tables?\b|truncate\b"
    r"|purge\s+(?:all|every|old|historical)\b|wipe\s+(?:all|out)\b"
    r"|bulk[\s-]+(?:delete|remove|archive|update)|mass[\s-]+(?:delete|update|reassign|archive)"
    r"|override\s+(?:every|all)\b|overwrite\s+(?:every|all)\b"
    r"|reset\s+(?:every|all)\b|erase\s+(?:every|all)\b"
    r"|reassign\s+every\b|change\s+every\b"
    r"|update\s+(?:all|every)\b.+\bto\s+(?:zero|0|null|terminated|closed|draft|deleted)\b"
    r")\b",
    re.IGNORECASE,
)

_NEGATIVE_WRITE_GUARD_RE = re.compile(
    r"\b(?:don't|do\s+not|without|instead\s+of|hold\s+off\s+on|never|before\s+(?:deleting|purging|archiving)"
    r"|what\s+(?:is|does|are)\b|how\s+(?:do\s+i|does|to)\b|explain\b|sample\s+(?:sql\s+)?(?:query|script)|example\s+(?:sql\s+)?(?:query|script))\b",
    re.IGNORECASE,
)


def normalize_chat_history(
    chat_history: Optional[List[Dict[str, Any]]],
    max_turns: int = 6,
) -> List[Dict[str, str]]:
    """Keeps the most recent `max_turns` valid user/assistant turns with non-empty content."""
    if not chat_history:
        return []
    cleaned: List[Dict[str, str]] = []
    for turn in chat_history[-max_turns:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def build_ollama_messages(
    prompt: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    include_system: bool = False,
    system_prompt: str = OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
    max_turns: int = 6,
) -> List[Dict[str, str]]:
    """Builds Ollama `/api/chat` messages list with optional short history."""
    messages: List[Dict[str, str]] = []
    if include_system:
        messages.append({"role": "system", "content": system_prompt})
    for turn in normalize_chat_history(chat_history, max_turns=max_turns):
        messages.append(turn)
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def extract_ollama_token_probabilities(response_json: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Extracts calibrated 4-class probabilities from an Ollama `/api/chat` response
    that requested `logprobs=True, top_logprobs=...` on single-token outputs (CHAT/READ/WRITE/ASK).
    Returns None if no valid class signal could be parsed.
    """
    prob_sums: Dict[str, float] = {c: 0.0 for c in CLASSES}
    logprobs = response_json.get("logprobs") or []

    if logprobs and isinstance(logprobs, list):
        first_step = logprobs[0] or {}
        top_entries = first_step.get("top_logprobs") or []
        if not top_entries and "token" in first_step and "logprob" in first_step:
            top_entries = [{"token": first_step["token"], "logprob": first_step["logprob"]}]

        for entry in top_entries:
            tok = str(entry.get("token", "")).strip().upper()
            lp = entry.get("logprob")
            if lp is None:
                continue
            route = TOKEN_TO_ROUTE.get(tok)
            if route:
                prob_sums[route] += math.exp(float(lp))

    total = sum(prob_sums.values())
    if total > 0.0:
        return {c: round(prob_sums[c] / total, 4) for c in CLASSES}

    # Fallback: parse text content if logprobs did not surface expected tokens
    content = str((response_json.get("message") or {}).get("content", "")).strip().upper()
    other_prob = round((1.0 - 0.91) / (len(CLASSES) - 1), 4)
    for token, route in TOKEN_TO_ROUTE.items():
        if content == token or content.startswith(token) or content == route:
            out = {c: other_prob for c in CLASSES}
            out[route] = 0.91
            return out

    return None


def is_destructive_bulk_write(prompt: Optional[str]) -> bool:
    """Returns True if prompt clearly expresses an unguarded bulk or destructive DB mutation."""
    if not prompt:
        return False
    return bool(_DESTRUCTIVE_BULK_WRITE_RE.search(prompt) and not _NEGATIVE_WRITE_GUARD_RE.search(prompt))


def apply_routing_guardrails(
    probabilities: Dict[str, float],
    confidence_threshold: float = 0.60,
    write_threshold: float = 0.40,
    clarify_threshold: float = 0.40,
    fallback_route: str = "CLARIFICATION_REQUIRED",
    prompt: Optional[str] = None,
) -> Tuple[str, str, float, bool]:
    """
    Applies 4-class threshold guardrails and returns
    (final_route, raw_prediction, final_confidence, fallback_triggered).

    Precedence:
    1. Clarification Priority: if P(CLARIFICATION_REQUIRED) >= clarify_threshold
       and P(CLARIFICATION_REQUIRED) >= P(DB_WRITE), route to CLARIFICATION_REQUIRED
       so incomplete/destructive writes still ask for clarification.
    2. Write Sensitivity: else if P(DB_WRITE) >= write_threshold, route to DB_WRITE
       so mixed READ+WRITE or actionable mutations are not dropped.
    3. Confidence Fallback: else if max(P) < confidence_threshold, route to fallback_route.
    4. Primary Prediction: else route to raw argmax class.
    5. Post-Model Destructive Write Guard: if final_route is DB_WRITE and prompt matches
       an unguarded bulk/destructive mutation, intercept to CLARIFICATION_REQUIRED.
    """
    raw_prediction = max(
        ((c, probabilities.get(c, 0.0)) for c in CLASSES),
        key=lambda kv: kv[1],
    )[0]
    raw_confidence = probabilities.get(raw_prediction, 0.0)
    write_prob = probabilities.get("DB_WRITE", 0.0)
    clarify_prob = probabilities.get("CLARIFICATION_REQUIRED", 0.0)

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

    if final_route == "DB_WRITE" and is_destructive_bulk_write(prompt):
        final_route = "CLARIFICATION_REQUIRED"
        final_confidence = probabilities.get("CLARIFICATION_REQUIRED", raw_confidence)
        fallback_triggered = True

    return final_route, raw_prediction, final_confidence, fallback_triggered


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Loads JSONL records preserving optional `chat_history`."""
    rows: List[Dict[str, Any]] = []
    if not file_path.exists():
        return rows
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({
                "text": str(item["text"]),
                "label": str(item["label"]),
                "chat_history": item.get("chat_history") or [],
            })
    return rows


def load_seed_dataset(seed_file: Path) -> List[Dict[str, Any]]:
    """Loads seed_prompts.json into the same canonical record format."""
    with open(seed_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows: List[Dict[str, Any]] = []
    for intent, examples in data.items():
        if intent not in CLASSES:
            continue
        for ex in examples:
            rows.append({"text": str(ex), "label": intent, "chat_history": []})
    return rows
