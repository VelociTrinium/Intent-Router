"""
Canonical intent specification, dataset loaders, probability extractors,
and conversation formatting helpers shared across training, evaluation, and serving.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

CLASSES: List[str] = ["GENERAL_CHAT", "DB_READ", "DB_WRITE"]

ROUTE_TO_TOKEN: Dict[str, str] = {
    "GENERAL_CHAT": "CHAT",
    "DB_READ": "READ",
    "DB_WRITE": "WRITE",
}

TOKEN_TO_ROUTE: Dict[str, str] = {
    "CHAT": "GENERAL_CHAT",
    "READ": "DB_READ",
    "WRITE": "DB_WRITE",
}

OLLAMA_CLASSIFIER_SYSTEM_PROMPT = (
    "Classify the user's latest prompt into exactly one token: CHAT, READ, or WRITE.\n"
    "CHAT: greetings, conceptual/SQL syntax help, architecture questions, or drafting text without saving to DB.\n"
    "READ: fetch, search, filter, count, aggregate, or inspect live, deleted, or audit DB records.\n"
    "WRITE: create, insert, update, delete, log, schedule, mutate DB state, or mixed read+write requests."
)


def normalize_chat_history(
    chat_history: Optional[List[Dict[str, Any]]],
    max_turns: int = 4,
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
    max_turns: int = 4,
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
    Extracts calibrated 3-class probabilities from an Ollama `/api/chat` response
    that requested `logprobs=True, top_logprobs=...` on single-token outputs (CHAT/READ/WRITE).
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
    for token, route in TOKEN_TO_ROUTE.items():
        if content == token or content.startswith(token) or content == route:
            out = {c: 0.05 for c in CLASSES}
            out[route] = 0.90
            return out

    return None


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
