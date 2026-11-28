# Intent Router v3 — Integration & Handoff Guide

> **4-Class Low-Latency NLP Intent Classifier for MiraChat / ReConnect CRM**  
> Fine-tuned `Qwen2.5-0.5B-Instruct` (`F16` GGUF via Ollama) • Model Tag: `intent-router-qwen2.5-0.5b-instruct:v3`

| Property | Value / Specification |
|---|---|
| **Hugging Face Repository** | `VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf` (`main` branch; `v2` archived under tag `v2.0.0`) |
| **Primary Model Artifacts** | `intent-router-qwen2.5-0.5b-instruct-v3-f16.gguf` (`948.1 MiB` / `994 MB` decimal) + `Modelfile` |
| **Ollama Registration Tag** | **`intent-router-qwen2.5-0.5b-instruct:v3`** *(or `:v3.0.0`)* |
| **Base Model & Training** | `Qwen2.5-0.5B-Instruct` (ChatML, LoRA `r=24`/`alpha=48`, NEFTune `5.0`, 4-class calibrated loss on `1,534` train / `184` val rows, merged FP32 → `F16` GGUF) |
| **Output Classes (1-Token)** | `GENERAL_CHAT` (`CHAT`) \| `DB_READ` (`READ`) \| `DB_WRITE` (`WRITE`) \| `CLARIFICATION_REQUIRED` (`ASK`) |
| **Held-Out Test Benchmark** | **99.67%** Guarded Accuracy (`99.33%` Raw) \| **0.9965** Macro F1 \| **100%** `GENERAL_CHAT`, `DB_READ`, `CLARIFICATION_REQUIRED` & Multi-Turn Recall \| **98.65%** `DB_WRITE` Recall (`300` prompts, `48` multi-turn) |
| **Probability Calibration** | **NLL: `0.0363`** \| **Brier: `0.0108`** |
| **Local Inference Latency** | **163.40 ms** mean \| **127.23 ms** P50 median \| **347.51 ms** P95 (local CPU, single-token output) |

---

## What Changed in `v3` (vs. `v2`)

1. **Higher Accuracy & 100% `GENERAL_CHAT` Recall (`90.48%` → `100.0%`):** Fixed all `v2` misclassifications (live-data meta-questions, non-DB drafting/summaries, sample SQL generation, internal office notices) and expanded coverage to out-of-distribution everyday prompts (`1,953` unique records across splits with `0` train–test leakage).
2. **Better Probability Calibration (`NLL = 0.0363`, `Brier = 0.0108`):** Trained strictly on the single decision token using a hybrid `0.40 * CE_vocab + 0.60 * CE_4class` loss (`class_label_smoothing=0.03`) and `NEFTune` (`5.0`), eliminating overconfident `1.0` mispredictions while preserving pretrained intuition.
3. **Faster CPU Inference (`151.96 ms` → `127.23 ms` P50):** Compressed the baked-in `SYSTEM` prompt, capped `num_ctx 768` and `num_thread 6` in `Modelfile`, and added a thread-safe `ROUTE_CACHE_SIZE=512` LRU cache in `RouterEngine`.
4. **Expanded Multi-Turn Window & Post-Model Bulk-Write Guard:** Increased `chat_history` window from `4` → **`6` turns**, and added a post-model destructive/bulk-write safety check (Rule 5) in both `app/intent_spec.py` and `app/ollama_client.py`.

---

## 1. Quick-Start Deployment (Ollama Setup in 3 Steps)

This classifier runs as a local **Ollama** model outputting strictly **1 token** per request (`num_predict: 1`). Ensure Ollama is installed and running on `http://127.0.0.1:11434` (`ollama serve`) with at least ~1.2 GB free RAM/VRAM before continuing.

### Step 1: Download the `v3` GGUF & `Modelfile` from Hugging Face (`main`)

```bash
pip install -U huggingface_hub
huggingface-cli download VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf \
  intent-router-qwen2.5-0.5b-instruct-v3-f16.gguf Modelfile \
  --local-dir ./intent-router-v3
```
*(Use `hf download ...` instead of `huggingface-cli download ...` if running `huggingface_hub >= 1.0`. Download from default `main` — do not pass `--revision v3.0.0`, as `main` contains the patched `Modelfile` and `README.md`.)*

### Step 2: Register the Model in Ollama as `:v3`

```bash
cd intent-router-v3
ollama create intent-router-qwen2.5-0.5b-instruct:v3 -f Modelfile
ollama list
```

> [!WARNING]
> **Runtime Tag & `options` Note:** Update `.env` (`OLLAMA_MODEL=intent-router-qwen2.5-0.5b-instruct:v3`) or pass `model="intent-router-qwen2.5-0.5b-instruct:v3"` explicitly to `classify_intent()`. Legacy repo defaults in `.env.example` / `ollama_client.py` still default to `:v1` unless overridden. Also avoid passing `num_ctx` or `num_thread` inside per-request `options` (both are baked into `Modelfile`; changing them per request forces a ~1.2 s model reload in Ollama).

### Step 3: Smoke-Test Live Inference with `logprobs`

```bash
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "intent-router-qwen2.5-0.5b-instruct:v3",
  "messages": [{"role": "user", "content": "Update Kevin phone number"}],
  "stream": false,
  "keep_alive": "10m",
  "logprobs": true,
  "top_logprobs": 20,
  "options": {"temperature": 0, "num_predict": 1}
}'
```
*(Expect single token `"ASK"` with a populated `logprobs` array.)*

---

## 2. Intent Taxonomy & Downstream Routing Contract

Every prompt (with optional multi-turn `chat_history`) is classified into one of four mutually exclusive classes. The model emits a single uppercase token mapped back to the canonical API route below:

| Route Name | Token (ID) | Scope & Examples | Expected Downstream Action |
|---|---|---|---|
| `GENERAL_CHAT` | `CHAT` (`60717`) | Greetings, capability/live-data meta-questions, ReConnect workflow how-tos, CRE conceptual terms (cap rate, WALE, Grade A), SQL syntax/sample queries, office/HR notices, drafting/summarizing without DB actions, general non-CRM chat. | Send to conversational assistant / general LLM flow. Do **not** query or mutate the CRM database. |
| `DB_READ` | `READ` (`9560`) | Clear, specific queries to list, search, filter, count, aggregate, compare, or inspect live/deleted/audit records; explicit negative-write guards (*"don't update — just check pursuit #7724"*). | Send to read-only NL-to-SQL / CRM lookup pipeline. |
| `DB_WRITE` | `WRITE` (`32781`) | Clear, non-destructive CRM writes with explicit entity & values: creating records, updating fields to concrete values, logging calls/notes/emails, scheduling meetings, complete mixed `READ+WRITE`, or full inbound client inquiries. | Send to CRM write / mutation agent or confirmation-ready action builder. |
| `CLARIFICATION_REQUIRED` | `ASK` (`7384`) | Vague/broad prompts (*"all data."*), missing target or new value (*"Update Kevin's phone number"*, *"Add Sarah"*), unresolved pronouns without history (*"Update it"*), read-vs-write ambiguity (*"Pursuit #8436"*), or destructive/bulk deletes/drops/purges/mass-overrides. | Do **not** execute DB read/write. Prompt the user to supply missing details, clarify intent, or explicitly confirm destructive action. |

---

## 3. Integration Options for Your Application

### Option A: Zero-Dependency Python Client (Fastest Drop-In)

Copy `app/ollama_client.py` directly into your Python project. It uses only the Python standard library (`urllib.request`, `json`, `math`, `re`, `time`), calls Ollama `/api/chat` with `logprobs=True` and `num_predict=1`, normalizes token probabilities across the 4 classes, and enforces all 5 guardrail rules (including the post-model bulk/destructive-write guard).

```python
from ollama_client import classify_intent

# Example 1: Single-turn incomplete write -> CLARIFICATION_REQUIRED
r1 = classify_intent(
    prompt="Update Kevin's phone number",
    model="intent-router-qwen2.5-0.5b-instruct:v3",
)
# -> r1["route"] == "CLARIFICATION_REQUIRED"

# Example 2: Multi-turn resolution after MiraChat asks for the value -> DB_WRITE
r2 = classify_intent(
    prompt="Make it +1 214 555 8721",
    chat_history=[
        {"role": "user", "content": "Update Kevin Sauer's phone number at Microsoft"},
        {"role": "assistant", "content": "What new phone number should I set for Kevin Sauer?"},
    ],
    model="intent-router-qwen2.5-0.5b-instruct:v3",
)
# -> r2["route"] == "DB_WRITE"
```

| Parameter / Field | Default / Type | Description |
|---|---|---|
| `prompt` | `str` (required) | Latest user message to classify (whitespace-stripped). |
| `chat_history` | `list[dict] \| None` | Optional prior turns: `[{"role": "user" \| "assistant", "content": "..."}]`. Keeps last **6** turns. |
| `model` / `version` | `str` / `str \| None` | Pass `model="intent-router-qwen2.5-0.5b-instruct:v3"` (or `version="v3"`). |
| `base_url` | `"http://127.0.0.1:11434"` | Ollama server base URL. |
| `clarify_threshold` | `0.40` | Minimum `P(ASK)` to prioritize `CLARIFICATION_REQUIRED` over speculative `DB_WRITE`. |
| `write_threshold` | `0.40` | Minimum `P(WRITE)` to catch mixed `READ+WRITE` when clarification is not stronger. |
| `confidence_threshold` | `0.60` | Minimum top probability before falling back to `fallback_route`. |
| `fallback_route` | `"CLARIFICATION_REQUIRED"` | Route returned when max probability is below `confidence_threshold`. |
| `max_history_turns` | `6` | Maximum trailing user/assistant turns retained in context. |
| `keep_alive` / `timeout_sec` | `"10m"` / `15.0` | Keeps weights hot in memory for 10 min; 15s allows occasional cold start. |
| **Return Payload** | `dict` | `{route, raw_prediction, confidence, probabilities, fallback_triggered, model, latency_ms}` |

---

### Option B: Direct Ollama `/api/chat` Protocol (Node.js, TypeScript, Go, Java, C#)

If the target project is not written in Python, call Ollama's `POST /api/chat` endpoint directly and extract `logprobs` using the exact rules below.

> [!WARNING]
> **Do NOT Inject a System Message for Fine-Tuned `:v3`:** The `Modelfile` already bakes in the exact training `SYSTEM` instruction, ChatML template, `temperature 0`, `num_predict 1`, `num_ctx 768`, `num_thread 6`, and stop tokens. Send only `user`/`assistant` turns in `messages`.

```json
{
  "model": "intent-router-qwen2.5-0.5b-instruct:v3",
  "messages": [
    {"role": "user", "content": "Show Kevin Sauer at Microsoft"},
    {"role": "assistant", "content": "Found Kevin Sauer (+1 214 555 0100)."},
    {"role": "user", "content": "Update his mobile number to +1 214 555 8721"}
  ],
  "stream": false,
  "keep_alive": "10m",
  "logprobs": true,
  "top_logprobs": 20,
  "options": { "temperature": 0.0, "num_predict": 1 }
}
```

**Response Parsing & Guardrail Steps:**
1. **Extract linear probabilities:** Read `response.logprobs[0].top_logprobs`. For each entry, normalize `tok = entry.token.trim().toUpperCase()`. If `tok` matches `CHAT`, `READ`, `WRITE`, or `ASK`, add `Math.exp(entry.logprob)` to that class bucket.
2. **Normalize across 4 classes:** Divide each class sum by the total sum across all 4 buckets so `P(GENERAL_CHAT) + P(DB_READ) + P(DB_WRITE) + P(CLARIFICATION_REQUIRED) = 1.0`.
3. **Fallback if `logprobs` absent:** Inspect `response.message.content.trim().toUpperCase()`. If it matches a token or route name, assign `0.91` to that class and `0.03` to each of the other 3 classes.
4. **Apply 5-Step Guardrails:** Run the exact precedence rules defined in Section 4 below.

---

### Option C: FastAPI Microservice Wrapper (`app/main.py`)

If you prefer HTTP microservice isolation, set `OLLAMA_MODEL=intent-router-qwen2.5-0.5b-instruct:v3` in `.env` and start Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **`GET /health`** → returns `status`, `backend`, `model_loaded`, `confidence_threshold`, `write_threshold`, `clarify_threshold`, `fallback_route`, and `classes`.
- **`POST /route`** → accepts `{"prompt": "...", "chat_history": [...]}` (returns HTTP `422` on blank prompts) and responds with `route`, `raw_prediction`, `confidence`, `probabilities`, `fallback_triggered`, `backend`, and `latency_ms`.
- **Caching & Review Queue (`.env`):**
  - `ROUTE_CACHE_SIZE=512` enables thread-safe in-memory LRU caching on normalized `(chat_history, prompt)` tuples (`0` disables).
  - `ENABLE_REVIEW_LOGGING=True` (default `False`) logs predictions with `confidence < REVIEW_CONFIDENCE_THRESHOLD` (`0.75`) or `fallback_triggered=True` to `REVIEW_LOG_PATH` (`data/review/review_log.jsonl`) for human curation into `data/custom/`.

---

## 4. Safety Guardrails & Threshold Calibration

Evaluate calibrated probabilities in the exact 5-step order below (matching `app/intent_spec.py:apply_routing_guardrails` and `app/ollama_client.py`):

| Priority | Rule Name | Condition | Final Route & Behavior |
|---|---|---|---|
| **1** | **Clarification Priority** | `P(ASK) >= 0.40` **AND** `P(ASK) >= P(WRITE)` | Route to `CLARIFICATION_REQUIRED` (`confidence = P(ASK)`). Prevents incomplete updates or destructive deletes from executing via `WRITE_THRESHOLD`. |
| **2** | **Write Sensitivity** | `P(WRITE) >= 0.40` | Route to `DB_WRITE` (`confidence = P(WRITE)`). Captures compound `READ+WRITE` prompts or polite writes that split mass between `READ`/`CHAT` and `WRITE`. |
| **3** | **Confidence Fallback** | `max(P) < 0.60` | Route to `CLARIFICATION_REQUIRED` (`fallback_triggered = true`). Safe uncertainty handling. |
| **4** | **Primary Prediction** | `max(P) >= 0.60` | Route to `argmax(P)` (`fallback_triggered = false`). |
| **5** | **Post-Model Destructive Write Guard** | `final_route == DB_WRITE` **AND** prompt matches bulk/destructive pattern (`delete all`, `drop table`, `truncate`, `purge all`, `mass update`, `override every`, etc.) without a negative-write or conceptual cue | Intercept to `CLARIFICATION_REQUIRED` (`fallback_triggered = true`). Zero false positives on `GENERAL_CHAT` / `DB_READ`. |

---

## 5. Verified `v3` Benchmarks & Production Best Practices

| Class / Metric | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| `GENERAL_CHAT` (`CHAT`) | `1.0000` | `1.0000` | `1.0000` | `80` |
| `DB_READ` (`READ`) | `1.0000` | `1.0000` | `1.0000` | `76` |
| `DB_WRITE` (`WRITE`) | `1.0000` | `0.9865` | `0.9932` | `74` |
| `CLARIFICATION_REQUIRED` (`ASK`) | `0.9859` | `1.0000` | `0.9929` | `70` |
| **Overall Test Set (`v3`)** | **Macro P: `0.9965`** | **Acc: `99.67%`** | **Macro F1: `0.9965`** | **`300` (`48` multi-turn @ `100%`)** |

*(Single test miss: `"add a note to amazon"` — labeled `DB_WRITE` in legacy seed data, routed conservatively to `CLARIFICATION_REQUIRED` at `0.985` because no note content was supplied.)*

### Best Practices for MiraChat Integration

- **Close the Clarification Loop with `chat_history`:** When the router outputs `CLARIFICATION_REQUIRED`, ask the user a focused follow-up (e.g., *"Which contact or phone number should I update?"*). Append that assistant question and the user's answer into `chat_history` on the next turn — **`v3` achieved 100% accuracy across all 48 multi-turn test conversations**.
- **Bound History Length:** Keep the last **`6` turns** (`user`/`assistant`) and skip empty messages so token preprocessing stays fast (single-turn latency P50 = `127.23 ms` on CPU).
- **Confirmation Gate on Destructive Actions:** `v3` classifies destructive/bulk operations (e.g., *"Delete Microsoft from CRM"*, *"Drop leads table"*, *"Override every active deal stage"*) as `CLARIFICATION_REQUIRED` until confirmed.
- **Warm Keep-Alive:** Keep `"keep_alive": "10m"` in every Ollama request so the 948 MiB `F16` model remains pinned in memory between MiraChat requests.

### Go-Live Handoff Checklist

- [ ] Download `intent-router-qwen2.5-0.5b-instruct-v3-f16.gguf` and `Modelfile` from Hugging Face (`main`).
- [ ] Run `ollama create intent-router-qwen2.5-0.5b-instruct:v3 -f Modelfile` and verify via `ollama list`.
- [ ] Verify Ollama `/api/chat` returns `logprobs` and single tokens `CHAT`, `READ`, `WRITE`, or `ASK`.
- [ ] Set runtime model ref to **`intent-router-qwen2.5-0.5b-instruct:v3`** (do not leave legacy `:v1` / `:v2` defaults) with `keep_alive: "10m"` and do not override `num_ctx`/`num_thread` per request.
- [ ] Implement the 5-step guardrails (`Clarify >= 0.40 & >= Write` → `Write >= 0.40` → `Confidence < 0.60` → `Argmax` → Post-model destructive write check).
- [ ] Wire `CLARIFICATION_REQUIRED` responses to prompt the user clearly without executing SQL or CRM writes.
