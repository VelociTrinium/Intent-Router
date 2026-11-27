# Intent Router v2 — Quick-Start Implementation Guide

> **4-Class Low-Latency NLP Intent Classifier for MiraChat / ReConnect CRM**  
> Fine-tuned `Qwen2.5-0.5B-Instruct` (`F16` GGUF via Ollama) • Model Tag: `intent-router-qwen2.5-0.5b-instruct:v2`

| Property | Value / Specification |
|---|---|
| **Hugging Face Repository** | `VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf` |
| **Primary Model Artifacts** | `intent-router-qwen2.5-0.5b-instruct-v2-f16.gguf` (948.1 MB) + `Modelfile` |
| **Ollama Registration Tag** | **`intent-router-qwen2.5-0.5b-instruct:v2`** |
| **Base Model & Format** | `Qwen2.5-0.5B-Instruct` (ChatML, LoRA `r=16`/`alpha=32` merged FP32, `F16` GGUF) |
| **Output Classes (1-Token)** | `GENERAL_CHAT` (`CHAT`) \| `DB_READ` (`READ`) \| `DB_WRITE` (`WRITE`) \| `CLARIFICATION_REQUIRED` (`ASK`) |
| **Held-Out Test Benchmark** | **97.55%** Guarded Accuracy \| **0.9754** Macro F1 \| **100%** `DB_WRITE`, `DB_READ`, `CLARIFICATION_REQUIRED` & Multi-Turn Recall (163 prompts, 44 multi-turn) |
| **Local Inference Latency** | **181.55 ms** mean \| **151.96 ms** P50 median \| **361.35 ms** P95 (local CPU, single-token output) |

---

## 1. Quick-Start Deployment (Ollama Setup in 3 Steps)

This classifier runs as a local **Ollama** model outputting strictly **1 token** per request (`num_predict: 1`). Ensure Ollama is installed and running on `http://127.0.0.1:11434` (`ollama serve`) with at least ~1.2 GB free RAM/VRAM before continuing.

### Step 1: Download the `v2` GGUF & `Modelfile` from Hugging Face

```bash
pip install -U huggingface_hub
huggingface-cli download VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf \
  intent-router-qwen2.5-0.5b-instruct-v2-f16.gguf Modelfile \
  --local-dir ./intent-router-v2
```
*(Use `hf download ...` instead of `huggingface-cli download ...` if running `huggingface_hub >= 1.0`.)*

### Step 2: Register the Model in Ollama as `:v2`

```bash
cd intent-router-v2
ollama create intent-router-qwen2.5-0.5b-instruct:v2 -f Modelfile
ollama list
```

> [!WARNING]
> **Important Version Tag Note:** Always register and call the model tag **`intent-router-qwen2.5-0.5b-instruct:v2`**. Older repository defaults (such as `.env` templates, the default `model` parameter in `ollama_client.py`, or the auto-generated HF quickstart snippet) may still reference `:v1`. The Hugging Face `Modelfile` already points to `intent-router-qwen2.5-0.5b-instruct-v2-f16.gguf`.

### Step 3: Smoke-Test Live Inference with `logprobs`

```bash
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "intent-router-qwen2.5-0.5b-instruct:v2",
  "messages": [{"role": "user", "content": "Update Kevin phone number"}],
  "stream": false,
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
| `GENERAL_CHAT` | `CHAT` (`60717`) | Greetings, capability questions, ReConnect workflow how-tos, CRE conceptual terms (cap rate, WALE, Grade A), SQL syntax explanations, office/HR notices, drafting emails/proposals without saving to DB. | Send to conversational assistant / general LLM response flow. Do **not** query or mutate the CRM database. |
| `DB_READ` | `READ` (`9560`) | Clear, specific queries to list, search, filter, count, aggregate, compare, or inspect live/deleted/audit records; explicit negative-write guards (*"don't update — just check pursuit #7724"*). | Send to read-only NL-to-SQL / CRM lookup pipeline. |
| `DB_WRITE` | `WRITE` (`32781`) | Clear, non-destructive CRM writes with explicit entity & values: creating records, updating fields to concrete values, logging calls/notes/emails, scheduling meetings, or complete mixed `READ+WRITE` requests. | Send to CRM write / mutation agent or confirmation-ready action builder. |
| `CLARIFICATION_REQUIRED` | `ASK` (`7384`) | Vague/broad prompts (*"all data."*), missing target or new value (*"Update Kevin's phone number"*, *"Add Sarah"*), unresolved pronouns without history (*"Update it"*), ambiguous read-vs-write (*"Pursuit #8436"*), or destructive/bulk deletes/drops/purges. | Do **not** execute DB read/write. Prompt the user to supply the missing detail, clarify intent, or explicitly confirm destructive action. |

---

## 3. Integration Options for Your Application

### Option A: Zero-Dependency Python Client (Fastest Drop-In)

Copy `app/ollama_client.py` directly into your Python codebase. It uses only the Python standard library (`urllib.request`, `json`, `math`, `time`), calls Ollama `/api/chat` with `logprobs=True` and `num_predict=1`, normalizes token probabilities across the 4 classes, and enforces all safety guardrails automatically.

```python
from ollama_client import classify_intent

# Example 1: Single-turn incomplete write -> CLARIFICATION_REQUIRED
r1 = classify_intent(
    prompt="Update Kevin's phone number",
    model="intent-router-qwen2.5-0.5b-instruct:v2",
)
# -> r1["route"] == "CLARIFICATION_REQUIRED"

# Example 2: Multi-turn resolution after MiraChat asks for the value -> DB_WRITE
r2 = classify_intent(
    prompt="Make it +1 214 555 8721",
    chat_history=[
        {"role": "user", "content": "Update Kevin Sauer's phone number at Microsoft"},
        {"role": "assistant", "content": "What new phone number should I set for Kevin Sauer?"},
    ],
    model="intent-router-qwen2.5-0.5b-instruct:v2",
)
# -> r2["route"] == "DB_WRITE"
```

| Parameter / Field | Default / Type | Description |
|---|---|---|
| `prompt` | `str` (required) | Latest user message to classify (whitespace-stripped). |
| `chat_history` | `list[dict] \| None` | Optional prior turns: `[{"role": "user" \| "assistant", "content": "..."}]`. Keeps last 4 turns. |
| `model` / `version` | `str` / `str \| None` | Pass `model="intent-router-qwen2.5-0.5b-instruct:v2"` (or `version="v2"`). |
| `base_url` | `"http://127.0.0.1:11434"` | Ollama server base URL. |
| `clarify_threshold` | `0.40` | Minimum `P(ASK)` to prioritize `CLARIFICATION_REQUIRED` over speculative `DB_WRITE`. |
| `write_threshold` | `0.40` | Minimum `P(WRITE)` to catch mixed `READ+WRITE` when clarification is not stronger. |
| `confidence_threshold` | `0.60` | Minimum top probability before falling back to `fallback_route`. |
| `fallback_route` | `"CLARIFICATION_REQUIRED"` | Route returned when max probability is below `confidence_threshold`. |
| `keep_alive` / `timeout_sec` | `"10m"` / `15.0` | Keeps weights hot in memory for 10 min; 15s allows occasional cold start. |
| **Return Payload** | `dict` | `{route, raw_prediction, confidence, probabilities, fallback_triggered, model, latency_ms}` |

---

### Option B: Direct Ollama `/api/chat` Protocol (Node.js, TypeScript, Go, Java, C#)

If the target project is not written in Python, call Ollama's `POST /api/chat` endpoint directly and extract `logprobs` using the exact rules below.

> [!WARNING]
> **Do NOT Inject a System Message for Fine-Tuned `:v2`:** The `Modelfile` already bakes in the exact training `SYSTEM` instruction, ChatML template, `temperature 0`, `num_predict 1`, and stop tokens. Send only `user`/`assistant` turns in `messages` — adding another `system` message can degrade calibration.

```json
{
  "model": "intent-router-qwen2.5-0.5b-instruct:v2",
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
4. **Apply 4-Class Guardrails:** Run the exact precedence rules defined in Section 4 below.

---

### Option C: FastAPI Microservice Wrapper (`app/main.py`)

If you prefer HTTP microservice isolation over embedding client code, update `OLLAMA_MODEL=intent-router-qwen2.5-0.5b-instruct:v2` in `.env` and start Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **`GET /health`** -> returns `status`, `backend`, `model_loaded`, `confidence_threshold`, `write_threshold`, `clarify_threshold`, `fallback_route`, and `classes`.
- **`POST /route`** -> accepts `{"prompt": "...", "chat_history": [...]}` (returns HTTP `422` on blank prompts) and responds with `route`, `raw_prediction`, `confidence`, `probabilities`, `fallback_triggered`, `backend`, and `latency_ms`.

---

## 4. Safety Guardrails & Threshold Calibration

Never rely solely on greedy argmax string output. Evaluate calibrated probabilities in the exact 4-step order below (matching `app/intent_spec.py:apply_routing_guardrails`):

| Priority | Rule Name | Condition | Final Route & Behavior |
|---|---|---|---|
| **1** | **Clarification Priority** | `P(ASK) >= 0.40` **AND** `P(ASK) >= P(WRITE)` | Route to `CLARIFICATION_REQUIRED` (`confidence = P(ASK)`). Prevents incomplete updates or destructive deletes from executing via `WRITE_THRESHOLD`. |
| **2** | **Write Sensitivity** | `P(WRITE) >= 0.40` | Route to `DB_WRITE` (`confidence = P(WRITE)`). Captures compound `READ+WRITE` prompts or polite writes that split mass between `READ`/`CHAT` and `WRITE`. |
| **3** | **Confidence Fallback** | `max(P) < 0.60` | Route to `CLARIFICATION_REQUIRED` (`fallback_triggered = true`). Safe uncertainty handling. |
| **4** | **Primary Prediction** | `max(P) >= 0.60` | Route to `argmax(P)` (`fallback_triggered = false`). |

### Guardrail Worked Examples

| Example Probability Vector (`CHAT / READ / WRITE / ASK`) | Raw Top | Guarded Route | Why |
|---|---|---|---|
| `0.04 / 0.06 / 0.42 / 0.48` | `ASK` | `CLARIFICATION_REQUIRED` | Rule 1: `P(ASK) >= 0.40` & `>= P(WRITE)` (destructive/missing details) |
| `0.05 / 0.05 / 0.45 / 0.45` | `WRITE` | `CLARIFICATION_REQUIRED` | Rule 1 tie-break favors asking for clarification safely |
| `0.03 / 0.50 / 0.44 / 0.03` | `READ` | `DB_WRITE` | Rule 2: Mixed `READ+WRITE` rescued (`P(ASK)` is low at `0.03`) |
| `0.20 / 0.55 / 0.15 / 0.10` | `READ` | `CLARIFICATION_REQUIRED` | Rule 3: Top prob `0.55 < CONFIDENCE_THRESHOLD (0.60)` |
| `0.02 / 0.03 / 0.92 / 0.03` | `WRITE` | `DB_WRITE` | Rule 4: High confidence `>= 0.60` |

---

## 5. Verified `v2` Benchmarks & Production Best Practices

| Class / Metric | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| `GENERAL_CHAT` (`CHAT`) | `1.0000` | `0.9048` | `0.9500` | `42` |
| `DB_READ` (`READ`) | `0.9545` | `1.0000` | `0.9767` | `42` |
| `DB_WRITE` (`WRITE`) | `0.9767` | `1.0000` | `0.9882` | `42` |
| `CLARIFICATION_REQUIRED` (`ASK`) | `0.9737` | `1.0000` | `0.9867` | `37` |
| **Overall Test Set (`v2`)** | **Macro P: `0.9762`** | **Acc: `97.55%`** | **Macro F1: `0.9754`** | **`163` (`44` multi-turn @ `100%`)** |

### Best Practices for MiraChat Integration

- **Close the Clarification Loop with `chat_history`:** When the router outputs `CLARIFICATION_REQUIRED`, ask the user a focused follow-up (e.g., *"Which contact or phone number should I update?"*). Append that assistant question and the user's answer into `chat_history` on the next turn — **`v2` achieved 100% accuracy across all 44 multi-turn test conversations**.
- **Bound History Length:** Only send the last **`4` turns** (`user`/`assistant`) and skip empty messages so token preprocessing stays fast (single-turn latency P50 = `152 ms` on CPU).
- **Confirmation Gate on Destructive Actions:** Note that `v2` intentionally classifies destructive/bulk operations (e.g., *"Delete Microsoft from CRM"*, *"Drop leads table"*) as `CLARIFICATION_REQUIRED` until confirmed. After the user replies with explicit confirmation in `chat_history`, your workflow can proceed safely.
- **Warm Keep-Alive:** Keep `"keep_alive": "10m"` in every Ollama request so the 948 MB `F16` model remains pinned in memory between MiraChat requests.

### Go-Live Handoff Checklist

- [ ] Download `intent-router-qwen2.5-0.5b-instruct-v2-f16.gguf` and `Modelfile` from Hugging Face.
- [ ] Run `ollama create intent-router-qwen2.5-0.5b-instruct:v2 -f Modelfile` and confirm via `ollama list`.
- [ ] Verify Ollama `/api/chat` returns `logprobs` and single tokens `CHAT`, `READ`, `WRITE`, or `ASK`.
- [ ] Configure runtime model ref to **`:v2`** (do not leave legacy `:v1` defaults) with `keep_alive: "10m"`.
- [ ] Implement 4-step guardrails (`Clarify >= 0.40 & >= Write` -> `Write >= 0.40` -> `Confidence < 0.60` -> `Argmax`).
- [ ] Wire `CLARIFICATION_REQUIRED` responses to prompt the user clearly without executing SQL or CRM writes.
