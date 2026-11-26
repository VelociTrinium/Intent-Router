# Intent Router

A low-latency, 4-class NLP intent router designed to classify incoming user prompts (single-turn or multi-turn with `chat_history`) into downstream execution pipelines: **`GENERAL_CHAT`**, **`DB_READ`**, **`DB_WRITE`**, or **`CLARIFICATION_REQUIRED`**.

The deployment artifact is a **fine-tuned `Qwen2.5-0.5B-Instruct` (`Q8_0` GGUF, ~531 MB) model running natively in Ollama**, outputting a single token (`CHAT`, `READ`, `WRITE`, or `ASK`) with exact `logprobs`-calibrated guardrails. A zero-dependency standalone Python client (`app/ollama_client.py`) and a FastAPI microservice (`app/main.py`) are both included.

---

## Architecture & Project Structure

```text
intent-router/
├── .env                          # Credentials & runtime config (gitignored)
├── .env.example                  # Committed template showing required env vars
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── PIPELINE.md
├── data/
│   ├── custom/                   # Your curated JSON/JSONL working examples (working_examples.json)
│   ├── seed_prompts.json         # Hand-written single-turn seed examples per class (649 seeds)
│   ├── raw/                      # Gemini-generated synthetic prompts (gitignored)
│   └── processed/                # Deduplicated stratified train.jsonl, val.jsonl, test.jsonl (gitignored)
├── models/
│   ├── hf_base/                  # Downloaded base Qwen2.5-0.5B-Instruct weights (gitignored)
│   ├── ollama_lora/              # Best LoRA adapter checkpoint (gitignored)
│   ├── ollama_merged/            # Merged FP32 HuggingFace weights (gitignored)
│   ├── ollama_gguf/              # Exported Q8_0 GGUF + Ollama Modelfile (gitignored)
│   └── evaluation_report.json    # Held-out test metrics, 4x4 confusion matrix, latency (gitignored)
├── scripts/
│   ├── build_dataset.py          # Validates data/custom/ + merges/splits train/val/test JSONL
│   ├── generate_data.py          # Gemini structured output generator (single + multi-turn, 4 classes)
│   ├── train_ollama.py           # CPU/GPU LoRA trainer for Qwen2.5-0.5B-Instruct (1-token)
│   ├── export_ollama_gguf.py     # Converts merged weights to Q8_0 GGUF & creates Ollama model
│   ├── evaluate.py               # Benchmarks accuracy, Macro F1, DB_WRITE/ASK recall, latency
│   ├── retrain_pipeline.py       # End-to-end orchestrator: custom/seed data -> fresh model -> Ollama -> eval
│   └── push_to_hf.py             # Uploads exported GGUF, Modelfile, and metrics to Hugging Face Hub
├── app/
│   ├── intent_spec.py            # Canonical system prompt, token map, logprobs & guardrail helpers
│   ├── config.py                 # Thresholds (CONFIDENCE=0.60, WRITE=0.40, CLARIFY=0.40, FALLBACK=CLARIFICATION_REQUIRED)
│   ├── schemas.py                # Pydantic request/response schemas (prompt + optional chat_history)
│   ├── router_engine.py          # OllamaRouter + guardrail RouterEngine
│   ├── ollama_client.py          # Zero-dependency standalone client for external Python projects
│   └── main.py                   # FastAPI service exposing POST /route and GET /health
└── tests/
    ├── test_routing_rules.py     # Unit tests for 4-class guardrails, chat_history, logprobs, and helpers
    └── test_api.py               # FastAPI TestClient & standalone client tests
```

---

## Intent Classes & Guardrail Rules

### Intent Classes (Single-Token Outputs)
1. **`GENERAL_CHAT`** (token: `CHAT`): Greetings, conceptual/CRE domain/SQL syntax questions, platform announcements/HR, and drafting text/emails without saving to the database.
2. **`DB_READ`** (token: `READ`): Clear, specific requests to fetch, search, filter, count, aggregate, or inspect live, deleted, or audit database records (including explicit negative-write guards like *"don't update — just show..."*).
3. **`DB_WRITE`** (token: `WRITE`): Clear, actionable, non-destructive mutations with explicit targets and values (create records, update fields to concrete values, log calls/notes/emails, schedule meetings/tasks, and complete mixed READ+WRITE requests).
4. **`CLARIFICATION_REQUIRED`** (token: `ASK`): Vague or underspecified requests missing target entity or new value (*"Update Kevin's phone number"*, *"Add Sarah"*, *"all data."*), unresolved pronouns without context (*"Update it"*), destructive/bulk deletes/drops/purges requiring confirmation, and ambiguous read-vs-write fragments (*"Pursuit #8436"*, *"Update or delete that lead?"*).

### Decision Guardrails (`CLARIFY=0.40`, `WRITE=0.40`, `CONFIDENCE=0.60`, `FALLBACK=CLARIFICATION_REQUIRED`)
1. **Clarification Priority Rule**: If `P(CLARIFICATION_REQUIRED) >= 0.40` and `P(CLARIFICATION_REQUIRED) >= P(DB_WRITE)`, route to `CLARIFICATION_REQUIRED` so incomplete/destructive writes still ask for clarification.
2. **Write Sensitivity Rule**: Else if `P(DB_WRITE) >= 0.40`, route to `DB_WRITE` so mixed READ+WRITE and actionable mutations are never dropped silently.
3. **Confidence Fallback Rule**: Else if `max(P) < 0.60`, fall back safely to `CLARIFICATION_REQUIRED`.
4. **Primary Prediction Rule**: Otherwise, route to the highest-probability class.

---

## Quickstart & Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` for synthetic dataset generation (and `HF_TOKEN` if pushing to Hugging Face). Default runtime settings:
```ini
CONFIDENCE_THRESHOLD=0.60
WRITE_THRESHOLD=0.40
CLARIFY_THRESHOLD=0.40
FALLBACK_ROUTE=CLARIFICATION_REQUIRED
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=intent-router-qwen2.5-0.5b-instruct:v1
```

---

## End-to-End Training & Ollama Export Pipeline

### 1. Build Splits Directly from Seeds OR Expand via Gemini
```bash
# Option A: Build splits directly from curated data/seed_prompts.json (649 seeds) + data/custom/
python scripts/build_dataset.py --include-seeds

# Option B: Expand seeds + multi-turn examples via Gemini (~1,080 balanced across 4 classes)
python scripts/generate_data.py --target-total 1080
python scripts/build_dataset.py
```
Outputs stratified splits (`75%` train, `10%` val, `15%` test) with optional `chat_history` in `data/processed/`.

### 2. Fine-Tune `Qwen2.5-0.5B-Instruct` with LoRA
```bash
python scripts/train_ollama.py --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4 --max-seq-length 384
```
Saves the best LoRA checkpoint to `models/ollama_lora/` and merges weights to `models/ollama_merged/`.

### 3. Export to `Q8_0` GGUF, Register in Ollama & Push to Hugging Face
```bash
python scripts/export_ollama_gguf.py --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1

# Optional: push the exported .gguf + Modelfile to Hugging Face Hub (uses HF_TOKEN from .env)
python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0
```
Produces `models/ollama_gguf/intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf` and `models/ollama_gguf/Modelfile` with baked-in system prompt, ChatML template, `temperature 0`, and `num_predict 1`, then runs `ollama create intent-router-qwen2.5-0.5b-instruct:v1`.

### 4. One-Command Retrain & Release Pipeline
Drop any extra classified prompts into [`data/custom/working_examples.json`](data/custom/working_examples.json) and run the full pipeline (data validation/merge -> fresh `Qwen2.5-0.5B-Instruct` LoRA training -> `Q8_0` GGUF export -> Ollama registration -> quality-gate evaluation):

```bash
# Preview split & validate seeds/custom JSON
python scripts/retrain_pipeline.py --include-seeds --dry-run

# Option A: Train fresh base model directly on seed_prompts.json + data/custom/
python scripts/retrain_pipeline.py --include-seeds --model-name intent-router-qwen2.5-0.5b-instruct --version v1

# Option B: Expand synthetic + multi-turn data via Gemini first (~1,080 samples across 4 classes)
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 1080 --model-name intent-router-qwen2.5-0.5b-instruct --version v1
```

See **[PIPELINE.md](PIPELINE.md)** and **[data/custom/README.md](data/custom/README.md)** for full JSON schemas, `--custom-only`, `--custom-train-repeat`, and versioned tags.

### 5. Evaluate Against Held-Out Test Split
```bash
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v1 --compare-untuned-ollama qwen2.5:0.5b
```

---

## Using `intent-router` in Your Separate Project

### Option A: Zero-Dependency Direct Ollama Client (Recommended)
Copy [`app/ollama_client.py`](app/ollama_client.py) into your project (uses only Python stdlib `urllib.request`, `json`, `math`, `time`). It calls `http://127.0.0.1:11434/api/chat` with `logprobs: True, top_logprobs: 20, num_predict: 1` and enforces `CLARIFY_THRESHOLD=0.40`, `WRITE_THRESHOLD=0.40`, `CONFIDENCE_THRESHOLD=0.60`, and `FALLBACK_ROUTE="CLARIFICATION_REQUIRED"`:

```python
from app.ollama_client import classify_intent

result = classify_intent(
    prompt="Update Kevin's phone number",
)
# -> {"route": "CLARIFICATION_REQUIRED", "confidence": ..., "probabilities": {...}, ...}
```

### Option B: FastAPI Microservice
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- `GET /health`: returns readiness (`model_loaded`), class list, and threshold configuration.
- `POST /route`: accepts `{"prompt": "...", "chat_history": [...]}` and returns `route`, `raw_prediction`, `confidence`, `probabilities`, `fallback_triggered`, `backend`, and `latency_ms`.

---

## Testing

```bash
pytest -v
```
