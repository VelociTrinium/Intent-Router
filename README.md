# Intent Router

A low-latency, 4-class NLP intent router designed to classify incoming user prompts (single-turn or multi-turn with `chat_history`) into downstream execution pipelines: **`GENERAL_CHAT`**, **`DB_READ`**, **`DB_WRITE`**, or **`CLARIFICATION_REQUIRED`**.

The deployment artifact is a **fine-tuned `Qwen2.5-0.5B-Instruct` (`Q8_0` or `F16` GGUF) model running natively in Ollama**, outputting a single token (`CHAT`, `READ`, `WRITE`, or `ASK`) with exact `logprobs`-calibrated guardrails. Both a zero-dependency standalone Python client (`app/ollama_client.py`) and a FastAPI microservice (`app/main.py`) are included.

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
├── WorkingCommands.md            # Direct execution commands for every workflow
├── docs/                         # Integration & handoff guides
├── data/
│   ├── custom/                   # Curated JSON/JSONL working examples (working_examples.json)
│   ├── seed_prompts.json         # Hand-written single-turn seed examples per class (649 seeds)
│   ├── raw/                      # Gemini-generated synthetic prompts (synthetic_prompts.json)
│   └── processed/                # Deduplicated stratified train.jsonl, val.jsonl, test.jsonl
├── models/
│   ├── hf_base/                  # Downloaded base Qwen2.5-0.5B-Instruct weights (gitignored)
│   ├── ollama_lora/              # Best LoRA adapter checkpoint (gitignored)
│   ├── ollama_merged/            # Merged FP32 HuggingFace weights (gitignored)
│   ├── ollama_gguf/              # Exported GGUF + Ollama Modelfile (gitignored)
│   └── evaluation_report.json    # Held-out test metrics, 4x4 confusion matrix, latency
├── scripts/
│   ├── build_dataset.py          # Validates data/custom/ + merges/splits train/val/test JSONL
│   ├── generate_data.py          # Gemini structured output generator (single + multi-turn, 4 classes)
│   ├── train_ollama.py           # CPU/GPU LoRA trainer for Qwen2.5-0.5B-Instruct (1-token)
│   ├── export_ollama_gguf.py     # Converts merged weights to GGUF & registers Ollama model
│   ├── evaluate.py               # Benchmarks accuracy, Macro F1, DB_WRITE/ASK recall, latency
│   ├── retrain_pipeline.py       # End-to-end orchestrator: custom/seed data -> fresh model -> Ollama -> eval
│   ├── push_to_hf.py             # Uploads exported GGUF, Modelfile, and metrics to Hugging Face Hub
│   └── cleanup.py                # Standalone cleaner removing checkpoints, LoRA/merged weights, and GGUFs
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

### End-to-End Pipeline Flow

```text
data/custom/*.json(l)  ──┐
data/seed_prompts.json ──┼──▶ scripts/build_dataset.py ──▶ data/processed/{train,val,test}.jsonl
data/raw/ (synthetic)  ──┘          │
                                    ▼
Fresh Base HF Model ──▶ scripts/train_ollama.py (LoRA, target-token masked) ──▶ models/ollama_merged/
                                    │
                                    ▼
                        scripts/export_ollama_gguf.py (Q8_0/F16 GGUF + Modelfile) ──▶ ollama create <name>
                                    │
                                    ▼
                        scripts/evaluate.py --ollama-model <name> ──▶ models/evaluation_report.json
                                    │
                                    ▼
                        scripts/push_to_hf.py (optional) ──▶ scripts/cleanup.py (reclaim build storage)
```

Everything can run step-by-step or via one command: [`scripts/retrain_pipeline.py`](scripts/retrain_pipeline.py). See [`WorkingCommands.md`](WorkingCommands.md) for direct copy-paste commands across every workflow.

---

## Intent Classes & Guardrail Rules

### Intent Classes (Single-Token Outputs)
1. **`GENERAL_CHAT`** (token: `CHAT`): Greetings, conceptual/domain/SQL syntax questions, office announcements/HR notices, and drafting text/emails without saving to the database.
2. **`DB_READ`** (token: `READ`): Clear, specific requests to fetch, search, filter, count, aggregate, or inspect live, deleted, or audit database records (including explicit negative-write guards like *"don't update — just show..."*).
3. **`DB_WRITE`** (token: `WRITE`): Clear, actionable, non-destructive mutations with explicit targets and values (create records, update fields to concrete values, log calls/notes/emails, schedule meetings/tasks, and complete mixed READ+WRITE requests).
4. **`CLARIFICATION_REQUIRED`** (token: `ASK`): Vague or underspecified requests missing target entity or new value (*"Update the contact's phone number"*, *"Add Sarah"*, *"all data."*), unresolved pronouns without context (*"Update it"*), destructive/bulk deletes/drops/purges requiring confirmation, and ambiguous read-vs-write fragments (*"Pursuit #8436"*, *"Update or delete that lead?"*).

### Decision Guardrails (`CLARIFY=0.40`, `WRITE=0.40`, `CONFIDENCE=0.60`, `FALLBACK=CLARIFICATION_REQUIRED`)
1. **Clarification Priority Rule**: If `P(CLARIFICATION_REQUIRED) >= 0.40` and `P(CLARIFICATION_REQUIRED) >= P(DB_WRITE)`, route to `CLARIFICATION_REQUIRED` so incomplete/destructive writes still ask for clarification.
2. **Write Sensitivity Rule**: Else if `P(DB_WRITE) >= 0.40`, route to `DB_WRITE` so mixed READ+WRITE and actionable mutations are never dropped silently.
3. **Confidence Fallback Rule**: Else if `max(P) < 0.60`, fall back safely to `CLARIFICATION_REQUIRED`.
4. **Primary Prediction Rule**: Otherwise, route to the highest-probability class.

### Runtime Contract Preserved Across Every Retrain
No matter which base model or dataset mix you train with, the pipeline enforces the exact same runtime interface for [`app/ollama_client.py`](app/ollama_client.py), [`app/main.py`](app/main.py), and downstream projects:
1. `Modelfile` bakes in `OLLAMA_CLASSIFIER_SYSTEM_PROMPT`, `ChatML` template, `temperature 0`, `num_predict 1`, and Qwen stop tokens.
2. The model emits 1 token: `CHAT` (`GENERAL_CHAT`), `READ` (`DB_READ`), `WRITE` (`DB_WRITE`), or `ASK` (`CLARIFICATION_REQUIRED`).
3. `logprobs` on that token give calibrated probabilities for `CLARIFY_THRESHOLD=0.40`, `WRITE_THRESHOLD=0.40`, `CONFIDENCE_THRESHOLD=0.60`, and `FALLBACK_ROUTE=CLARIFICATION_REQUIRED`.

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

## Adding Your Classified Working Data (`data/custom/`)

Add labeled prompts to [`data/custom/working_examples.json`](data/custom/working_examples.json) (or drop additional `.json` / `.jsonl` files in `data/custom/`). See [`data/custom/README.md`](data/custom/README.md) for all accepted JSON schemas (list-of-objects, class-grouped dict, and JSONL).

### Example `data/custom/working_examples.json`
```json
[
  {
    "prompt": "what's our biggest deal in miami?",
    "label": "DB_READ"
  },
  {
    "prompt": "Find HSBC's primary contact and update their mobile to +1-555-0192",
    "label": "DB_WRITE"
  },
  {
    "prompt": "Update Kevin's phone number",
    "label": "CLARIFICATION_REQUIRED"
  },
  {
    "prompt": "Make it +1 214 555 8721",
    "label": "DB_WRITE",
    "chat_history": [
      {"role": "user", "content": "Update Kevin Sauer's phone number at Microsoft"},
      {"role": "assistant", "content": "What new phone number should I set for Kevin Sauer?"}
    ]
  },
  {
    "prompt": "How do I write an UPDATE statement in Postgres?",
    "label": "GENERAL_CHAT",
    "split": "test"
  }
]
```

### Key Dataset Guarantees in [`scripts/build_dataset.py`](scripts/build_dataset.py)
- **Strict validation**: Rejects missing text, invalid labels (`GENERAL_CHAT`/`CHAT`, `DB_READ`/`READ`, `DB_WRITE`/`WRITE`, `CLARIFICATION_REQUIRED`/`ASK`), invalid `split`, or malformed `chat_history` before training starts.
- **Custom label always wins**: If an example in `data/custom/` matches a prompt (plus `chat_history`) in `data/raw/synthetic_prompts.json` or `data/seed_prompts.json`, your custom label overrides it.
- **Pinned evaluation safety**: Adding `"split": "test"` (or `"val"` / `"train"`) locks golden test cases into that exact split so they never leak into training.
- **Optional training weighting (`--custom-train-repeat`)**: When combining e.g. 50 curated production examples alongside synthetic data, `--custom-train-repeat 2` or `3` repeats custom examples inside `train.jsonl` only (never in `val.jsonl` or `test.jsonl`).

### Preview & Validate Splits (`--dry-run`)
Checks schema, deduplication, and split counts without touching `data/processed/` or models:
```bash
# Option A: Curated data/seed_prompts.json (649 seeds across 4 classes) + data/custom/
python scripts/retrain_pipeline.py --include-seeds --dry-run

# Option B: Seeds expanded via Gemini (1,080 balanced single-turn + multi-turn across 4 classes)
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 1080 --dry-run

# Option C: Strictly your custom JSON examples
python scripts/retrain_pipeline.py --custom-only --dry-run
```

---

## End-to-End Training, Retraining & Export Workflows

### Modular Step-by-Step Execution

#### 1. Build Splits Directly from Seeds OR Expand via Gemini
```bash
# Option A: Build splits directly from curated data/seed_prompts.json (649 seeds) + data/custom/
python scripts/build_dataset.py --include-seeds

# Option B: Expand seeds + multi-turn examples via Gemini (~1,080 balanced across 4 classes)
python scripts/generate_data.py --target-total 1080
python scripts/build_dataset.py
```
Outputs stratified splits (`75%` train, `10%` val, `15%` test) with optional `chat_history` in `data/processed/`.

#### 2. Fine-Tune `Qwen2.5-0.5B-Instruct` with LoRA
```bash
python scripts/train_ollama.py --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4 --max-seq-length 384
```
Saves the best LoRA checkpoint (by `0.50 * guarded_macro_f1 + 0.25 * db_write_recall + 0.25 * clarification_recall`) to `models/ollama_lora/` and merges FP32 weights to `models/ollama_merged/`.

#### 3. Export to GGUF & Register in Ollama
```bash
python scripts/export_ollama_gguf.py --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1
```
Produces `models/ollama_gguf/intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf` (or `-f16.gguf` with `--outtype f16`) and `models/ollama_gguf/Modelfile` with baked-in system prompt, ChatML template, `temperature 0`, and `num_predict 1`, then runs `ollama create intent-router-qwen2.5-0.5b-instruct:v1`.

#### 4. Evaluate Against Held-Out Test Split
```bash
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v1 --compare-untuned-ollama qwen2.5:0.5b
```
Outputs raw vs guarded accuracy, Macro F1, `DB_WRITE` recall, `CLARIFICATION_REQUIRED` recall, multi-turn accuracy, 4x4 confusion matrix, inference latency (`mean`, `p50`, `p95`), and saves `models/evaluation_report.json`.

---

### One-Command Retrain & Release Pipeline (`scripts/retrain_pipeline.py`)

Runs data validation/merge -> fresh base model LoRA fine-tuning (wiping any stale adapters first) -> GGUF export -> Ollama registration -> quality-gate evaluation (`DB_WRITE` recall `>= 0.95`, `CLARIFICATION_REQUIRED` recall `>= 0.90`, and Macro F1 `>= 0.90`):

#### A. Retrain Directly from `seed_prompts.json` (+ `data/custom/`, No Gemini Required)
```bash
python scripts/retrain_pipeline.py \
  --include-seeds \
  --base-model models/hf_base/Qwen2.5-0.5B-Instruct \
  --epochs 2 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 3e-4 \
  --lora-r 16 \
  --outtype q8_0 \
  --model-name intent-router-qwen2.5-0.5b-instruct \
  --version v1
```

#### B. Retrain With Gemini Synthetic + Multi-Turn Expansion (~1,080 Balanced Across 4 Classes)
Adds single-turn & multi-turn `chat_history` examples on top of `seed_prompts.json` and merges `data/custom/`:
```bash
python scripts/retrain_pipeline.py \
  --regenerate-synthetic \
  --target-total 1080 \
  --base-model models/hf_base/Qwen2.5-0.5B-Instruct \
  --epochs 2 \
  --batch-size 8 \
  --grad-accum 2 \
  --lr 3e-4 \
  --lora-r 16 \
  --outtype q8_0 \
  --model-name intent-router-qwen2.5-0.5b-instruct \
  --version v1
```

#### C. Release a New Version Without Overwriting `v1`
Pass a new `--version` tag (e.g. `v2`, `1.1.0`) so `v1` stays intact in Ollama while you test `v2`:
```bash
python scripts/retrain_pipeline.py --include-seeds --version v2 --custom-train-repeat 2
```

#### D. Train Strictly on Your Own Custom Data (`--custom-only`)
Once you accumulate enough curated real examples across all 4 classes (e.g. 300+ balanced):
```bash
python scripts/retrain_pipeline.py --custom-only --version v2
```

---

## Pushing the Exported GGUF Model to Hugging Face

Ensure `HF_TOKEN=...` is set in `.env` (or pass `--hf-token`), then run:
```bash
# Optional dry run first to inspect files:
python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0 --dry-run

# Upload .gguf, Modelfile, README.md model card, training_metadata.json, and evaluation_report.json with release tag:
python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0
```
*(Add `--private` for a private repository, or `--commit-message "..."` for custom release notes. This uploads all `.gguf` files in `models/ollama_gguf/`, `Modelfile`, `README.md`, `training_metadata.json`, and `evaluation_report.json` if present.)*

---

## Cleaning Up Build & Epoch Storage (`scripts/cleanup.py`)

Once Ollama (`ollama create`) holds the model in Ollama's local blob store and any Hugging Face upload has finished, run the standalone zero-dependency cleanup script to delete `checkpoints/`, `models/ollama_lora/`, `models/ollama_merged/`, and `models/ollama_gguf/` (~2.8+ GB per build) while preserving `models/hf_base/`, `models/evaluation_report.json`, and `data/`:
```bash
python scripts/cleanup.py --dry-run
python scripts/cleanup.py --yes
```
*(Optional flags: `--include-caches` to clear project `__pycache__`/`.pytest_cache`, `--include-data` to clear generated `data/raw` and `data/processed`, `--include-base` to remove `models/hf_base/`, `--include-reports` to remove `models/evaluation_report.json`.)*

---

## Using `intent-router` in Your Separate Project

### Option A: Zero-Dependency Direct Ollama Client (Recommended)
Copy [`app/ollama_client.py`](app/ollama_client.py) into your project (uses only Python stdlib `urllib.request`, `json`, `math`, `time`). It calls `http://127.0.0.1:11434/api/chat` with `logprobs: True, top_logprobs: 20, num_predict: 1` and enforces `CLARIFY_THRESHOLD=0.40`, `WRITE_THRESHOLD=0.40`, `CONFIDENCE_THRESHOLD=0.60`, and `FALLBACK_ROUTE="CLARIFICATION_REQUIRED"`:

```python
from app.ollama_client import classify_intent

result = classify_intent(
    prompt="Update Kevin's phone number",
    model="intent-router-qwen2.5-0.5b-instruct:v1",
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
