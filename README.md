# Intent Router

A low-latency, 3-class NLP intent router designed to classify incoming user prompts (single-turn or multi-turn with `chat_history`) into downstream execution pipelines: **`GENERAL_CHAT`**, **`DB_READ`**, or **`DB_WRITE`**.

The primary deployment artifact is a **fine-tuned `Qwen2.5-0.5B-Instruct` (`Q8_0` GGUF, ~531 MB) model running natively in Ollama as `intent-router`**, outputting a single token (`CHAT`, `READ`, or `WRITE`) with exact `logprobs`-calibrated guardrails. A FastAPI microservice (`app/main.py`) and zero-dependency standalone Python client (`app/ollama_client.py`) are both included, alongside SetFit and TF-IDF benchmark backends.

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
├── data/
│   ├── custom/                   # Your curated JSON/JSONL working examples (working_examples.json)
│   ├── seed_prompts.json         # Hand-written single-turn seed examples per class
│   ├── raw/                      # Gemini-generated synthetic prompts & verified reviews
│   ├── review_queue/             # Borderline prompts flagged for human review
│   └── processed/                # Deduplicated stratified train.jsonl, val.jsonl, test.jsonl
├── models/
│   ├── hf_base/                  # Downloaded base HuggingFace weights (gitignored)
│   ├── ollama_lora/              # Best LoRA adapter checkpoint (gitignored)
│   ├── ollama_merged/            # Merged FP32 HuggingFace weights (gitignored)
│   ├── ollama_gguf/              # Exported Q8_0 GGUF + Ollama Modelfile (gitignored)
│   ├── intent_router/            # SetFit benchmark artifacts
│   ├── baseline_tfidf/           # TF-IDF + LogisticRegression benchmark artifact
│   └── evaluation_report.json    # Side-by-side test metrics, confusion matrices, latency
├── scripts/
│   ├── build_dataset.py          # Validates data/custom/ + merges/splits train/val/test JSONL
│   ├── retrain_pipeline.py       # End-to-end orchestrator: custom data -> fresh model -> Ollama -> eval
│   ├── generate_data.py          # Gemini structured output generator (single + multi-turn)
│   ├── review_uncertain.py       # Interactive HITL CLI for reviewing borderline prompts
│   ├── train.py                  # Unified trainer (--backend ollama | setfit | baseline)
│   ├── train_ollama.py           # CPU/GPU LoRA trainer for Qwen2.5-0.5B-Instruct (1-token)
│   ├── export_ollama_gguf.py     # Converts merged weights to Q8_0 GGUF & creates Ollama model
│   ├── push_to_hf.py             # Uploads exported GGUF, Modelfile, and metrics to Hugging Face Hub
│   ├── train_setfit.py           # SetFit contrastive benchmark trainer
│   ├── train_baseline.py         # TF-IDF + LogisticRegression benchmark trainer
│   └── evaluate.py               # Benchmarks accuracy, Macro F1, DB_WRITE recall, latency
├── app/
│   ├── intent_spec.py            # Canonical system prompt, token map, logprobs extractor, loaders
│   ├── config.py                 # Thresholds (CONFIDENCE_THRESHOLD=0.60, WRITE_THRESHOLD=0.40)
│   ├── schemas.py                # Pydantic request/response schemas (prompt + optional chat_history)
│   ├── router_engine.py          # Unified engine (OllamaRouter, SetFitRouter, BaselineRouter)
│   ├── ollama_client.py          # Zero-dependency standalone client for external Python projects
│   └── main.py                   # FastAPI service exposing POST /route and GET /health
└── tests/
    ├── test_routing_rules.py     # Unit tests for guardrails, chat_history, and logprobs extraction
    └── test_api.py               # FastAPI TestClient & standalone client tests
```

---

## Intent Classes & Guardrail Rules

### Intent Classes
1. **`GENERAL_CHAT`** (model token: `CHAT`): Greetings, conceptual/SQL syntax questions, architecture advice, and drafting text/emails without saving to the database.
2. **`DB_READ`** (model token: `READ`): Fetching, searching, filtering, counting, aggregating, or inspecting live, deleted, or audit database records.
3. **`DB_WRITE`** (model token: `WRITE`): Creating, inserting, updating, deleting, logging calls/notes, scheduling meetings, polite/indirect write commands (`"Can you delete..."`), and **mixed READ+WRITE** compound prompts.

### Decision Guardrails (`WRITE_THRESHOLD=0.40`, `CONFIDENCE_THRESHOLD=0.60`)
1. **Write Sensitivity Rule**: If `P(DB_WRITE) >= 0.40`, route to `DB_WRITE` immediately so mutations are never dropped silently.
2. **Confidence Fallback Rule**: Otherwise, if `max(P) < 0.60`, fall back safely to `GENERAL_CHAT`.
3. **Primary Prediction Rule**: Otherwise, route to the highest-probability class.

---

## Quickstart & Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` for synthetic dataset generation. Default runtime settings:
```ini
ROUTER_BACKEND=ollama
CONFIDENCE_THRESHOLD=0.60
WRITE_THRESHOLD=0.40
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=intent-router-qwen2.5-0.5b-instruct:v1
OLLAMA_MODE=single_token
```

---

## End-to-End Training & Ollama Export Pipeline

### 1. Generate Single-Turn & Multi-Turn Dataset
```bash
python scripts/generate_data.py --target-total 810 --strict-api
python scripts/review_uncertain.py
```
Outputs stratified splits (`75%` train, `10%` val, `15%` test) with optional `chat_history` in `data/processed/`.

### 2. Fine-Tune `Qwen2.5-0.5B-Instruct` with LoRA
```bash
python scripts/train.py --backend ollama --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4
```
Saves the best LoRA checkpoint to `models/ollama_lora/` and merges weights to `models/ollama_merged/`.

### 3. Export to `Q8_0` GGUF, Register in Ollama & Push to Hugging Face
```bash
python scripts/export_ollama_gguf.py --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1

# Optional: push the exported .gguf + Modelfile to Hugging Face Hub (uses HF_TOKEN from .env)
python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0
```
Produces `models/ollama_gguf/intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf` and `models/ollama_gguf/Modelfile` with baked-in system prompt, ChatML template, `temperature 0`, and `num_predict 1`, then runs `ollama create intent-router-qwen2.5-0.5b-instruct:v1`.

### 4. Adding Custom Working Data & Retraining From a Fresh Base Model

Drop your own classified prompts into [`data/custom/working_examples.json`](data/custom/working_examples.json) and run the full pipeline (data validation/merge -> fresh base model LoRA training -> `Q8_0` GGUF export -> Ollama registration -> quality-gate evaluation):

```bash
# Preview split & validate custom JSON
python scripts/retrain_pipeline.py --dry-run

# Retrain fresh base model and publish versioned tag to Ollama
python scripts/retrain_pipeline.py --base-model Qwen/Qwen2.5-0.5B-Instruct --model-name intent-router-qwen2.5-0.5b-instruct --version v1
```

See **[PIPELINE.md](PIPELINE.md)** and **[data/custom/README.md](data/custom/README.md)** for full JSON schemas, `--custom-only`, `--custom-train-repeat`, candidate tags, and swapping base models.

### 5. Train Benchmark Models & Evaluate All Backends
```bash
python scripts/train.py --backend baseline
python scripts/train.py --backend setfit --num-iterations 2 --epochs 1
python scripts/evaluate.py --backend all --compare-untuned-ollama qwen2.5:0.5b
```

---

## Benchmark Results (126 Held-Out Test Prompts, CPU-Only Windows)

Measured on `data/processed/test.jsonl` (`42` samples per class, including `33` multi-turn `chat_history` follow-ups) with `WRITE_THRESHOLD=0.40` and `CONFIDENCE_THRESHOLD=0.60`:

| Backend | Guarded Acc | Macro F1 | `DB_WRITE` Recall | Multi-Turn Acc | Mean Latency (CPU) | P95 Latency | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`intent-router` (`Qwen2.5-0.5B` LoRA `Q8_0` in Ollama)** | **98.4%** | **0.9841** | **100.0%** | **97.0%** | **523.28 ms** | **698.12 ms** | **2 / 126** |
| SetFit (`BAAI/bge-small-en-v1.5`) | 89.7% | 0.8962 | 95.2% | 69.7% | 16.12 ms | 24.76 ms | 13 / 126 |
| TF-IDF + LogisticRegression | 75.4% | 0.7299 | 90.5% | 63.6% | 0.52 ms | 0.90 ms | 31 / 126 |
| Untuned `qwen2.5:3b` (zero-shot) | 70.6% | 0.6566 | 97.6% | 66.7% | 1120.30 ms | 2019.21 ms | 37 / 126 |
| Untuned `qwen2.5:0.5b` (zero-shot) | 38.1% | 0.2988 | 90.5% | 51.5% | 598.42 ms | 858.94 ms | 78 / 126 |

---

## Using `intent-router` in Your Separate Project

### Option A: Zero-Dependency Direct Ollama Client (Recommended)
Copy [`app/ollama_client.py`](app/ollama_client.py) into your project (uses only Python stdlib `urllib.request`, `json`, `math`, `time`). It calls `http://127.0.0.1:11434/api/chat` with `logprobs: True, top_logprobs: 20, num_predict: 1` and enforces `WRITE_THRESHOLD=0.40` and `CONFIDENCE_THRESHOLD=0.60`:

```python
from app.ollama_client import classify_intent

result = classify_intent(
    prompt="Actually just show them first, don't delete anything yet.",
    chat_history=[
        {"role": "user", "content": "Find all inactive leads from 2024 and delete them."},
        {"role": "assistant", "content": "I found 14 inactive leads. Should I proceed with deletion?"},
    ],
)
# -> {"route": "DB_READ", "confidence": ..., "probabilities": {...}, "fallback_triggered": False, ...}
```

### Option B: FastAPI Microservice
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- `GET /health`: returns active backend (`ollama`), readiness (`model_loaded`), and threshold configuration.
- `POST /route`: accepts `{"prompt": "...", "chat_history": [...]}` and returns `route`, `raw_prediction`, `confidence`, `probabilities`, `fallback_triggered`, `backend`, and `latency_ms`.

---

## Testing

```bash
pytest -v
```
