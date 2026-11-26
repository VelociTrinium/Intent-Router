# Retraining & Custom Data Pipeline Guide

This guide explains how to add your own classified working data in JSON, merge or override synthetic examples, and retrain a **fresh base model** from scratch into Ollama while keeping the exact same `CHAT` / `READ` / `WRITE` output contract and `logprobs` guardrails.

---

## 1. End-to-End Flow

```text
data/custom/*.json(l)  ──┐
                         ├──▶ scripts/build_dataset.py ──▶ data/processed/{train,val,test}.jsonl
data/raw/ (synthetic)  ──┘          │
                                    ▼
Fresh Base HF Model ──▶ scripts/train_ollama.py (LoRA, target-token masked) ──▶ models/ollama_merged/
                                    │
                                    ▼
                        scripts/export_ollama_gguf.py (Q8_0 GGUF + Modelfile) ──▶ ollama create <name>
                                    │
                                    ▼
                        scripts/evaluate.py --ollama-model <name> ──▶ models/evaluation_report.json
```

Everything is orchestrated by one command: [`scripts/retrain_pipeline.py`](scripts/retrain_pipeline.py).

---

## 2. Adding Your Classified Working Data

Add your labeled prompts to [`data/custom/working_examples.json`](data/custom/working_examples.json) (or drop additional `.json` / `.jsonl` files in `data/custom/`). See [`data/custom/README.md`](data/custom/README.md) for all accepted JSON schemas.

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
    "prompt": "How does that compare to Q4 of last year?",
    "label": "DB_READ",
    "chat_history": [
      {"role": "user", "content": "Retrieve occupancy rate stats for retail."},
      {"role": "assistant", "content": "Current retail occupancy is 91.4%."}
    ]
  },
  {
    "prompt": "How do I write an UPDATE statement in Postgres?",
    "label": "GENERAL_CHAT",
    "split": "test"
  }
]
```

### Key Guarantees in [`scripts/build_dataset.py`](scripts/build_dataset.py)
- **Strict validation**: Rejects missing text, invalid labels, or malformed `chat_history` before training starts.
- **Custom label always wins**: If an example in `data/custom/` matches a prompt in `data/raw/synthetic_prompts.json`, your custom label overrides the synthetic one.
- **Pinned evaluation safety**: Adding `"split": "test"` (or `"val"` / `"train"`) locks golden test cases into `test.jsonl` so they never leak into training.
- **Optional weighting (`--custom-train-repeat`)**: If you have e.g. 50 real production examples alongside ~800 synthetic examples, `--custom-train-repeat 2` or `3` repeats your real examples inside `train.jsonl` only (never in `val.jsonl` or `test.jsonl`).

---

### Preview & Validate Splits (Dry Run)
```bash
# If using Option A (seeds -> Gemini expansion):
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 810 --dry-run

# If using Option B (strictly your custom JSON examples):
python scripts/retrain_pipeline.py --custom-only --dry-run
```

---

## 3. Running the Retrain Pipeline

### A. Validate Your Custom Data First (Dry Run)
Checks schema, deduplication, and split counts without touching `data/processed/` or models:
```bash
python scripts/retrain_pipeline.py --dry-run
```

### B. Standard Retrain (Fresh `Qwen2.5-0.5B-Instruct`, Versioned Tag)
Wipes any stale LoRA/merged adapters, trains fresh LoRA weights from the base model, converts to `Q8_0` GGUF, registers `intent-router-qwen2.5-0.5b-instruct:v1` in Ollama, and enforces `DB_WRITE` recall (`>= 0.95`) and Macro F1 (`>= 0.90`):

#### If expanding seed_prompts.json via Gemini (and merging any data/custom/ examples):
```bash
python scripts/retrain_pipeline.py \
  --regenerate-synthetic \
  --target-total 810 \
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
#### If training strictly on your curated working_examples.json (--custom-only):
```bash
python scripts/retrain_pipeline.py \
  --custom-only \
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
(Pass --include-seeds instead of --custom-only if you want to mix seed_prompts.json directly into working_examples.json without calling Gemini; change --version v2 for future releases.)

### C. Release a New Version Without Overwriting `v1`
Pass a new `--version` tag (e.g. `v2`, `1.1.0`) so `v1` stays intact in Ollama while you test `v2`:
```bash
python scripts/retrain_pipeline.py --version v2 --custom-train-repeat 2
```

### D. Train Strictly on Your Own Custom Data (`--custom-only`)
Once you accumulate enough curated real examples (e.g., 300+ balanced across all 3 classes), drop any synthetic data completely:
```bash
python scripts/retrain_pipeline.py --custom-only --version v2
```

### E. Regenerate Synthetic Data With Expanded Seeds Before Merging
If you also add new seed prompts to `data/seed_prompts.json`:
```bash
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 900 --version v2
```

---

## 4. Pushing the Exported GGUF Model to Hugging Face

Ensure `HF_TOKEN=...` is set in `.env`, then run:
```bash
# Optional dry run first to inspect files:
python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0 --dry-run

# Upload .gguf, Modelfile, README.md model card, training_metadata.json, and evaluation_report.json with release tag:
python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0
```
*(Add `--private` for a private repo, or `--dry-run` to inspect the files first.)*

This uploads all `.gguf` files in `models/ollama_gguf/`, the `Modelfile`, `README.md` model card, `training_metadata.json`, and `evaluation_report.json` (if present).

---

## 5. Contract Preserved Across Every Retrain

No matter which base model or dataset mix you train with, the pipeline enforces the exact same runtime interface for [`app/ollama_client.py`](app/ollama_client.py) and your separate project:
1. `Modelfile` bakes in `OLLAMA_CLASSIFIER_SYSTEM_PROMPT`, `ChatML` template, `temperature 0`, `num_predict 1`, and Qwen stop tokens.
2. The model emits 1 token: `CHAT` (`GENERAL_CHAT`), `READ` (`DB_READ`), or `WRITE` (`DB_WRITE`).
3. `logprobs` on that token give calibrated probabilities for `WRITE_THRESHOLD=0.40` and `CONFIDENCE_THRESHOLD=0.60`.
