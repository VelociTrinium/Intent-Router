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
                        scripts/evaluate.py --backend ollama ──▶ models/evaluation_report.json
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
- **Preserved human reviews**: Verified items from `scripts/review_uncertain.py` are stored in `data/raw/verified_reviews.json` and automatically kept in `train.jsonl`.
- **Optional weighting (`--custom-train-repeat`)**: If you have e.g. 50 real production examples alongside ~800 synthetic examples, `--custom-train-repeat 2` or `3` repeats your real examples inside `train.jsonl` only (never in `val.jsonl` or `test.jsonl`).

---

## 3. Running the Retrain Pipeline

### A. Validate Your Custom Data First (Dry Run)
Checks schema, deduplication, and split counts without touching `data/processed/` or models:
```bash
python scripts/retrain_pipeline.py --dry-run
```

### B. Standard Retrain (Custom + Synthetic Pool, Fresh `Qwen2.5-0.5B-Instruct`)
Wipes any stale LoRA/merged adapters, trains fresh LoRA weights from the base model, converts to `Q8_0` GGUF, updates `intent-router` in Ollama, and enforces `DB_WRITE` recall (`>= 0.95`) and Macro F1 (`>= 0.90`):
```bash
python scripts/retrain_pipeline.py --base-model Qwen/Qwen2.5-0.5B-Instruct --model-name intent-router
```

### C. Test a New Candidate Tag Before Overwriting Production
Register as `intent-router-v2` first so your running `intent-router` stays untouched until you verify:
```bash
python scripts/retrain_pipeline.py --model-name intent-router-v2 --custom-train-repeat 2
```

### D. Train Strictly on Your Own Custom Data (`--custom-only`)
Once you accumulate enough curated real examples (e.g., 300+ balanced across all 3 classes), drop the synthetic data completely:
```bash
python scripts/retrain_pipeline.py --custom-only --model-name intent-router
```

### E. Swap to a Different Fresh Base Model
Any causal instruct model whose tokenizer encodes `CHAT`, `READ`, and `WRITE` as single tokens works cleanly (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-0.6B`, `HuggingFaceTB/SmolLM2-360M-Instruct`, etc.). `scripts/train_ollama.py` and `scripts/export_ollama_gguf.py` auto-detect the chat template (`chatml`, `llama3`, or `gemma`) and preserve the exact same `CHAT`/`READ`/`WRITE` output contract:
```bash
python scripts/retrain_pipeline.py --base-model Qwen/Qwen2.5-1.5B-Instruct --model-name intent-router-1.5b
```

### F. Regenerate Synthetic Data With Expanded Seeds Before Merging
If you also add new seed prompts to `data/seed_prompts.json`:
```bash
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 900 --model-name intent-router
```

---

## 4. Contract Preserved Across Every Retrain

No matter which base model or dataset mix you train with, the pipeline enforces the exact same runtime interface for [`app/ollama_client.py`](app/ollama_client.py) and your separate project:
1. `Modelfile` bakes in `OLLAMA_CLASSIFIER_SYSTEM_PROMPT`, `temperature 0`, `num_predict 1`, and architecture-matched stop tokens.
2. The model emits 1 token: `CHAT` (`GENERAL_CHAT`), `READ` (`DB_READ`), or `WRITE` (`DB_WRITE`).
3. `logprobs` on that token give calibrated probabilities for `WRITE_THRESHOLD=0.40` and `CONFIDENCE_THRESHOLD=0.60`.
