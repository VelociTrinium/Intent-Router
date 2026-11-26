# Retraining & Custom Data Pipeline Guide

This guide explains how to add your own classified working data in JSON, merge seeds or synthetic examples across all **4 classes** (`GENERAL_CHAT` / `CHAT`, `DB_READ` / `READ`, `DB_WRITE` / `WRITE`, `CLARIFICATION_REQUIRED` / `ASK`), and retrain a **fresh base model** from scratch into Ollama with `logprobs` guardrails.

---

## 1. End-to-End Flow

```text
data/custom/*.json(l)  ──┐
data/seed_prompts.json ──┼──▶ scripts/build_dataset.py ──▶ data/processed/{train,val,test}.jsonl
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

### Key Guarantees in [`scripts/build_dataset.py`](scripts/build_dataset.py)
- **Strict validation**: Rejects missing text, invalid labels (`GENERAL_CHAT`/`CHAT`, `DB_READ`/`READ`, `DB_WRITE`/`WRITE`, `CLARIFICATION_REQUIRED`/`ASK`), or malformed `chat_history` before training starts.
- **Custom label always wins**: If an example in `data/custom/` matches a prompt in `data/raw/synthetic_prompts.json` or `data/seed_prompts.json`, your custom label overrides it.
- **Pinned evaluation safety**: Adding `"split": "test"` (or `"val"` / `"train"`) locks golden test cases into `test.jsonl` so they never leak into training.
- **Optional weighting (`--custom-train-repeat`)**: Repeats your curated real examples inside `train.jsonl` only (never in `val.jsonl` or `test.jsonl`).

---

### Preview & Validate Splits (Dry Run)
```bash
# Option A: Curated data/seed_prompts.json (649 seeds across 4 classes) + data/custom/
python scripts/retrain_pipeline.py --include-seeds --dry-run

# Option B: Seeds expanded via Gemini (1,080 balanced single-turn + multi-turn across 4 classes)
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 1080 --dry-run

# Option C: Strictly your custom JSON examples
python scripts/retrain_pipeline.py --custom-only --dry-run
```

---

## 3. Running the Retrain Pipeline

### A. Retrain Directly from `seed_prompts.json` (+ `data/custom/`, No Gemini Required)
Trains a fresh `Qwen2.5-0.5B-Instruct`, converts to `Q8_0` GGUF, registers `intent-router-qwen2.5-0.5b-instruct:v1` in Ollama, and enforces `DB_WRITE` recall (`>= 0.95`), `CLARIFICATION_REQUIRED` recall (`>= 0.90`), and Macro F1 (`>= 0.90`):

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
```bash
python scripts/retrain_pipeline.py --include-seeds --base-model models/hf_base/Qwen2.5-0.5B-Instruct --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4 --lora-r 16 --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1
```

### B. Retrain With Gemini Synthetic + Multi-Turn Expansion (~1,080 Balanced Across 4 Classes)
Adds single-turn & multi-turn `chat_history` examples on top of `seed_prompts.json` (and merges `data/custom/`):
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

### C. Release a New Version Without Overwriting `v1`
Pass a new `--version` tag (e.g. `v2`, `1.1.0`) so `v1` stays intact in Ollama while you test `v2`:
```bash
python scripts/retrain_pipeline.py --include-seeds --version v2 --custom-train-repeat 2
```

### D. Train Strictly on Your Own Custom Data (`--custom-only`)
Once you accumulate enough curated real examples across all 4 classes:
```bash
python scripts/retrain_pipeline.py --custom-only --version v2
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

---

## 5. Contract Preserved Across Every Retrain

No matter which dataset mix you train with, the pipeline enforces the exact same runtime interface for [`app/ollama_client.py`](app/ollama_client.py) and your separate project:
1. `Modelfile` bakes in `OLLAMA_CLASSIFIER_SYSTEM_PROMPT`, `ChatML` template, `temperature 0`, `num_predict 1`, and Qwen stop tokens.
2. The model emits 1 token: `CHAT` (`GENERAL_CHAT`), `READ` (`DB_READ`), `WRITE` (`DB_WRITE`), or `ASK` (`CLARIFICATION_REQUIRED`).
3. `logprobs` on that token give calibrated probabilities for:
   - `CLARIFY_THRESHOLD=0.40` (clarification beats `DB_WRITE` when stronger or tied),
   - `WRITE_THRESHOLD=0.40` (mixed read+write safety catch),
   - `CONFIDENCE_THRESHOLD=0.60` (`FALLBACK_ROUTE=CLARIFICATION_REQUIRED`).
