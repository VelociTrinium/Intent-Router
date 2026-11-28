# Working Commands Reference (`intent-router`)

Direct execution commands covering setup, dataset generation, training, evaluation, GGUF export, Ollama serving, Hugging Face publishing, cleanup, testing, and Git.

---

## 1. Environment Setup

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```bash
# Check PyTorch device (CUDA vs CPU cores)
python -c "import os, torch; print('cuda:', torch.cuda.is_available(), 'cpu_count:', os.cpu_count())"
```

---

## 2. Ollama Service & Model Checks

```bash
# List registered Ollama models & inspect Modelfile details
ollama list
ollama show intent-router-qwen2.5-0.5b-instruct:v1

# Remove an old Ollama model tag
ollama rm intent-router-qwen2.5-0.5b-instruct:v1

# Check Ollama HTTP API tags
curl http://127.0.0.1:11434/api/tags
```

---

## 3. Dataset Inspection, Generation & Split Building

```bash
# Inspect prompt counts per class in data/seed_prompts.json
python -c "import json; d=json.load(open('data/seed_prompts.json', encoding='utf-8')); [print(k, len(v)) for k,v in d.items()]; print('total', sum(len(v) for v in d.values()))"

# Generate synthetic single-turn + multi-turn dataset via Gemini (requires GEMINI_API_KEY in .env)
python scripts/generate_data.py --target-total 1080 --multi-turn-ratio 0.25

# Dry-run validate dataset splits (writes nothing)
python scripts/build_dataset.py --include-seeds --dry-run
python scripts/build_dataset.py --custom-only --dry-run

# Option A: Build train/val/test JSONL from data/seed_prompts.json + data/custom/
python scripts/build_dataset.py --include-seeds

# Option B: Build train/val/test JSONL from data/raw/synthetic_prompts.json + data/custom/
python scripts/build_dataset.py

# Option C: Regenerate Gemini synthetic pool AND build splits in one step
python scripts/build_dataset.py --regenerate-synthetic --target-total 1080

# Option D: Build strictly from data/custom/ with 2x upsampling in train.jsonl
python scripts/build_dataset.py --custom-only --custom-train-repeat 2

# Option E: Build with explicit custom file(s) and custom split ratios
python scripts/build_dataset.py --custom-file data/custom/working_examples.json --include-seeds --train-ratio 0.75 --val-ratio 0.10 --seed 42
```

---

## 4. Training Only (`scripts/train_ollama.py`)

```bash
# Standard LoRA fine-tune from local base weights
python scripts/train_ollama.py --model-id models/hf_base/Qwen2.5-0.5B-Instruct --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4 --lora-r 16 --lora-alpha 32 --max-seq-length 384

# Higher-quality CPU-tuned training run (more optimizer steps + smoother LR)
python scripts/train_ollama.py --model-id models/hf_base/Qwen2.5-0.5B-Instruct --epochs 3 --batch-size 4 --grad-accum 2 --lr 2e-4 --lora-r 16 --lora-alpha 32 --max-seq-length 384

# Download & fine-tune directly from Hugging Face base model ID (if local hf_base absent)
python scripts/train_ollama.py --model-id Qwen/Qwen2.5-0.5B-Instruct --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4

# Quick pipeline sanity check capped at 5 optimizer steps
python scripts/train_ollama.py --max-steps 5

# Verify single-token class IDs in Qwen2.5 tokenizer
python -c "from transformers import AutoTokenizer; from scripts.train_ollama import resolve_class_token_ids; tok=AutoTokenizer.from_pretrained('models/hf_base/Qwen2.5-0.5B-Instruct'); print(resolve_class_token_ids(tok))"
```

---

## 5. GGUF Export & Ollama Registration (`scripts/export_ollama_gguf.py`)

```bash
# Export merged HF model -> Q8_0 GGUF + Modelfile and register as :v1 in Ollama
python scripts/export_ollama_gguf.py --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1

# Export higher-precision F16 GGUF and register as :v2 in Ollama
python scripts/export_ollama_gguf.py --outtype f16 --model-name intent-router-qwen2.5-0.5b-instruct --version v2

# Export GGUF + Modelfile only (skip ollama create)
python scripts/export_ollama_gguf.py --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1 --skip-ollama-create

# Export with Ollama secondary quantization (optional)
python scripts/export_ollama_gguf.py --outtype f16 --ollama-quantize q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v2
```

---

## 6. End-to-End One-Command Pipeline (`scripts/retrain_pipeline.py`)

```bash
# Dry-run validation & planned command preview
python scripts/retrain_pipeline.py --include-seeds --dry-run
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 1080 --dry-run
python scripts/retrain_pipeline.py --custom-only --dry-run

# Full run directly from curated seeds + data/custom/ (no Gemini call)
python scripts/retrain_pipeline.py --include-seeds --base-model models/hf_base/Qwen2.5-0.5B-Instruct --epochs 2 --batch-size 8 --grad-accum 2 --lr 3e-4 --lora-r 16 --outtype q8_0 --model-name intent-router-qwen2.5-0.5b-instruct --version v1

# Best-quality run with Gemini synthetic + multi-turn expansion (~1,080 samples)
python scripts/retrain_pipeline.py --regenerate-synthetic --target-total 2000 --multi-turn-ratio 0.25 --custom-train-repeat 2 --base-model models/hf_base/Qwen2.5-0.5B-Instruct --epochs 4 --batch-size 8 --grad-accum 2 --lr 2e-4 --lora-r 24 --outtype f16 --model-name intent-router-qwen2.5-0.5b-instruct --version v3

# Retrain strictly on data/custom/ as new version v2
python scripts/retrain_pipeline.py --custom-only --custom-train-repeat 2 --model-name intent-router-qwen2.5-0.5b-instruct --version v2

# Run without quality-gate eval, or customize recall/F1 gate thresholds
python scripts/retrain_pipeline.py --include-seeds --skip-eval --version v1
python scripts/retrain_pipeline.py --include-seeds --min-write-recall 0.95 --min-clarify-recall 0.90 --min-macro-f1 0.90 --version v1
```

---

## 7. Model Evaluation & Benchmarking (`scripts/evaluate.py`)

```bash
# Evaluate model configured in .env (OLLAMA_MODEL)
python scripts/evaluate.py

# Evaluate specific version
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v2

# Benchmark v2 head-to-head against v1 or untuned base qwen2.5:0.5b
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v2 --compare-untuned-ollama intent-router-qwen2.5-0.5b-instruct:v1
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v1 --compare-untuned-ollama qwen2.5:0.5b
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v3 --compare-untuned-ollama intent-router-qwen2.5-0.5b-instruct:v2 --output-report models/v3_vs_v2_report.json --fail-on-regression

# Write report to custom path
python scripts/evaluate.py --ollama-model intent-router-qwen2.5-0.5b-instruct:v2 --output-report models/evaluation_report_v2.json
```

---

## 8. Running Inference (FastAPI Server, Python Client, Direct Ollama API)

```bash
# Start FastAPI microservice
uvicorn app.main:app --host 0.0.0.0 --port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
python -m app.main
```

```bash
# Query FastAPI endpoints (Linux / macOS / Git Bash curl)
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show all companies in Singapore", "chat_history": []}'

curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Make it USD 2,500,000", "chat_history": [{"role": "user", "content": "Update pursuit #7724"}, {"role": "assistant", "content": "What deal value should I set?"}]}'
```

```powershell
# Query FastAPI endpoints (Windows PowerShell)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/route" -ContentType "application/json" -Body '{"prompt":"Update Kevin phone number","chat_history":[]}'
```

```bash
# Run standalone zero-dependency Python client
python app/ollama_client.py
python -c "from app.ollama_client import classify_intent; print(classify_intent('Show all companies in China', model='intent-router-qwen2.5-0.5b-instruct:v1'))"
```

```bash
# Direct raw Ollama /api/chat classification check (1-token + logprobs)
curl http://127.0.0.1:11434/api/chat -d '{"model": "intent-router-qwen2.5-0.5b-instruct:v1", "messages": [{"role": "user", "content": "Update Kevin phone number"}], "stream": false, "logprobs": true, "top_logprobs": 20, "options": {"temperature": 0, "num_predict": 1}}'
```

---

## 9. Hugging Face Hub Publishing & Downloading

```bash
# Dry-run verify artifacts staged for Hugging Face upload (requires repo ID)
python scripts/push_to_hf.py --repo-id VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0 --dry-run

# Push public model + release tag (uses HF_TOKEN from .env)
python scripts/push_to_hf.py --repo-id VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf --tag v1.0.0 --commit-message "Release v1.0.0"

# Push private repo with explicit token
python scripts/push_to_hf.py --repo-id VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf --private --tag v1.0.0 --hf-token <your-hf-token>

# Download exported GGUF + Modelfile on another machine and register in Ollama
huggingface-cli download VelociTrinium/intent-router-qwen2.5-0.5b-instruct-gguf <gguf-filename>.gguf Modelfile --local-dir ./intent-router-model
cd intent-router-model && ollama create intent-router-qwen2.5-0.5b-instruct:v1 -f Modelfile
```

---

## 10. Storage Cleanup (`scripts/cleanup.py`)

```bash
# Dry-run preview of reclaimable build artifacts
python scripts/cleanup.py --dry-run
python scripts/cleanup.py --include-caches --dry-run

# Remove build artifacts (checkpoints, ollama_lora, ollama_merged, ollama_gguf)
python scripts/cleanup.py --yes

# Remove build artifacts + project __pycache__/.pytest_cache (use -B to avoid new pycache)
python -B scripts/cleanup.py --include-caches --yes

# Full reset including generated data/raw + data/processed, eval reports, and cached base model
python scripts/cleanup.py --include-caches --include-data --include-reports --include-base --yes
```

---

## 11. Running Tests

```bash
# Run full test suite
pytest -v

# Run unit tests (routing guardrails, chat_history, logprobs, build_dataset, export, push, cleanup)
pytest tests/test_routing_rules.py -v

# Run API & standalone client integration tests
pytest tests/test_api.py -v
pytest tests/test_api.py -k test_standalone_ollama_client -v
```
