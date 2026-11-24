# Intent Router

A production-grade, low-latency intent router microservice designed for multi-agent LLM systems. Routes incoming user requests to the appropriate processing pipeline (`READ`, `WRITE`, or `GENERAL_CHAT`) using a few-shot SetFit classifier, a fast TF-IDF baseline, or an Ollama local LLM backend.

---

## Architecture & Project Structure

```
intent-router/
├── .env                          # Put GEMINI_API_KEY=... and HF_TOKEN=... here (gitignored)
├── .env.example                  # Committed template showing required env vars
├── .gitignore                    # Ignore .env, venv/, __pycache__/, models/, data/processed/
├── pyproject.toml                # Dependencies & project metadata
├── requirements.txt              # Standard pip dependencies
├── README.md                     # Setup, pipeline commands, and API usage
├── data/
│   ├── seed_prompts.json         # 5–6 hand-written seed examples per class
│   ├── raw/                      # Gemini-generated synthetic prompts before human review
│   ├── review_queue/             # Borderline/uncertain prompts flagged for your CLI check
│   └── processed/                # Final deduplicated train.jsonl, val.jsonl, test.jsonl
├── models/
│   ├── baseline_tfidf/           # Saved scikit-learn TF-IDF + LogisticRegression artifact
│   └── intent_router/            # Saved SetFit model weights, config, and label_map.json
├── scripts/
│   ├── __init__.py
│   ├── generate_data.py          # Calls Gemini API using seed_prompts.json -> data/raw/
│   ├── review_uncertain.py       # Interactive CLI for you to verify uncertain labels
│   ├── train_baseline.py         # Trains TF-IDF + LogisticRegression benchmark
│   ├── train_setfit.py           # Trains SetFit model + optional --push-to-hub flag
│   └── evaluate.py               # Outputs Precision/Recall/F1, 3x3 confusion matrix, latency
├── app/
│   ├── __init__.py
│   ├── config.py                 # Thresholds (CONFIDENCE_THRESHOLD=0.60, WRITE_THRESHOLD=0.40) & backend toggle
│   ├── schemas.py                # Pydantic request/response models (prompt + optional chat_history)
│   ├── router_engine.py          # Unified interface switching between SetFitRouter and OllamaRouter
│   └── main.py                   # FastAPI entrypoint exposing POST /route and GET /health
└── tests/
    ├── __init__.py
    ├── test_routing_rules.py     # Unit tests for fallback-to-GENERAL_CHAT and write-threshold logic
    └── test_api.py               # FastAPI TestClient checks for /route
```

---

## Intent Classes & Routing Logic

### Intent Classes
1. **`READ`**: Retrieval, lookups, query requests, data inspection, and read-only searches.
2. **`WRITE`**: Mutations, creation, updates, deletes, and action execution.
3. **`GENERAL_CHAT`**: Conversational chit-chat, greetings, open-ended questions, and fallback.

### Decision Rules
- **Write Sensitivity Rule**: If `probability(WRITE) >= WRITE_THRESHOLD` (default: `0.40`), the request is routed to `WRITE` immediately. Mutating actions carry high risk/importance, so a dedicated threshold prevents accidental misrouting to chat or read.
- **Confidence Fallback Rule**: If the highest predicted probability is `< CONFIDENCE_THRESHOLD` (default: `0.60`), the system safely falls back to `GENERAL_CHAT`.
- **Primary Prediction Rule**: Otherwise, routes to the intent with the highest probability.

---

## Quickstart & Setup

### 1. Environment Setup

```bash
cd intent-router

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or with editable install:
pip install -e .
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your credentials:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
ROUTER_BACKEND=setfit
CONFIDENCE_THRESHOLD=0.60
WRITE_THRESHOLD=0.40
```

---

## Data Pipeline & Model Training

### Step 1: Generate Synthetic Data
Generates realistic variations across all 3 classes from `data/seed_prompts.json` using the Google Gemini API:
```bash
python scripts/generate_data.py --samples-per-class 30
```
- Raw prompts are saved in `data/raw/synthetic_prompts.json`.
- Borderline/uncertain examples are automatically routed to `data/review_queue/`.
- Clean splits are written to `data/processed/train.jsonl`, `val.jsonl`, and `test.jsonl`.

### Step 2: Interactive Human-in-the-Loop Review
Review borderline or uncertain prompts via the CLI:
```bash
python scripts/review_uncertain.py
```
Options in the CLI:
- `[1]` Keep proposed label
- `[2]` Re-label as READ
- `[3]` Re-label as WRITE
- `[4]` Re-label as GENERAL_CHAT
- `[d]` Discard prompt
- `[s]` Skip

### Step 3: Train Baseline Benchmark (TF-IDF + Logistic Regression)
```bash
python scripts/train_baseline.py
```
Saves artifacts to `models/baseline_tfidf/`.

### Step 4: Train SetFit Classifier
Trains the few-shot sentence transformer SetFit model:
```bash
python scripts/train_setfit.py --model-id sentence-transformers/all-MiniLM-L6-v2 --epochs 2
```
To push to the Hugging Face Hub:
```bash
python scripts/train_setfit.py --push-to-hub --repo-id your-org/intent-router
```
Saves artifacts and label mapping to `models/intent_router/`.

### Step 5: Evaluate Models
Evaluates against the held-out `data/processed/test.jsonl` set:
```bash
# Evaluate SetFit
python scripts/evaluate.py --backend setfit

# Evaluate Baseline
python scripts/evaluate.py --backend baseline
```
Outputs:
- Inference Latency metrics (mean, p50, p95)
- Per-class Precision, Recall, and F1-score
- Complete 3x3 Confusion Matrix

---

## Running the FastAPI Microservice

Start the API server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at `http://localhost:8000/docs`.

### API Endpoints

#### 1. Health Check: `GET /health`
```bash
curl -X GET http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "backend": "setfit",
  "model_loaded": true,
  "confidence_threshold": 0.6,
  "write_threshold": 0.4
}
```

#### 2. Route Intent: `POST /route`
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Can you delete order #4912 from the database?",
    "chat_history": []
  }'
```
Response:
```json
{
  "intent": "WRITE",
  "confidence": 0.942,
  "raw_intent": "WRITE",
  "probabilities": {
    "READ": 0.031,
    "WRITE": 0.942,
    "GENERAL_CHAT": 0.027
  },
  "backend": "setfit",
  "routed_by_rule": "write_threshold_triggered",
  "latency_ms": 11.45
}
```

---

## Testing

Run unit tests and API integration tests:
```bash
pytest -v
```
Tests cover:
- Confidence threshold fallback to `GENERAL_CHAT`
- `WRITE_THRESHOLD=0.40` prioritization logic
- Conversational chat history context handling
- FastAPI `/health` and `/route` endpoint responses and status codes
