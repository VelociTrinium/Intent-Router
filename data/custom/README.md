# Custom Working Data (`data/custom/`)

Drop or append your real classified examples into `working_examples.json` (or any `*.json` / `*.jsonl` file in this directory).

## Supported Formats

### 1. List of Objects (Recommended — `working_examples.json`)
```json
[
  {
    "prompt": "Show all Grade A offices in Singapore with rent below $110/sqft",
    "label": "DB_READ"
  },
  {
    "prompt": "Find HSBC's primary contact and update their mobile to +1-555-0192",
    "label": "DB_WRITE"
  },
  {
    "prompt": "Actually just show them first, don't delete anything yet.",
    "label": "DB_READ",
    "chat_history": [
      {"role": "user", "content": "Delete all inactive leads from 2024"},
      {"role": "assistant", "content": "Found 14 inactive leads. Proceed with deletion?"}
    ]
  },
  {
    "prompt": "How do I write a SQL query to remove duplicate contacts?",
    "label": "GENERAL_CHAT",
    "split": "test"
  }
]
```

### 2. Grouped-by-Class Dictionary (Same Format as `data/seed_prompts.json`)
```json
{
  "GENERAL_CHAT": [
    "Draft a follow-up email to Microsoft regarding their lease renewal."
  ],
  "DB_READ": [
    "Show deleted rows from the pursuits archive."
  ],
  "DB_WRITE": [
    "Mark deal #9921 as closed-won."
  ]
}
```

### 3. JSON Lines (`.jsonl`)
One JSON object per line with `"prompt"` (or `"text"`) and `"label"` (or `"intent"` / `"route"`), plus optional `"chat_history"` and `"split"`.

## Field Rules
- **Text field**: `"prompt"` or `"text"` (required, non-empty).
- **Label field**: `"label"`, `"intent"`, or `"route"` — accepted values: `GENERAL_CHAT` (`CHAT`), `DB_READ` (`READ`), `DB_WRITE` (`WRITE`).
- **Optional `chat_history`**: list of `{"role": "user" | "assistant", "content": "..."}` turns preceding the prompt.
- **Optional `split`**: `"train"`, `"val"`, or `"test"` to pin specific golden examples into a fixed split.
- **Override priority**: If a prompt (plus its `chat_history`) exists in both `data/custom/` and synthetic data (`data/raw/synthetic_prompts.json`), **your custom label always wins**.

## Running the Pipeline After Adding Data
```bash
# Dry-run validation + split preview (writes nothing)
python scripts/build_dataset.py --dry-run

# Full retrain from a fresh base model -> Q8_0 GGUF -> Ollama -> Evaluation
python scripts/retrain_pipeline.py --base-model Qwen/Qwen2.5-0.5B-Instruct --model-name intent-router
```
