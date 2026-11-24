"""
Synthetic data generation pipeline using Google Gemini API.
Reads seed_prompts.json, generates synthetic variations per class,
flags borderline/ambiguous samples to review_queue/, and writes
deduplicated train/val/test splits to data/processed/.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass

# Try importing official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Try importing legacy google.generativeai if genai not present
if not HAS_GENAI:
    try:
        import google.generativeai as legacy_genai
        HAS_LEGACY_GENAI = True
    except ImportError:
        HAS_LEGACY_GENAI = False
else:
    HAS_LEGACY_GENAI = False


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REVIEW_DIR = DATA_DIR / "review_queue"
PROCESSED_DIR = DATA_DIR / "processed"
SEED_FILE = DATA_DIR / "seed_prompts.json"


PROMPT_TEMPLATE = """You are an expert data generation assistant for training an intent classification router in an LLM agent system.

We have 3 intent classes:
1. "READ": Information retrieval, database queries, reading customer logs, inspecting status, lookups, search.
2. "WRITE": Database mutations, record creation, updates, deletes, transactional inserts, modifying system settings.
3. "GENERAL_CHAT": Chit-chat, conversational pleasantries, greetings, open-ended trivia, philosophical questions, explanations.

Seed examples for intent '{intent}':
{seed_examples}

Generate {num_samples} realistic, diverse, and unique user prompts for the '{intent}' intent.
Include variations in:
- Sentence length (very short requests vs. detailed multi-clause sentences)
- Phrasing, verb synonyms, and colloquial tone
- Mild typos or casual chat formats
- Realistic entities (names, IDs, dates, quantities)

Also identify 1-2 borderline or ambiguous examples that might test routing boundaries (e.g. asking to preview a write action, or asking how to delete something).

Respond ONLY with valid JSON conforming to this schema:
{{
  "synthetic_prompts": [
    {{"prompt": "string", "intent": "{intent}", "confidence": 0.95}}
  ],
  "borderline_prompts": [
    {{"prompt": "string", "intended_intent": "{intent}", "reason": "string"}}
  ]
}}
"""


def load_seed_prompts() -> Dict[str, List[str]]:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed prompts file not found at {SEED_FILE}")
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def call_gemini(api_key: str, prompt: str) -> Dict[str, Any]:
    """Calls Gemini API using google-genai SDK or legacy fallback."""
    if HAS_GENAI:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            )
        )
        text = response.text
    elif HAS_LEGACY_GENAI:
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json", "temperature": 0.7}
        )
        response = model.generate_content(prompt)
        text = response.text
    else:
        raise ImportError(
            "Neither 'google-genai' nor 'google-generativeai' is installed. "
            "Please run: pip install google-genai"
        )

    # Clean potential markdown wrapping if returned
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def generate_mock_data(intent: str, seeds: List[str], count: int) -> Dict[str, Any]:
    """Generates synthetic examples deterministically if API key is not yet set."""
    templates = {
        "READ": [
            "Can you show me {}?",
            "Look up {} in the system.",
            "I need to check the status of {}.",
            "Fetch the current details for {}.",
            "Display all records related to {}.",
            "Could you pull up the info on {}?",
            "Search for {} and list the results.",
        ],
        "WRITE": [
            "Please create a new entry for {}.",
            "Update the record for {} immediately.",
            "Remove {} from the active database.",
            "Insert a new {} into our table.",
            "Modify the existing details of {}.",
            "Delete {} as requested by user.",
            "Schedule and save {} right away.",
        ],
        "GENERAL_CHAT": [
            "What do you think about {}?",
            "Tell me more about {}.",
            "How does {} work in simple terms?",
            "Write a brief summary regarding {}.",
            "Good morning! Can we talk about {}?",
            "Give me some fun trivia about {}.",
        ]
    }
    entities = [
        "account #991", "the Q4 financial report", "customer Alice Smith",
        "inventory SKU-402", "order #5821", "the payment gateway timeout",
        "our team roadmap", "the user settings profile", "invoice #8801"
    ]

    synthetic = []
    for _ in range(count):
        tpl = random.choice(templates.get(intent, ["Help with {}."]))
        ent = random.choice(entities)
        synthetic.append({
            "prompt": tpl.format(ent),
            "intent": intent,
            "confidence": round(random.uniform(0.90, 0.99), 2)
        })

    borderline = [
        {
            "prompt": f"Can you show me how to delete {random.choice(entities)}?",
            "intended_intent": intent,
            "reason": "Contains both read and write keywords; borderline routing ambiguity."
        }
    ]
    return {"synthetic_prompts": synthetic, "borderline_prompts": borderline}


def split_data(records: List[Dict[str, str]], train_ratio=0.7, val_ratio=0.15):
    """Splits records into train, val, and test partitions."""
    random.seed(42)
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


def save_jsonl(filepath: Path, records: List[Dict[str, Any]]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic intent data using Gemini API.")
    parser.add_argument("--samples-per-class", type=int, default=25, help="Number of synthetic samples per class")
    parser.add_argument("--mock", action="store_true", help="Force local synthetic mock generation without calling Gemini")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    seeds = load_seed_prompts()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_synthetic: List[Dict[str, Any]] = []
    all_borderline: List[Dict[str, Any]] = []

    # Include original seed prompts
    for intent, examples in seeds.items():
        for ex in examples:
            all_synthetic.append({"prompt": ex, "label": intent, "source": "seed"})

    print(f"Loaded {len(seeds)} classes from {SEED_FILE.name}.")

    use_mock = args.mock or not api_key or api_key == "your_gemini_api_key_here"
    if use_mock:
        print("[INFO] No valid GEMINI_API_KEY detected or --mock specified. Using local synthetic template generation.")
    else:
        print("[INFO] Connecting to Gemini API...")

    for intent, seed_list in seeds.items():
        print(f"-> Generating {args.samples_per_class} samples for class '{intent}'...")
        if use_mock:
            result = generate_mock_data(intent, seed_list, args.samples_per_class)
        else:
            seed_text = "\n".join(f"- {s}" for s in seed_list)
            prompt = PROMPT_TEMPLATE.format(
                intent=intent,
                seed_examples=seed_text,
                num_samples=args.samples_per_class
            )
            result = call_gemini(api_key, prompt)

        for p in result.get("synthetic_prompts", []):
            all_synthetic.append({
                "prompt": p["prompt"],
                "label": intent,
                "confidence": p.get("confidence", 0.95),
                "source": "gemini_synthetic" if not use_mock else "mock_synthetic"
            })

        for b in result.get("borderline_prompts", []):
            all_borderline.append({
                "prompt": b["prompt"],
                "proposed_label": b.get("intended_intent", intent),
                "reason": b.get("reason", "Ambiguous wording")
            })

    # Save raw outputs
    raw_path = RAW_DIR / "synthetic_prompts.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_synthetic, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(all_synthetic)} raw prompts to {raw_path}")

    # Save review queue
    if all_borderline:
        review_path = REVIEW_DIR / "review_needed.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(all_borderline, f, indent=2, ensure_ascii=False)
        print(f"[OK] Flagged {len(all_borderline)} borderline prompts to {review_path}")

    # Deduplicate prompts by lowercased text
    seen = set()
    deduped = []
    for item in all_synthetic:
        key = item["prompt"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append({"text": item["prompt"], "label": item["label"]})

    print(f"Total deduplicated dataset size: {len(deduped)}")

    # Split into train, val, test
    train_data, val_data, test_data = split_data(deduped)
    save_jsonl(PROCESSED_DIR / "train.jsonl", train_data)
    save_jsonl(PROCESSED_DIR / "val.jsonl", val_data)
    save_jsonl(PROCESSED_DIR / "test.jsonl", test_data)

    print(f"[OK] Successfully saved processed splits to {PROCESSED_DIR}:")
    print(f"     - train.jsonl: {len(train_data)} samples")
    print(f"     - val.jsonl:   {len(val_data)} samples")
    print(f"     - test.jsonl:  {len(test_data)} samples")


if __name__ == "__main__":
    main()
