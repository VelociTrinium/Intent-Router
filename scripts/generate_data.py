"""
Seed-Guided Synthetic Data Generator.
Uses the official google-genai SDK with Structured Outputs to generate
a balanced, high-diversity synthetic pool (single-turn + multi-turn chat_history)
from `data/seed_prompts.json` and saves it to `data/raw/synthetic_prompts.json`.

Boundary rules enforced:
  - "How do I delete a row?" / drafting emails without DB save -> GENERAL_CHAT
  - "Show deleted rows" / inspect audit logs -> DB_READ
  - "Delete row 42" / "Can you delete row 42?" / mixed READ+WRITE -> DB_WRITE

Run `python scripts/build_dataset.py` afterwards (or pass `--regenerate-synthetic`
to `build_dataset.py` / `retrain_pipeline.py`) to merge with `data/custom/` and split
into `data/processed/{train,val,test}.jsonl`.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.intent_spec import CLASSES

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SEED_FILE = DATA_DIR / "seed_prompts.json"

PREFERRED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SyntheticPrompt(BaseModel):
    prompt: str = Field(..., description="Realistic user query or prompt")
    intent: Literal["GENERAL_CHAT", "DB_READ", "DB_WRITE"]
    confidence: float = Field(default=0.95, description="Synthetic labeling confidence score")
    variation_type: str = Field(
        default="standard",
        description="short, multi_clause, colloquial, typo, boundary, or mixed_read_write",
    )


class SyntheticMultiTurnPrompt(BaseModel):
    chat_history: List[ChatTurn] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="2 to 4 preceding user/assistant conversation turns",
    )
    prompt: str = Field(..., description="Follow-up user message whose intent depends on chat_history")
    intent: Literal["GENERAL_CHAT", "DB_READ", "DB_WRITE"]
    confidence: float = Field(default=0.95)
    variation_type: str = Field(default="multi_turn")


class SingleTurnBatchResponse(BaseModel):
    synthetic_prompts: List[SyntheticPrompt]


class MultiTurnBatchResponse(BaseModel):
    multi_turn_prompts: List[SyntheticMultiTurnPrompt]


PROMPT_SYSTEM_GUIDE = """You are an expert NLP data generation engine specializing in enterprise CRM and commercial real estate database intent routing (ReConnect / Mira assistant).

Classify English prompts into exactly 3 intent classes:
1. "GENERAL_CHAT":
   - General conversation, greetings, pleasantries, office announcements, conceptual questions.
   - SQL syntax explanations, how-to tutorials, architecture advice, or writing sample SQL WITHOUT executing on live DB.
   - Drafting emails, proposals, or summaries WITHOUT saving/logging them into the database.
   - Examples: "How do I delete a row in PostgreSQL?", "Explain indexing", "Can you draft a follow-up email to HSBC?", "Write an example SQL query to update user email".

2. "DB_READ":
   - Requests to fetch, search, filter, count, aggregate, compare, or inspect live, archived, or deleted records/audit logs/schemas from the database.
   - Examples: "Show deleted rows", "List all clients in China", "What is the primary contact for Acme Corp?", "Inspect recent updates on deal #8436".

3. "DB_WRITE":
   - Requests (direct imperative OR polite/indirect like "Can you...", "Please...") to insert, create, update, delete, archive, assign, log a call/note, schedule a meeting, or mutate live records/state in the database.
   - Mixed READ+WRITE compound prompts (fetching something AND creating/updating/deleting something in the same request) MUST be labeled "DB_WRITE".
   - Examples: "Delete row 42", "Can you remove contact #8436?", "Find HSBC's primary contact and update their mobile to +1-555-0192", "Log a call with Microsoft".

CRITICAL BOUNDARY DISTINCTIONS:
- "How do I delete a row?" -> GENERAL_CHAT | "Show deleted rows" -> DB_READ | "Delete row 42" / "Can you delete row 42?" -> DB_WRITE.
- "Draft an email to Sarah at HSBC" -> GENERAL_CHAT | "Log an email/call with Sarah at HSBC in the CRM" -> DB_WRITE.
- "Find all inactive leads" -> DB_READ | "Find all inactive leads and mark them archived" -> DB_WRITE.
"""


def load_seed_prompts() -> Dict[str, List[str]]:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found at {SEED_FILE}")
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in CLASSES}


def _generate_with_models(client: Any, prompt_text: str, schema_cls: Any, temperature: float = 0.8) -> Any:
    from google.genai import types

    last_err = None
    for model_name in PREFERRED_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt_text)],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema_cls,
                        temperature=temperature,
                    ),
                )
                data = json.loads(response.text)
                return schema_cls(**data)
            except Exception as e:
                last_err = e
                print(f"   [WARN] Model '{model_name}' attempt {attempt + 1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"All Gemini models failed: {last_err}")


def call_gemini_single_turn(
    client: Any,
    intent: str,
    seed_examples: List[str],
    batch_size: int,
    sub_theme: str,
    existing_samples: List[str],
) -> SingleTurnBatchResponse:
    avoid_snippet = "\n".join(f"- {s}" for s in existing_samples[-15:]) if existing_samples else "(none yet)"
    user_prompt = f"""Target Intent: {intent}
Focus Variation Theme: {sub_theme}

Seed Examples for reference:
{chr(10).join(f"- {s}" for s in seed_examples)}

Do NOT repeat or closely paraphrase these already-generated prompts:
{avoid_snippet}

Generate {batch_size} unique, highly realistic, diverse user prompts for the '{intent}' class.
Ensure wide variety in:
- Sentence structure, length (from 2-word terse commands to detailed multi-clause requests), and tone (direct, polite "can you/could you", urgent, colloquial, occasional mild typos).
- Realistic commercial real estate & CRM entities (broker names, companies across APAC/EMEA/US, sq ft areas, Grade A/B buildings, lease terms, deal IDs, dates, currencies, phone/email formats).
- Realistic boundary cases that strictly obey the intent rules above.
"""
    return _generate_with_models(client, f"{PROMPT_SYSTEM_GUIDE}\n\n{user_prompt}", SingleTurnBatchResponse)


def call_gemini_multi_turn(
    client: Any,
    intent: str,
    batch_size: int,
    scenario_focus: str,
) -> MultiTurnBatchResponse:
    user_prompt = f"""Target Intent for the FINAL user turn: {intent}
Scenario Focus: {scenario_focus}

Generate {batch_size} unique multi-turn conversations (2 to 4 alternating user/assistant turns in `chat_history`, plus a final `prompt` from the user).
CRITICAL REQUIREMENT:
- The final `prompt` MUST be a natural follow-up (e.g., short confirmation like "Yes, go ahead", location/filter tweak like "What about Singapore?", entity pronoun like "Update his email to...", pivot like "Actually just show them first, don't delete", or conceptual tangent like "Wait, what does Grade A mean?").
- Its true intent ({intent}) should depend meaningfully on `chat_history` so the model learns to read context rather than guessing blindly.
- All `intent` fields in `multi_turn_prompts` must be '{intent}'.
"""
    return _generate_with_models(client, f"{PROMPT_SYSTEM_GUIDE}\n\n{user_prompt}", MultiTurnBatchResponse)


def main():
    parser = argparse.ArgumentParser(description="Seed-Guided Synthetic Data Generator with Gemini API.")
    parser.add_argument("--target-total", type=int, default=810, help="Target dataset size across 3 classes")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.25, help="Fraction of multi-turn chat_history samples")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError("Valid GEMINI_API_KEY not found in .env or environment.")

    from google import genai
    client = genai.Client(api_key=api_key)
    print("[INFO] Connected to Google GenAI SDK.")

    seeds = load_seed_prompts()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    per_class_target = max(60, args.target_total // len(CLASSES))
    print(f"[INFO] Target per class: ~{per_class_target} ({per_class_target * len(CLASSES)} total, multi_turn_ratio={args.multi_turn_ratio})")

    all_records: List[Dict[str, Any]] = []
    for intent, items in seeds.items():
        for prompt_text in items:
            all_records.append({
                "text": prompt_text.strip(),
                "label": intent,
                "source": "seed",
            })

    single_themes = [
        "enterprise CRM workflows: companies, contacts, brokers, deals, pursuits, and commercial properties",
        "tricky boundaries: asking how to mutate/delete vs showing deleted/audit records vs executing mutations",
        "mixed READ+WRITE compound prompts (e.g. find X and update Y) and polite/indirect write requests ('can you', 'please')",
        "short, terse, colloquial, and typo-tolerant user prompts (2 to 6 words)",
        "complex multi-clause prompts with filters, sq ft ranges, cities, dates, and deal values",
    ]
    multi_turn_scenarios = [
        "pronoun and entity carry-over across 2-4 turns (e.g. 'Update his phone', 'What about Singapore?', 'Show their pursuits')",
        "confirmation or rejection/pivot after assistant clarification (e.g. 'Yes go ahead', 'Actually just show them first, don't delete', 'Wait explain how that works first')",
    ]

    for intent in CLASSES:
        seed_list = seeds.get(intent, [])
        existing_texts = [r["text"] for r in all_records if r["label"] == intent]
        current_count = len(existing_texts)
        needed_total = max(0, per_class_target - current_count)
        needed_multi = int(per_class_target * args.multi_turn_ratio)
        needed_single = max(0, needed_total - needed_multi)

        print(f"\n-> Class '{intent}': seeds={current_count}, single_target={needed_single}, multi_target={needed_multi}")

        # 1. Single-turn generation
        batch_size = max(25, (needed_single + len(single_themes) - 1) // len(single_themes))
        gen_single = 0
        for theme in single_themes:
            if gen_single >= needed_single:
                break
            req_count = min(batch_size, needed_single - gen_single)
            print(f"   [Single-Turn] Theme: '{theme[:55]}...' | Requesting {req_count}...")
            resp = call_gemini_single_turn(client, intent, seed_list, req_count, theme, existing_texts)
            for sp in resp.synthetic_prompts:
                txt = sp.prompt.strip()
                all_records.append({
                    "text": txt,
                    "label": intent,
                    "variation_type": sp.variation_type,
                    "confidence": sp.confidence,
                    "source": "gemini_synthetic",
                })
                existing_texts.append(txt)
                gen_single += 1

        # 2. Multi-turn generation
        if needed_multi > 0:
            mt_batch = max(15, (needed_multi + len(multi_turn_scenarios) - 1) // len(multi_turn_scenarios))
            gen_multi = 0
            for scenario in multi_turn_scenarios:
                if gen_multi >= needed_multi:
                    break
                req_count = min(mt_batch, needed_multi - gen_multi)
                print(f"   [Multi-Turn] Scenario: '{scenario[:55]}...' | Requesting {req_count}...")
                mt_resp = call_gemini_multi_turn(client, intent, req_count, scenario)
                for mp in mt_resp.multi_turn_prompts:
                    hist = [{"role": t.role, "content": t.content.strip()} for t in mp.chat_history if t.content.strip()]
                    if not hist:
                        continue
                    all_records.append({
                        "text": mp.prompt.strip(),
                        "chat_history": hist,
                        "label": intent,
                        "variation_type": mp.variation_type,
                        "confidence": mp.confidence,
                        "source": "gemini_multi_turn",
                    })
                    gen_multi += 1

    raw_file = RAW_DIR / "synthetic_prompts.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved {len(all_records)} raw records to {raw_file}")
    print("[INFO] Run `python scripts/build_dataset.py` to deduplicate and build train/val/test JSONL splits.")


if __name__ == "__main__":
    main()
