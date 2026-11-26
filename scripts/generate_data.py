"""
Seed-Guided Synthetic Data Generator.
Uses the official google-genai SDK with Structured Outputs to generate
a balanced, high-diversity 4-class synthetic pool (single-turn + multi-turn chat_history)
from `data/seed_prompts.json` and saves it to `data/raw/synthetic_prompts.json`.

Boundary rules enforced:
  - Conceptual / SQL syntax / drafting without DB save -> GENERAL_CHAT
  - Clear, specific fetch / search / count / inspect -> DB_READ
  - Clear, complete write / update with value / log / schedule / complete mixed READ+WRITE -> DB_WRITE
  - Vague/underspecified, missing entity or value, unresolved pronouns, destructive deletes/drops/purges, or ambiguous read-vs-write -> CLARIFICATION_REQUIRED

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
    intent: Literal["GENERAL_CHAT", "DB_READ", "DB_WRITE", "CLARIFICATION_REQUIRED"]
    confidence: float = Field(default=0.95, description="Synthetic labeling confidence score")
    variation_type: str = Field(
        default="standard",
        description=(
            "short, multi_clause, colloquial, typo, boundary, mixed_read_write, "
            "missing_details, unresolved_pronoun, destructive_confirm, or ambiguous_read_write"
        ),
    )


class SyntheticMultiTurnPrompt(BaseModel):
    chat_history: List[ChatTurn] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="2 to 4 preceding user/assistant conversation turns",
    )
    prompt: str = Field(..., description="Follow-up user message whose intent depends on chat_history")
    intent: Literal["GENERAL_CHAT", "DB_READ", "DB_WRITE", "CLARIFICATION_REQUIRED"]
    confidence: float = Field(default=0.95)
    variation_type: str = Field(default="multi_turn")


class SingleTurnBatchResponse(BaseModel):
    synthetic_prompts: List[SyntheticPrompt]


class MultiTurnBatchResponse(BaseModel):
    multi_turn_prompts: List[SyntheticMultiTurnPrompt]


PROMPT_SYSTEM_GUIDE = """You are an expert NLP data generation engine specializing in enterprise CRM and commercial real estate database intent routing (ReConnect platform / MiraChat assistant).

Classify English prompts into exactly 4 intent classes:
1. "GENERAL_CHAT" (1-token output: CHAT):
   - General conversation, greetings, pleasantries, capability questions, office announcements, HR/internal admin notices.
   - How-to questions about ReConnect workflows/fields or commercial real estate terminology (cap rate, WALE, Grade A, carpet area).
   - SQL syntax explanations, how-to tutorials, architecture advice, or writing sample SQL WITHOUT executing on a live DB.
   - Drafting/polishing emails, proposals, agendas, or summaries WITHOUT saving/logging them into the CRM.
   - Examples: "How do I delete a row in PostgreSQL?", "Explain indexing", "Can you draft a follow-up email to HSBC?", "What is a lead vs a pursuit?".

2. "DB_READ" (1-token output: READ):
   - Clear, specific requests to fetch, search, filter, count, aggregate, compare, or inspect live, archived, or deleted records/audit logs/schemas from the CRM database.
   - Explicit negative-write guards ("Don't update — only retrieve pursuit #7724's current deal value").
   - Examples: "List all clients in China", "Show deleted rows from the archive table", "What is the primary contact for Orion Financial?", "Inspect recent updates on pursuit #8436".

3. "DB_WRITE" (1-token output: WRITE):
   - Clear, actionable, non-destructive CRM mutations WITH explicit target entity and required values:
     creating companies, contacts, properties, leads, or pursuits with concrete fields; updating a named entity/ID to a concrete new value; logging calls, meetings, notes, or emails; scheduling tasks/meetings; assigning accounts to named/ID'd brokers; changing status to an explicit target.
   - Complete, specific mixed READ+WRITE compound prompts (fetching something specific AND updating/logging with concrete details in one prompt) MUST be labeled "DB_WRITE".
   - Examples: "Create a new company called Apex Digital Solutions in India and assign it to me.", "Update Kevin Sauer's mobile number at Microsoft to +1 214 555 8721.", "Find HSBC's primary contact and update their phone number to +1 555 0192.", "Log a call with Microsoft today regarding Hong Kong expansion."

4. "CLARIFICATION_REQUIRED" (1-token output: ASK):
   - Vague or underspecified requests missing the target entity, record ID, or required field/new value (e.g. "all data.", "add this company", "Add Sarah", "Update Kevin's phone number", "Change the deal value for pursuit #8436", "Need info", "Check the database", "Pursuit #8436").
   - Unresolved pronouns/anaphora in a standalone prompt without preceding context (e.g. "Update it", "Move it to next week", "Pull their details", "Do that thing").
   - Destructive, irreversible, or bulk mutations — deletes, drops, truncates, purges, wipes, mass archives/merges/overwrites/resets, or explicit confirmation requests — that require user confirmation BEFORE touching the database (e.g. "Delete Microsoft from the CRM", "Can you delete row 42 from the staging leads table?", "Archive old leads", "Drop the leads table").
   - Ambiguous read-vs-write queries or bare entity snippets with no clear operation (e.g. "Microsoft 25,000 sq ft — look up or create?", "One BKC Mumbai — search or add?", "Update or delete that lead?").

CRITICAL BOUNDARY DISTINCTIONS:
- "How do I delete row 42?" -> GENERAL_CHAT | "Show deleted rows" -> DB_READ | "Delete row 42" / "Delete Microsoft from the CRM" -> CLARIFICATION_REQUIRED.
- "Update Kevin Sauer's mobile at Microsoft to +1 214 555 8721" -> DB_WRITE | "Update Kevin's phone number" (no new number!) -> CLARIFICATION_REQUIRED | "Update it" -> CLARIFICATION_REQUIRED.
- "List all properties in BKC above 25,000 sq ft" -> DB_READ | "all data." / "One BKC Mumbai" / "Search" -> CLARIFICATION_REQUIRED.
- "Draft an email to Sarah at HSBC" -> GENERAL_CHAT | "Log an email sent today to Sarah Wong at HSBC re: lease terms" -> DB_WRITE.
"""

CLASS_SINGLE_THEMES: Dict[str, List[str]] = {
    "GENERAL_CHAT": [
        "greetings, MiraChat capability/mode questions, and casual broker banter",
        "ReConnect platform how-to questions (field meanings, workflows, lead vs pursuit conversions) without DB queries",
        "commercial real estate domain concepts (cap rate, WALE, Grade A/B, NNN lease, FSI) and office announcements/HR",
        "SQL syntax questions, sample scripts without execution, and drafting emails/agendas/proposals without CRM saving",
    ],
    "DB_READ": [
        "clear company, contact, and account-manager lookup queries across APAC/EMEA/US",
        "multi-filter commercial property and pursuit searches with explicit sq ft, city, Grade A/B, or deal value thresholds",
        "specific counts, quarterly aggregations, win-rate comparisons, and audit-log/deleted-archive inspections",
        "negative-write guards ('do not create/update/delete, only show/check...') and polite indirect specific read queries",
    ],
    "DB_WRITE": [
        "creating companies, contacts, or properties with complete concrete fields (location, email, phone, sq ft, grade)",
        "updating a named contact/pursuit/property field to an explicit new value (phone, email, stage, deal value, broker ID)",
        "logging concrete calls, meetings, notes, site visits, or scheduling dated tasks/follow-ups against named accounts",
        "specific mixed READ+WRITE compound prompts with complete details (e.g. find primary contact and update phone to X / convert lead into pursuit with value Y)",
    ],
    "CLARIFICATION_REQUIRED": [
        "underspecified write commands missing target entity or missing new value (e.g. 'Add Sarah', 'Update Kevin's phone number', 'Change pursuit #8436 deal value')",
        "vague, overly broad, or bare-fragment queries ('all data.', 'Show me the numbers', 'Check the database', 'Pursuit #8436', 'Sarah Wong +852 9123 7890')",
        "unresolved pronouns/anaphora with no prior context ('Update it', 'Move it to next week', 'Check that for me', 'Assign those to Sarah')",
        "destructive, irreversible, or bulk mutations needing confirmation ('Delete Microsoft from CRM', 'Bulk delete inactive accounts', 'Drop leads table', 'Archive old leads', 'Merge duplicates automatically')",
        "ambiguous read-vs-write prompts or multi-action uncertainty ('HSBC meeting tomorrow 3 PM — check schedule or book new?', 'Update or delete that lead?')",
    ],
}

CLASS_MULTI_TURN_SCENARIOS: Dict[str, List[str]] = {
    "GENERAL_CHAT": [
        "user pauses a CRM workflow in history and switches to asking a conceptual CRE/ReConnect question ('Wait what does WALE mean?', 'Actually explain how lead conversion works first')",
        "user pivots from looking up data to asking Mira to draft an email/summary without saving anything to the CRM ('Just draft a polite follow-up email for them, do not log it yet')",
    ],
    "DB_READ": [
        "follow-up filter tweaks or entity carry-over from prior DB_READ turns where history makes the read unambiguous ('What about Singapore?', 'Show their open pursuits instead', 'Break that down by service line')",
        "user pulls back from a suggested write in history to inspect records first ('Actually just show the 14 leads first, don't archive anything')",
    ],
    "DB_WRITE": [
        "assistant previously asked a clarifying question (missing phone/value/target) and the user supplies the concrete details so the write is now complete ('Set it to USD 2,400,000 and stage Negotiation', 'Yes, log a call with Sarah Wong at HSBC for today at 3 PM')",
        "user follows up on a looked-up record in history with an explicit, complete non-destructive update ('Update his email to rahul.mehta@microsoft.com', 'Schedule a site visit with her at Maker Maxity next Wednesday at 11 AM')",
    ],
    "CLARIFICATION_REQUIRED": [
        "assistant listed multiple matching companies/contacts/pursuits in history and the user responds ambiguously without saying which one or what value ('Update the deal value', 'Change her phone number', 'Pick that one and mark it won')",
        "even with prior browsing context, user issues a sudden destructive/bulk deletion or ambiguous read-vs-write command ('Delete all of them now', 'Purge those old records', 'Should we update or remove that contact?')",
    ],
}


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
    ref_seeds = seed_examples[-20:] if len(seed_examples) > 20 else seed_examples
    avoid_snippet = "\n".join(f"- {s}" for s in existing_samples[-20:]) if existing_samples else "(none yet)"
    user_prompt = f"""Target Intent: {intent}
Focus Variation Theme: {sub_theme}

Seed Examples for reference (sample):
{chr(10).join(f"- {s}" for s in ref_seeds)}

Do NOT repeat or closely paraphrase these already-generated prompts:
{avoid_snippet}

Generate {batch_size} unique, highly realistic, diverse user prompts strictly belonging to the '{intent}' class.
Ensure wide variety in:
- Writing style from very loose/colloquial/typo-tolerant shorthand (2 to 6 words) to formal multi-clause enterprise requests.
- Realistic commercial real estate & CRM entities (brokers, companies across APAC/EMEA/US, sq ft ranges, Grade A/B buildings, lease terms, deal IDs, currencies, phones/emails).
- Strict adherence to the 4-class boundary rules above (e.g. do NOT generate incomplete updates or destructive deletes for DB_WRITE; keep those in CLARIFICATION_REQUIRED).
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
- The final `prompt` MUST be a natural follow-up whose true 4-class intent ({intent}) depends meaningfully on `chat_history` (e.g., resolved reference -> DB_READ/DB_WRITE; still ambiguous/destructive -> CLARIFICATION_REQUIRED; conceptual pivot -> GENERAL_CHAT).
- All `intent` fields in `multi_turn_prompts` must be '{intent}'.
"""
    return _generate_with_models(client, f"{PROMPT_SYSTEM_GUIDE}\n\n{user_prompt}", MultiTurnBatchResponse)


def main():
    parser = argparse.ArgumentParser(description="Seed-Guided 4-Class Synthetic Data Generator with Gemini API.")
    parser.add_argument("--target-total", type=int, default=1080, help="Target dataset size across 4 classes (default: 1080 = 270/class)")
    parser.add_argument("--multi-turn-ratio", type=float, default=0.25, help="Fraction of multi-turn chat_history samples per class")
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
    print(f"[INFO] Target per class: ~{per_class_target} ({per_class_target * len(CLASSES)} total across {len(CLASSES)} classes, multi_turn_ratio={args.multi_turn_ratio})")

    all_records: List[Dict[str, Any]] = []
    for intent, items in seeds.items():
        for prompt_text in items:
            all_records.append({
                "text": prompt_text.strip(),
                "label": intent,
                "source": "seed",
            })

    for intent in CLASSES:
        seed_list = seeds.get(intent, [])
        existing_texts = [r["text"] for r in all_records if r["label"] == intent]
        current_count = len(existing_texts)
        needed_total = max(0, per_class_target - current_count)
        needed_multi = min(needed_total, int(per_class_target * args.multi_turn_ratio))
        needed_single = max(0, needed_total - needed_multi)

        print(f"\n-> Class '{intent}': seeds={current_count}, single_target={needed_single}, multi_target={needed_multi}")

        themes = CLASS_SINGLE_THEMES[intent]
        if needed_single > 0:
            batch_size = max(15, (needed_single + len(themes) - 1) // len(themes))
            gen_single = 0
            for theme in themes:
                if gen_single >= needed_single:
                    break
                req_count = min(batch_size, needed_single - gen_single)
                print(f"   [Single-Turn] Theme: '{theme[:58]}...' | Requesting {req_count}...")
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

        scenarios = CLASS_MULTI_TURN_SCENARIOS[intent]
        if needed_multi > 0:
            mt_batch = max(15, (needed_multi + len(scenarios) - 1) // len(scenarios))
            gen_multi = 0
            for scenario in scenarios:
                if gen_multi >= needed_multi:
                    break
                req_count = min(mt_batch, needed_multi - gen_multi)
                print(f"   [Multi-Turn] Scenario: '{scenario[:58]}...' | Requesting {req_count}...")
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
