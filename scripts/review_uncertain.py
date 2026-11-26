"""
Interactive Human-in-the-Loop (HITL) Review CLI.
Presents uncertain, ambiguous, or borderline prompts to a human reviewer to confirm
or override the label before final model training:
  0: GENERAL_CHAT
  1: DB_READ
  2: DB_WRITE
  s: skip
  d: discard
  q: save & quit

Saves verified corrections directly into data/processed/train.jsonl.
Can also scan existing datasets with a preliminary baseline model to find borderline cases.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REVIEW_QUEUE_FILE = DATA_DIR / "review_queue" / "review_needed.json"
TRAIN_FILE = DATA_DIR / "processed" / "train.jsonl"
VERIFIED_REVIEWS_FILE = DATA_DIR / "raw" / "verified_reviews.json"
BASELINE_MODEL_FILE = BASE_DIR / "models" / "baseline_tfidf" / "baseline_model.joblib"

LABEL_OPTIONS = {
    "0": "GENERAL_CHAT",
    "1": "DB_READ",
    "2": "DB_WRITE"
}


def load_review_queue() -> List[Dict[str, Any]]:
    if not REVIEW_QUEUE_FILE.exists():
        return []
    with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_review_queue(items: List[Dict[str, Any]]):
    REVIEW_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def append_verified_to_train(prompt: str, label: str):
    """Appends verified sample to train.jsonl and ensures no duplicates."""
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing_lines = []
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            existing_lines = [json.loads(line) for line in f if line.strip()]

    # Check if prompt already exists (single-turn); if so, update label
    updated = False
    norm_prompt = prompt.strip().lower()
    for row in existing_lines:
        if not row.get("chat_history") and row.get("text", "").strip().lower() == norm_prompt:
            row["label"] = label
            row["reviewed_by_human"] = True
            updated = True
            break

    if not updated:
        existing_lines.append({
            "text": prompt.strip(),
            "label": label,
            "reviewed_by_human": True
        })

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for row in existing_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Persist verified reviews separately so build_dataset.py retains them across rebuilds
    VERIFIED_REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    reviews = []
    if VERIFIED_REVIEWS_FILE.exists():
        try:
            reviews = json.loads(VERIFIED_REVIEWS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reviews = []
    rev_updated = False
    for r in reviews:
        if not r.get("chat_history") and r.get("text", "").strip().lower() == norm_prompt:
            r["label"] = label
            rev_updated = True
            break
    if not rev_updated:
        reviews.append({"text": prompt.strip(), "label": label, "source": "human_reviewed"})
    VERIFIED_REVIEWS_FILE.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")


def scan_dataset_for_uncertainty(file_path: Path, margin_threshold: float = 0.20) -> List[Dict[str, Any]]:
    """
    Uses preliminary baseline model to compute margin between top-1 and top-2
    probabilities. If margin < margin_threshold, flags prompt for review.
    """
    if not BASELINE_MODEL_FILE.exists():
        print(f"[WARN] No baseline model found at {BASELINE_MODEL_FILE}. Cannot scan dataset.")
        return []

    import joblib
    pipeline = joblib.load(BASELINE_MODEL_FILE)
    classes = list(pipeline.classes_)

    uncertain = []
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    for r in records:
        text = r["text"]
        probs = pipeline.predict_proba([text])[0]
        sorted_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top1_cls, top1_prob = sorted_probs[0]
        top2_cls, top2_prob = sorted_probs[1]
        margin = top1_prob - top2_prob

        if margin < margin_threshold or top1_prob < 0.60:
            uncertain.append({
                "prompt": text,
                "proposed_label": top1_cls,
                "reason": (
                    f"Low model certainty: {top1_cls} ({top1_prob:.2f}) vs "
                    f"{top2_cls} ({top2_prob:.2f}), margin={margin:.2f}"
                )
            })

    return uncertain


def display_prompt(index: int, total: int, item: Dict[str, Any]):
    prompt_text = item.get("prompt", "")
    proposed = item.get("proposed_label", "GENERAL_CHAT")
    reason = item.get("reason", "Borderline or ambiguous")

    if HAS_RICH:
        content = (
            f"[bold cyan]Prompt:[/bold cyan] {prompt_text}\n\n"
            f"[bold yellow]Proposed Label:[/bold yellow] [bold green]{proposed}[/bold green]\n"
            f"[dim]Flag Reason:[/dim] {reason}"
        )
        console.print(Panel(content, title=f"Human Review Item [{index + 1}/{total}]", expand=False))
    else:
        print("\n" + "=" * 65)
        print(f"Human Review Item [{index + 1}/{total}]")
        print(f"Prompt:         {prompt_text}")
        print(f"Proposed Label: {proposed}")
        print(f"Flag Reason:    {reason}")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Human-in-the-Loop review for borderline/uncertain prompts.")
    parser.add_argument("--scan-file", type=str, default=None, help="Scan a jsonl file for low-certainty samples")
    parser.add_argument("--non-interactive", action="store_true", help="Auto-accept proposed labels without prompting")
    args = parser.parse_args()

    queue: List[Dict[str, Any]] = []

    # If scan requested, add flagged samples
    if args.scan_file:
        scan_path = Path(args.scan_file)
        print(f"[INFO] Scanning {scan_path} for low-confidence samples...")
        scanned = scan_dataset_for_uncertainty(scan_path)
        print(f"[INFO] Found {len(scanned)} borderline items from scan.")
        queue.extend(scanned)

    # Load existing review queue
    existing_queue = load_review_queue()
    queue.extend(existing_queue)

    # Deduplicate queue
    seen = set()
    deduped_queue = []
    for item in queue:
        p = item.get("prompt", "").strip().lower()
        if p and p not in seen:
            seen.add(p)
            deduped_queue.append(item)
    queue = deduped_queue

    if not queue:
        msg = "The review queue is currently empty. No borderline prompts require review!"
        if HAS_RICH:
            console.print(f"[bold green]{msg}[/bold green]")
        else:
            print(msg)
        return

    print(f"\nLoaded {len(queue)} borderline prompts for human review.\n")

    accepted_count = 0
    overridden_count = 0
    discarded_count = 0
    remaining: List[Dict[str, Any]] = []

    valid_choices = ["0", "1", "2", "s", "d", "q", ""]

    for idx, item in enumerate(queue):
        display_prompt(idx, len(queue), item)
        proposed = item.get("proposed_label", "GENERAL_CHAT")

        if args.non_interactive:
            # Auto-accept in non-interactive mode
            append_verified_to_train(item["prompt"], proposed)
            accepted_count += 1
            print(f"Auto-accepted proposed label: '{proposed}'\n")
            continue

        prompt_str = (
            "[bold white]Choose action:[/bold white]\n"
            "  [0] GENERAL_CHAT  [1] DB_READ  [2] DB_WRITE\n"
            f"  [Enter] Confirm proposed ({proposed})  [s] Skip  [d] Discard  [q] Quit & Save\n"
            "Selection: "
        )

        while True:
            if HAS_RICH:
                choice = Prompt.ask(
                    "[bold white]Select label (0: GENERAL_CHAT, 1: DB_READ, 2: DB_WRITE, s: skip)[/bold white]",
                    choices=["0", "1", "2", "s", "d", "q", ""],
                    default=""
                ).strip().lower()
            else:
                choice = input(prompt_str).strip().lower()

            if choice in valid_choices:
                break
            print("Invalid input. Please enter 0, 1, 2, s, d, q, or press Enter.")

        if choice == "q":
            remaining.extend(queue[idx:])
            break
        elif choice == "s":
            remaining.append(item)
            print("Item skipped for now.\n")
            continue
        elif choice == "d":
            discarded_count += 1
            print("Item discarded from dataset.\n")
            continue
        elif choice == "" or choice == "enter":
            append_verified_to_train(item["prompt"], proposed)
            accepted_count += 1
            print(f"Confirmed proposed label '{proposed}' -> Saved to train.jsonl\n")
        elif choice in LABEL_OPTIONS:
            selected_label = LABEL_OPTIONS[choice]
            append_verified_to_train(item["prompt"], selected_label)
            if selected_label == proposed:
                accepted_count += 1
            else:
                overridden_count += 1
            print(f"Overridden as '{selected_label}' -> Saved to train.jsonl\n")

    # Update review queue with remaining
    save_review_queue(remaining)

    summary_text = (
        f"Review Session Complete:\n"
        f"  - Confirmed:  {accepted_count}\n"
        f"  - Overridden: {overridden_count}\n"
        f"  - Discarded:  {discarded_count}\n"
        f"  - Remaining:  {len(remaining)}"
    )
    if HAS_RICH:
        console.print(Panel(summary_text, title="Summary", expand=False))
    else:
        print("\n" + "=" * 45)
        print(summary_text)
        print("=" * 45)


if __name__ == "__main__":
    main()
