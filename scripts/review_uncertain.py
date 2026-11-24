"""
Interactive CLI for human review of uncertain and borderline prompts.
Reads from data/review_queue/, allows the reviewer to accept, re-label,
or discard items, and appends verified prompts into data/processed/train.jsonl.
"""

import json
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
REVIEW_QUEUE_FILE = BASE_DIR / "data" / "review_queue" / "review_needed.json"
TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.jsonl"

VALID_CLASSES = ["READ", "WRITE", "GENERAL_CHAT"]


def print_prompt(index: int, total: int, item: Dict[str, Any]):
    prompt_text = item.get("prompt", "")
    proposed = item.get("proposed_label", "UNKNOWN")
    reason = item.get("reason", "Borderline or ambiguous")

    if HAS_RICH:
        content = (
            f"[bold cyan]Prompt:[/bold cyan] {prompt_text}\n"
            f"[bold yellow]Proposed Label:[/bold yellow] [bold green]{proposed}[/bold green]\n"
            f"[dim]Flag Reason:[/dim] {reason}"
        )
        console.print(Panel(content, title=f"Review Item [{index + 1}/{total}]", expand=False))
    else:
        print("=" * 60)
        print(f"Review Item [{index + 1}/{total}]")
        print(f"Prompt:         {prompt_text}")
        print(f"Proposed Label: {proposed}")
        print(f"Flag Reason:    {reason}")
        print("=" * 60)


def append_to_train(text: str, label: str):
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        record = {"text": text, "label": label, "reviewed_by_human": True}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    if not REVIEW_QUEUE_FILE.exists():
        msg = f"No review queue found at {REVIEW_QUEUE_FILE}. Run generate_data.py first."
        if HAS_RICH:
            console.print(f"[bold green]{msg}[/bold green]")
        else:
            print(msg)
        return

    with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
        queue: List[Dict[str, Any]] = json.load(f)

    if not queue:
        msg = "The review queue is empty! Nothing to review."
        if HAS_RICH:
            console.print(f"[bold green]{msg}[/bold green]")
        else:
            print(msg)
        return

    print(f"\nFound {len(queue)} items awaiting human review.\n")
    remaining = []
    accepted_count = 0
    discarded_count = 0

    for idx, item in enumerate(queue):
        print_prompt(idx, len(queue), item)
        options = (
            "Options: [1] Accept proposed  [2] READ  [3] WRITE  [4] GENERAL_CHAT  [d] Discard  [s] Skip  [q] Quit\n"
            "Choice: "
        )

        while True:
            if HAS_RICH:
                choice = Prompt.ask("[bold white]Choose an action[/bold white]", choices=["1", "2", "3", "4", "d", "s", "q"], default="1")
            else:
                choice = input(options).strip().lower()

            if choice in ["1", "2", "3", "4", "d", "s", "q"]:
                break
            print("Invalid selection. Please choose from 1, 2, 3, 4, d, s, or q.")

        if choice == "q":
            remaining.extend(queue[idx:])
            break
        elif choice == "s":
            remaining.append(item)
            continue
        elif choice == "d":
            discarded_count += 1
            print("Item discarded.\n")
        elif choice == "1":
            label = item.get("proposed_label", "GENERAL_CHAT")
            append_to_train(item["prompt"], label)
            accepted_count += 1
            print(f"Accepted as '{label}' -> added to train.jsonl\n")
        elif choice == "2":
            append_to_train(item["prompt"], "READ")
            accepted_count += 1
            print("Labeled as 'READ' -> added to train.jsonl\n")
        elif choice == "3":
            append_to_train(item["prompt"], "WRITE")
            accepted_count += 1
            print("Labeled as 'WRITE' -> added to train.jsonl\n")
        elif choice == "4":
            append_to_train(item["prompt"], "GENERAL_CHAT")
            accepted_count += 1
            print("Labeled as 'GENERAL_CHAT' -> added to train.jsonl\n")

    # Update review queue with remaining unreviewed items
    with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

    summary = (
        f"Review session completed.\n"
        f"- Accepted: {accepted_count}\n"
        f"- Discarded: {discarded_count}\n"
        f"- Remaining in queue: {len(remaining)}"
    )
    if HAS_RICH:
        console.print(Panel(summary, title="Session Summary", expand=False))
    else:
        print("\n" + "=" * 40)
        print(summary)
        print("=" * 40)


if __name__ == "__main__":
    main()
