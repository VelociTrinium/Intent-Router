"""
Dataset Builder & Validator for Intent Router.

Combines:
1. User-curated working data from `data/custom/*.json` and `data/custom/*.jsonl`
   (supporting list-of-objects, class-grouped dicts, and JSONL).
2. Existing synthetic + seed pool (`data/raw/synthetic_prompts.json` or `data/seed_prompts.json`),
   unless `--custom-only` is specified.
3. Optional fresh Gemini generation (`--regenerate-synthetic`).

Custom examples take strict priority during deduplication, and optional `"split"`
pins ("train" | "val" | "test") are honored without leakage across splits.
"""

import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.intent_spec import CLASSES, TOKEN_TO_ROUTE, normalize_chat_history

DATA_DIR = BASE_DIR / "data"
CUSTOM_DIR = DATA_DIR / "custom"
RAW_FILE = DATA_DIR / "raw" / "synthetic_prompts.json"
SEED_FILE = DATA_DIR / "seed_prompts.json"
PROCESSED_DIR = DATA_DIR / "processed"

VALID_SPLITS = {"train", "val", "test"}


def _canonical_label(raw_label: Any, source_desc: str) -> str:
    val = str(raw_label or "").strip().upper()
    if val in CLASSES:
        return val
    if val in TOKEN_TO_ROUTE:
        return TOKEN_TO_ROUTE[val]
    raise ValueError(f"Invalid label '{raw_label}' in {source_desc}. Expected one of {CLASSES} or {list(TOKEN_TO_ROUTE.keys())}.")


def _normalize_record(item: Any, source_desc: str, default_label: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(item, str):
        if default_label is None:
            raise ValueError(f"Bare string entry without label in {source_desc}: {item!r}")
        text = item.strip()
        if not text:
            raise ValueError(f"Empty prompt string in {source_desc}")
        return {"text": text, "label": default_label}

    if not isinstance(item, dict):
        raise ValueError(f"Unsupported item type {type(item).__name__} in {source_desc}")

    raw_text = item.get("text") if item.get("text") is not None else item.get("prompt")
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError(f"Missing/empty 'text' or 'prompt' in {source_desc}: {item}")

    raw_label = item.get("label") or item.get("intent") or item.get("route") or default_label
    label = _canonical_label(raw_label, source_desc)

    rec: Dict[str, Any] = {"text": text, "label": label}

    raw_history = item.get("chat_history")
    if raw_history is not None:
        if not isinstance(raw_history, list):
            raise ValueError(f"'chat_history' must be a list in {source_desc}")
        hist = normalize_chat_history(raw_history, max_turns=6)
        if hist:
            rec["chat_history"] = hist

    raw_split = item.get("split")
    if raw_split is not None:
        split_val = str(raw_split).strip().lower()
        if split_val not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{raw_split}' in {source_desc}. Expected one of {sorted(VALID_SPLITS)}.")
        rec["split"] = split_val

    return rec


def load_custom_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parses a custom .json or .jsonl file in list, dict-by-class, or JSONL format."""
    records: List[Dict[str, Any]] = []
    if file_path.suffix.lower() == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(_normalize_record(obj, f"{file_path.name}:line {idx}"))
        return records

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for idx, item in enumerate(data):
            records.append(_normalize_record(item, f"{file_path.name}[{idx}]"))
    elif isinstance(data, dict):
        for raw_cls, items in data.items():
            cls_label = _canonical_label(raw_cls, f"{file_path.name} key '{raw_cls}'")
            if not isinstance(items, list):
                raise ValueError(f"Expected list under key '{raw_cls}' in {file_path.name}")
            for idx, item in enumerate(items):
                records.append(_normalize_record(item, f"{file_path.name}.{raw_cls}[{idx}]", default_label=cls_label))
    else:
        raise ValueError(f"Top-level JSON in {file_path.name} must be a list or object.")

    return records


def load_custom_directory(custom_dir: Path, explicit_files: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    files: List[Path] = []
    if explicit_files:
        files = list(explicit_files)
    elif custom_dir.exists():
        for p in sorted(custom_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in (".json", ".jsonl") and not p.name.startswith("example"):
                files.append(p)

    all_custom: List[Dict[str, Any]] = []
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Custom data file not found: {fp}")
        loaded = load_custom_file(fp)
        print(f"[INFO] Loaded {len(loaded)} custom records from {fp}")
        for rec in loaded:
            rec["_is_custom"] = True
        all_custom.extend(loaded)
    return all_custom


def load_base_pool(include_seeds: bool = False) -> List[Dict[str, Any]]:
    """Loads existing raw synthetic pool (and seed_prompts.json only if include_seeds=True)."""
    out: List[Dict[str, Any]] = []
    if RAW_FILE.exists():
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        for idx, item in enumerate(raw_items):
            rec = _normalize_record(item, f"{RAW_FILE.name}[{idx}]")
            rec["_is_custom"] = False
            out.append(rec)
    elif include_seeds and SEED_FILE.exists():
        loaded = load_custom_file(SEED_FILE)
        for rec in loaded:
            rec["_is_custom"] = False
            out.append(rec)

    return out


def _dedup_key(rec: Dict[str, Any]) -> str:
    hist = rec.get("chat_history") or []
    hist_norm = [(t["role"].lower(), t["content"].lower()) for t in hist]
    return json.dumps({"h": hist_norm, "p": rec["text"].strip().lower()}, ensure_ascii=False)


def _strip_internal(rec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"text": rec["text"], "label": rec["label"]}
    if rec.get("chat_history"):
        out["chat_history"] = rec["chat_history"]
    return out


def build_splits(
    custom_records: List[Dict[str, Any]],
    base_records: List[Dict[str, Any]],
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    custom_train_repeat: int = 1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Merges custom + base records with custom-wins deduplication, honors pinned splits,
    and performs stratified splitting by (label, is_multi_turn).
    """
    merged_by_key: Dict[str, Dict[str, Any]] = {}

    # Load base pool first, then let custom overwrite any collisions
    for rec in base_records:
        k = _dedup_key(rec)
        if k not in merged_by_key:
            merged_by_key[k] = rec

    overridden = 0
    for rec in custom_records:
        k = _dedup_key(rec)
        if k in merged_by_key and not merged_by_key[k].get("_is_custom"):
            overridden += 1
        merged_by_key[k] = rec

    train_split: List[Dict[str, Any]] = []
    val_split: List[Dict[str, Any]] = []
    test_split: List[Dict[str, Any]] = []
    unpinned: List[Dict[str, Any]] = []

    for rec in merged_by_key.values():
        pinned = rec.get("split")
        if pinned == "train":
            rep = custom_train_repeat if rec.get("_is_custom") else 1
            for _ in range(rep):
                train_split.append(_strip_internal(rec))
        elif pinned == "val":
            val_split.append(_strip_internal(rec))
        elif pinned == "test":
            test_split.append(_strip_internal(rec))
        else:
            unpinned.append(rec)

    random.seed(seed)
    buckets: Dict[Tuple[str, bool], List[Dict[str, Any]]] = {}
    for rec in unpinned:
        buckets.setdefault((rec["label"], bool(rec.get("chat_history"))), []).append(rec)

    for _, items in sorted(buckets.items()):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = max(1, int(n * val_ratio)) if n >= 8 else int(n * val_ratio)
        if n_train + n_val >= n and n >= 3:
            n_val = max(0, n - n_train - 1)

        for rec in items[:n_train]:
            rep = custom_train_repeat if rec.get("_is_custom") else 1
            for _ in range(rep):
                train_split.append(_strip_internal(rec))
        for rec in items[n_train:n_train + n_val]:
            val_split.append(_strip_internal(rec))
        for rec in items[n_train + n_val:]:
            test_split.append(_strip_internal(rec))

    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)

    stats = {
        "unique_total": len(merged_by_key),
        "custom_unique": sum(1 for r in merged_by_key.values() if r.get("_is_custom")),
        "custom_overrode_synthetic": overridden,
        "train_count": len(train_split),
        "val_count": len(val_split),
        "test_count": len(test_split),
    }
    return train_split, val_split, test_split, stats


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build clean train/val/test JSONL splits from custom + synthetic data.")
    parser.add_argument("--custom-file", type=Path, action="append", default=None, help="Explicit custom JSON/JSONL file(s) (defaults to all data/custom/*.json*)")
    parser.add_argument("--custom-only", action="store_true", help="Train strictly on data/custom/ without synthetic/seed pool")
    parser.add_argument("--include-seeds", action="store_true", help="Include data/seed_prompts.json when data/raw/synthetic_prompts.json is absent")
    parser.add_argument("--regenerate-synthetic", action="store_true", help="Re-run generate_data.py via Gemini before merging")
    parser.add_argument("--target-total", type=int, default=1080, help="Target synthetic size if --regenerate-synthetic is set (default: 1080 across 4 classes)")
    parser.add_argument("--custom-train-repeat", type=int, default=1, help="Optional upsample factor for custom rows inside train.jsonl only")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print split statistics without writing processed/*.jsonl")
    args = parser.parse_args()

    if args.regenerate_synthetic and not args.custom_only:
        gen_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "generate_data.py"),
            "--target-total",
            str(args.target_total),
        ]
        print(f"[INFO] Regenerating synthetic data: {' '.join(gen_cmd)}")
        subprocess.run(gen_cmd, check=True)

    custom_records = load_custom_directory(CUSTOM_DIR, explicit_files=args.custom_file)
    base_records = [] if args.custom_only else load_base_pool(include_seeds=args.include_seeds)

    if not custom_records and not base_records:
        raise RuntimeError(
            "No records found in data/custom/ or data/raw/. Add examples to data/custom/working_examples.json, "
            "pass --include-seeds to build from data/seed_prompts.json, or pass --regenerate-synthetic to run generate_data.py."
        )

    train_rows, val_rows, test_rows, stats = build_splits(
        custom_records=custom_records,
        base_records=base_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        custom_train_repeat=max(1, args.custom_train_repeat),
        seed=args.seed,
    )

    print("\n[DATASET BUILD SUMMARY]")
    for k, v in stats.items():
        print(f"  - {k}: {v}")
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        counts = {c: sum(1 for r in rows if r["label"] == c) for c in CLASSES}
        mt = sum(1 for r in rows if r.get("chat_history"))
        print(f"  - {split_name:<5}: {len(rows)} total (multi_turn={mt}) | {counts}")

    if args.dry_run:
        print("\n[DRY-RUN] Validation succeeded. No files were overwritten.")
        return

    save_jsonl(PROCESSED_DIR / "train.jsonl", train_rows)
    save_jsonl(PROCESSED_DIR / "val.jsonl", val_rows)
    save_jsonl(PROCESSED_DIR / "test.jsonl", test_rows)
    print(f"\n[OK] Wrote train/val/test JSONL splits to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
