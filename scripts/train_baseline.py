"""
Trains a baseline TF-IDF + LogisticRegression intent classifier benchmark.
Supports single-turn and multi-turn (chat_history) samples cleanly.
Saves model pipeline artifact and label_map.json into models/baseline_tfidf/.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.intent_spec import (
    CLASSES,
    format_text_with_history,
    load_jsonl_dataset,
    load_seed_dataset,
)

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models" / "baseline_tfidf"
SEED_FILE = DATA_DIR / "seed_prompts.json"


def _to_xy(rows) -> Tuple[List[str], List[str]]:
    texts = [format_text_with_history(r["text"], r.get("chat_history")) for r in rows]
    labels = [r["label"] for r in rows]
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline router.")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max solver iterations")
    args = parser.parse_args()

    train_file = PROCESSED_DIR / "train.jsonl"
    val_file = PROCESSED_DIR / "val.jsonl"

    if train_file.exists():
        print(f"[INFO] Loading training data from {train_file}...")
        X_train, y_train = _to_xy(load_jsonl_dataset(train_file))
        X_val, y_val = _to_xy(load_jsonl_dataset(val_file)) if val_file.exists() else ([], [])
    else:
        print(f"[WARN] No processed train.jsonl found at {train_file}. Falling back to seed_prompts.json...")
        X_train, y_train = _to_xy(load_seed_dataset(SEED_FILE))
        X_val, y_val = [], []

    print(f"[INFO] Loaded {len(X_train)} training samples across {len(set(y_train))} classes.")

    unique_labels = sorted(list(set(y_train)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
            strip_accents="unicode",
            lowercase=True
        )),
        ("clf", LogisticRegression(
            C=args.c,
            max_iter=args.max_iter,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("[INFO] Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    if X_val:
        print("\n[INFO] Evaluating on validation set...")
        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        print(f"Validation Accuracy: {acc * 100:.2f}%")
        print(f"Validation Macro F1: {macro_f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_val, preds, labels=CLASSES, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact_path = MODEL_DIR / "baseline_model.joblib"
    joblib.dump(pipeline, model_artifact_path)
    print(f"[OK] Saved model artifact to: {model_artifact_path}")

    label_map_path = MODEL_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": unique_labels,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label
        }, f, indent=2)
    print(f"[OK] Saved label map to: {label_map_path}")


if __name__ == "__main__":
    main()
