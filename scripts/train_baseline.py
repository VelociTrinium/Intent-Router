"""
Trains a baseline TF-IDF + LogisticRegression intent classifier benchmark.
Saves model pipeline artifact and label_map.json into models/baseline_tfidf/.
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models" / "baseline_tfidf"
SEED_FILE = DATA_DIR / "seed_prompts.json"


def load_dataset(file_path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(row["label"])
    return texts, labels


def load_from_seeds() -> Tuple[List[str], List[str]]:
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for intent, examples in data.items():
        for ex in examples:
            texts.append(ex)
            labels.append(intent)
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline router.")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for solver")
    args = parser.parse_args()

    train_file = PROCESSED_DIR / "train.jsonl"
    val_file = PROCESSED_DIR / "val.jsonl"

    if train_file.exists():
        print(f"Loading training data from {train_file}...")
        X_train, y_train = load_dataset(train_file)
        X_val, y_val = load_dataset(val_file) if val_file.exists() else ([], [])
    else:
        print(f"No processed train.jsonl found at {train_file}. Falling back to seed_prompts.json...")
        X_train, y_train = load_from_seeds()
        X_val, y_val = [], []

    print(f"Loaded {len(X_train)} training samples across {len(set(y_train))} classes.")

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

    print("Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    if X_val:
        print("\nEvaluating on validation set...")
        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        print(f"Validation Accuracy: {acc * 100:.2f}%")
        print(f"Validation Macro F1: {macro_f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_val, preds, zero_division=0))

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
