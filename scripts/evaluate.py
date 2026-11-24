"""
Model Evaluation and Benchmark script.
Outputs Precision, Recall, F1-score per class, Macro/Weighted F1,
a 3x3 confusion matrix, and latency statistics (mean, p50, p95).
"""

import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_FILE = PROCESSED_DIR / "test.jsonl"
SEED_FILE = DATA_DIR / "seed_prompts.json"
BASELINE_MODEL_FILE = BASE_DIR / "models" / "baseline_tfidf" / "baseline_model.joblib"
SETFIT_MODEL_DIR = BASE_DIR / "models" / "intent_router"

INTENT_CLASSES = ["READ", "WRITE", "GENERAL_CHAT"]


def load_test_data() -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    if TEST_FILE.exists():
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    texts.append(item["text"])
                    labels.append(item["label"])
    else:
        # Fallback to seeds if test.jsonl has not been generated
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        for intent, items in seeds.items():
            for it in items:
                texts.append(it)
                labels.append(intent)
    return texts, labels


def evaluate_baseline(texts: List[str]) -> Tuple[List[str], List[float]]:
    if not BASELINE_MODEL_FILE.exists():
        raise FileNotFoundError(f"Baseline model artifact not found at {BASELINE_MODEL_FILE}. Run train_baseline.py first.")
    pipeline = joblib.load(BASELINE_MODEL_FILE)

    predictions = []
    latencies = []

    for text in texts:
        t0 = time.perf_counter()
        pred = pipeline.predict([text])[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        predictions.append(pred)
        latencies.append(elapsed_ms)

    return predictions, latencies


def evaluate_setfit(texts: List[str]) -> Tuple[List[str], List[float]]:
    if not SETFIT_MODEL_DIR.exists():
        raise FileNotFoundError(f"SetFit model directory not found at {SETFIT_MODEL_DIR}. Run train_setfit.py first.")
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(str(SETFIT_MODEL_DIR))

    # Load label map if present
    label_map_file = SETFIT_MODEL_DIR / "label_map.json"
    id_to_label = None
    if label_map_file.exists():
        with open(label_map_file, "r", encoding="utf-8") as f:
            id_to_label = json.load(f).get("id_to_label")
            if id_to_label:
                id_to_label = {int(k): v for k, v in id_to_label.items()}

    predictions = []
    latencies = []

    for text in texts:
        t0 = time.perf_counter()
        raw_pred = model.predict([text])[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(raw_pred, (int, np.integer)) and id_to_label:
            pred = id_to_label[int(raw_pred)]
        else:
            pred = str(raw_pred)

        predictions.append(pred)
        latencies.append(elapsed_ms)

    return predictions, latencies


def print_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str]):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    col_width = 14

    header = f"{'True \\ Pred':<{col_width}}" + "".join(f"{c:>{col_width}}" for c in classes)
    separator = "-" * len(header)
    print("\n" + separator)
    print("3x3 CONFUSION MATRIX")
    print(separator)
    print(header)
    print(separator)
    for i, true_class in enumerate(classes):
        row_str = f"{true_class:<{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(classes)))
        print(row_str)
    print(separator + "\n")


def print_latency_stats(latencies: List[float]):
    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print("=" * 45)
    print("INFERENCE LATENCY METRICS (ms)")
    print("=" * 45)
    print(f"Sample Count : {len(latencies)}")
    print(f"Mean Latency : {mean_lat:.2f} ms")
    print(f"P50 (Median) : {p50:.2f} ms")
    print(f"P95 Latency  : {p95:.2f} ms")
    print(f"Min / Max    : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print("=" * 45 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Intent Router models.")
    parser.add_argument("--backend", choices=["setfit", "baseline"], default="baseline", help="Which model to evaluate")
    args = parser.parse_args()

    texts, y_true = load_test_data()
    print(f"Evaluating {args.backend.upper()} on {len(texts)} test samples...")

    if args.backend == "baseline":
        y_pred, latencies = evaluate_baseline(texts)
    else:
        y_pred, latencies = evaluate_setfit(texts)

    # Classification Metrics
    classes = INTENT_CLASSES
    p_micro, r_micro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print(f"EVALUATION REPORT: {args.backend.upper()}")
    print("=" * 55)
    print(f"Macro Precision : {p_micro:.4f}")
    print(f"Macro Recall    : {r_micro:.4f}")
    print(f"Macro F1 Score  : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weight:.4f}")
    print("=" * 55)

    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))

    # Confusion matrix
    print_confusion_matrix(y_true, y_pred, classes)

    # Latency
    print_latency_stats(latencies)


if __name__ == "__main__":
    main()
