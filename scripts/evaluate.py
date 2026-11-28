"""
Evaluation Suite for the fine-tuned 4-class Ollama Qwen2.5-0.5B Intent Router.
Evaluates single-turn and multi-turn test samples from `data/processed/test.jsonl`
(falling back to `data/seed_prompts.json` if absent) and outputs:
- Raw model accuracy/F1 vs Guarded RouterEngine accuracy/F1
- Per-class Precision, Recall, and F1-score (highlighting DB_WRITE and CLARIFICATION_REQUIRED Recall)
- Multi-turn subset accuracy
- Full 4x4 Confusion Matrix
- Inference Latency Statistics (mean, p50 median, p95) on local hardware
- Full list of misclassified test examples
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.intent_spec import CLASSES, load_jsonl_dataset, load_seed_dataset
from app.router_engine import RouterEngine

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_FILE = PROCESSED_DIR / "test.jsonl"
SEED_FILE = DATA_DIR / "seed_prompts.json"


def load_test_records() -> List[Dict[str, Any]]:
    records = load_jsonl_dataset(TEST_FILE) if TEST_FILE.exists() else load_seed_dataset(SEED_FILE)
    if not records:
        raise RuntimeError(f"No evaluation records found in {TEST_FILE} or {SEED_FILE}.")
    return records


def print_confusion_matrix(y_true: List[str], y_pred: List[str], classes: List[str], title: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    col_width = max(15, max((len(c) for c in classes), default=12) + 2)
    header = f"{'True \\ Pred':<{col_width}}" + "".join(f"{c:>{col_width}}" for c in classes)
    separator = "=" * len(header)
    print("\n" + separator)
    print(title)
    print(separator)
    print(header)
    print("-" * len(header))
    for i, true_class in enumerate(classes):
        row_str = f"{true_class:<{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(classes)))
        print(row_str)
    print(separator + "\n")


def print_latency_stats(latencies: List[float], label: str):
    arr = np.array(latencies)
    print("=" * 55)
    print(f"LOCAL INFERENCE LATENCY BENCHMARK ({label})")
    print("=" * 55)
    print(f"Sample Count       : {len(arr)}")
    print(f"Mean Latency       : {np.mean(arr):.2f} ms")
    print(f"P50 Median Latency : {np.percentile(arr, 50):.2f} ms")
    print(f"P95 Latency        : {np.percentile(arr, 95):.2f} ms")
    print(f"Min / Max Latency  : {np.min(arr):.2f} ms / {np.max(arr):.2f} ms")
    print("=" * 55 + "\n")


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)

    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": int(sup_cls[i]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "db_write_recall": per_class["DB_WRITE"]["recall"],
        "clarification_recall": per_class["CLARIFICATION_REQUIRED"]["recall"],
        "per_class": per_class,
    }


def evaluate_ollama_model(
    records: List[Dict[str, Any]],
    ollama_model: Optional[str] = None,
    label_override: Optional[str] = None,
) -> Dict[str, Any]:
    target_model = ollama_model or settings.OLLAMA_MODEL
    display_name = label_override or target_model
    print("\n" + "#" * 65)
    print(f"EVALUATING OLLAMA MODEL: {display_name}")
    print("#" * 65)

    engine = RouterEngine(ollama_model_override=target_model, cache_size=0)
    if not engine.router.is_ready():
        raise RuntimeError(
            f"Ollama model '{target_model}' is not ready. "
            "Verify `ollama serve` is running and the model has been created."
        )

    # Warm-up 1 inference (excluded from warm latency stats)
    t_warm = time.perf_counter()
    _ = engine.route(records[0]["text"], chat_history=records[0].get("chat_history"))
    cold_start_ms = round((time.perf_counter() - t_warm) * 1000.0, 2)

    y_true: List[str] = []
    y_raw: List[str] = []
    y_guarded: List[str] = []
    latencies: List[float] = []
    errors: List[Dict[str, Any]] = []
    mt_indices: List[int] = []
    nll_sum = 0.0
    brier_sum = 0.0

    for idx, rec in enumerate(records):
        hist = rec.get("chat_history") or []
        if hist:
            mt_indices.append(idx)
        res = engine.route(rec["text"], chat_history=hist)
        y_true.append(rec["label"])
        y_raw.append(res.raw_prediction)
        y_guarded.append(res.route)
        latencies.append(res.latency_ms or 0.0)

        p_true = float(res.probabilities.get(rec["label"], 0.0))
        nll_sum += -math.log(max(p_true, 1e-12))
        brier_sum += sum(
            (float(res.probabilities.get(c, 0.0)) - (1.0 if c == rec["label"] else 0.0)) ** 2
            for c in CLASSES
        )

        if res.route != rec["label"]:
            errors.append({
                "index": idx,
                "prompt": rec["text"],
                "has_history": bool(hist),
                "ground_truth": rec["label"],
                "raw_prediction": res.raw_prediction,
                "predicted": res.route,
                "confidence": res.confidence,
                "probabilities": res.probabilities,
            })

    raw_metrics = _compute_metrics(y_true, y_raw)
    guarded_metrics = _compute_metrics(y_true, y_guarded)

    mt_acc = None
    if mt_indices:
        mt_acc = round(
            float(accuracy_score([y_true[i] for i in mt_indices], [y_guarded[i] for i in mt_indices])),
            4,
        )

    n_samples = max(1, len(records))
    mean_nll = round(nll_sum / n_samples, 4)
    mean_brier = round(brier_sum / n_samples, 4)

    print("\n" + "=" * 65)
    print(f"PERFORMANCE SUMMARY: {display_name}")
    print("=" * 65)
    print(f"Guarded Accuracy      : {guarded_metrics['accuracy'] * 100:.2f}%  (Raw: {raw_metrics['accuracy'] * 100:.2f}%)")
    print(f"Guarded Macro F1      : {guarded_metrics['macro_f1']:.4f}   (Raw: {raw_metrics['macro_f1']:.4f})")
    print(f"DB_WRITE Recall       : {guarded_metrics['db_write_recall'] * 100:.2f}%  (Raw: {raw_metrics['db_write_recall'] * 100:.2f}%)")
    print(f"CLARIFICATION Recall  : {guarded_metrics['clarification_recall'] * 100:.2f}%  (Raw: {raw_metrics['clarification_recall'] * 100:.2f}%)")
    print(f"Calibration NLL/Brier : NLL={mean_nll:.4f}  Brier={mean_brier:.4f}")
    if mt_acc is not None:
        print(f"Multi-Turn Accuracy   : {mt_acc * 100:.2f}% ({len(mt_indices)} samples)")
    print(f"Cold-Start Latency    : {cold_start_ms:.2f} ms")
    print("=" * 65)

    print("\nGuarded Per-Class Classification Report:")
    print(classification_report(y_true, y_guarded, labels=CLASSES, zero_division=0))

    print_confusion_matrix(y_true, y_guarded, CLASSES, f"4x4 CONFUSION MATRIX ({display_name} - GUARDED)")
    print_latency_stats(latencies, display_name)

    print("=" * 70)
    print(f"MISCLASSIFIED EXAMPLES ({len(errors)} errors / {len(records)} total)")
    print("=" * 70)
    for err in errors[:20]:
        prob_summary = ", ".join(f"{k}: {v:.2f}" for k, v in err["probabilities"].items())
        hist_tag = " [multi-turn]" if err["has_history"] else ""
        print(f"[{err['index'] + 1}]{hist_tag} Prompt: \"{err['prompt']}\"")
        print(f"     Truth: {err['ground_truth']} | Guarded: {err['predicted']} (raw: {err['raw_prediction']})")
        print(f"     Probs: [{prob_summary}]\n")
    if len(errors) > 20:
        print(f"... ({len(errors) - 20} additional errors omitted from console; see JSON report)")

    return {
        "model": target_model,
        "sample_count": len(records),
        "multi_turn_count": len(mt_indices),
        "multi_turn_accuracy": mt_acc,
        "accuracy": guarded_metrics["accuracy"],
        "macro_f1": guarded_metrics["macro_f1"],
        "weighted_f1": guarded_metrics["weighted_f1"],
        "db_write_recall": guarded_metrics["db_write_recall"],
        "clarification_recall": guarded_metrics["clarification_recall"],
        "calibration_nll": mean_nll,
        "calibration_brier": mean_brier,
        "per_class": guarded_metrics["per_class"],
        "raw": raw_metrics,
        "cold_start_ms": cold_start_ms,
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "num_errors": len(errors),
        "errors": errors,
    }


def check_regression(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
    tolerance: float = 0.005,
) -> Tuple[bool, List[str]]:
    """Checks candidate metrics against baseline within tolerance. Returns (passed, failure_reasons)."""
    checks = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("db_write_recall", "DB_WRITE Recall"),
        ("clarification_recall", "CLARIFICATION Recall"),
    ]
    failures: List[str] = []
    for key, label in checks:
        cand_v = float(candidate.get(key, 0.0))
        base_v = float(baseline.get(key, 0.0))
        if cand_v < base_v - tolerance:
            failures.append(f"{label} regressed: {cand_v:.4f} < {base_v:.4f} - {tolerance:.4f}")
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned 4-class Ollama Qwen2.5-0.5B Intent Router on test.jsonl.")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model name override (default from .env/config)")
    parser.add_argument("--compare-untuned-ollama", type=str, default=None, help="Optional extra Ollama model to compare alongside (e.g. qwen2.5:0.5b)")
    parser.add_argument("--baseline-report", type=Path, default=None, help="Optional saved evaluation_report.json to compare against without re-running Ollama")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero if candidate regresses vs comparison/baseline model")
    parser.add_argument("--regression-tolerance", type=float, default=0.005, help="Allowed metric delta before triggering regression failure")
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(BASE_DIR / "models" / "evaluation_report.json"),
        help="Path to save structured evaluation results JSON",
    )
    args = parser.parse_args()

    records = load_test_records()
    print(f"[INFO] Loaded {len(records)} test samples from {TEST_FILE if TEST_FILE.exists() else SEED_FILE}.")

    results: Dict[str, Any] = {
        "ollama": evaluate_ollama_model(records=records, ollama_model=args.ollama_model)
    }

    baseline_metrics: Optional[Dict[str, Any]] = None
    if args.compare_untuned_ollama:
        try:
            key = f"ollama_compare_{args.compare_untuned_ollama}"
            results[key] = evaluate_ollama_model(
                records=records,
                ollama_model=args.compare_untuned_ollama,
                label_override=key,
            )
            baseline_metrics = results[key]
        except Exception as e:
            print(f"[WARN] Skipping comparison model '{args.compare_untuned_ollama}': {e}")

    if baseline_metrics is None and args.baseline_report and args.baseline_report.exists():
        saved = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline_metrics = saved.get("ollama")
        if baseline_metrics:
            results[f"baseline_{baseline_metrics.get('model', args.baseline_report.name)}"] = baseline_metrics

    if len(results) >= 2:
        print("\n" + "=" * 104)
        print(
            f"COMPARATIVE SUMMARY (CLARIFY={settings.CLARIFY_THRESHOLD:.2f}, "
            f"WRITE={settings.WRITE_THRESHOLD:.2f}, CONF={settings.CONFIDENCE_THRESHOLD:.2f}, "
            f"FALLBACK={settings.FALLBACK_ROUTE})"
        )
        print("=" * 104)
        print(f"{'Model':<34} {'Acc':<9} {'MacroF1':<9} {'WRITE_Rec':<11} {'ASK_Rec':<9} {'MT_Acc':<9} {'Mean_ms':<10} {'P95_ms':<10} {'Errs'}")
        print("-" * 104)
        for name, m in results.items():
            mt_str = f"{m['multi_turn_accuracy']*100:.1f}%" if m.get("multi_turn_accuracy") is not None else "n/a"
            print(
                f"{name:<34} "
                f"{m['accuracy']*100:>5.1f}%   "
                f"{m['macro_f1']:<9.4f} "
                f"{m['db_write_recall']*100:>6.1f}%    "
                f"{m['clarification_recall']*100:>5.1f}%   "
                f"{mt_str:<9} "
                f"{m['latency_mean_ms']:<10.2f} "
                f"{m['latency_p95_ms']:<10.2f} "
                f"{m['num_errors']}"
            )
        print("=" * 104 + "\n")

    if args.output_report and results:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Saved evaluation report to: {report_path}")

    if args.fail_on_regression:
        if baseline_metrics is None:
            raise SystemExit("[FAIL] --fail-on-regression requires --compare-untuned-ollama or a valid --baseline-report.")
        passed, reasons = check_regression(
            results["ollama"],
            baseline_metrics,
            tolerance=args.regression_tolerance,
        )
        if not passed:
            for r in reasons:
                print(f"[REGRESSION] {r}")
            raise SystemExit(1)
        print("[OK] Regression gate passed against baseline!")


if __name__ == "__main__":
    main()
