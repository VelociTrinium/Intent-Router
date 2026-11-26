"""
End-to-End Retrain & Ollama Release Pipeline.

Takes your custom working examples (`data/custom/*.json` / `.jsonl`), optionally
merges or regenerates synthetic data, fine-tunes a FRESH base model from scratch,
exports to GGUF (`Q8_0` by default), registers the model in Ollama preserving the
1-token `CHAT`/`READ`/`WRITE` output contract, and verifies evaluation thresholds.

Example:
    python scripts/retrain_pipeline.py --version v1
    python scripts/retrain_pipeline.py --model-name intent-router-qwen2.5-0.5b-instruct --version v2
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.export_ollama_gguf import resolve_ollama_ref


def _run_step(title: str, cmd: List[str], dry_run: bool = False):
    print("\n" + "=" * 72)
    print(f"[PIPELINE] {title}")
    print("  CMD:", " ".join(cmd))
    print("=" * 72)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)


def main():
    default_local_base = BASE_DIR / "models" / "hf_base" / "Qwen2.5-0.5B-Instruct"
    default_base_model = str(default_local_base) if default_local_base.exists() else "Qwen/Qwen2.5-0.5B-Instruct"

    parser = argparse.ArgumentParser(description="End-to-end custom data -> fresh base model -> Ollama pipeline.")
    # Dataset options
    parser.add_argument("--custom-file", type=Path, action="append", default=None, help="Explicit custom JSON/JSONL file(s) (defaults to all data/custom/*.json*)")
    parser.add_argument("--custom-only", action="store_true", help="Train strictly on data/custom/ without synthetic data")
    parser.add_argument("--include-seeds", action="store_true", help="Include data/seed_prompts.json when data/raw/synthetic_prompts.json is absent")
    parser.add_argument("--regenerate-synthetic", action="store_true", help="Re-run Gemini synthetic data generation before merging")
    parser.add_argument("--target-total", type=int, default=810, help="Target synthetic dataset size when --regenerate-synthetic is set")
    parser.add_argument("--custom-train-repeat", type=int, default=1, help="Upsample factor for custom rows inside train.jsonl")

    # Fresh base model training options
    parser.add_argument("--base-model", type=str, default=default_base_model, help="Fresh HuggingFace model ID or local base directory")
    parser.add_argument("--epochs", type=float, default=2.0, help="LoRA fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=16)

    # Export & Ollama registration options
    parser.add_argument("--model-name", type=str, default="intent-router-qwen2.5-0.5b-instruct", help="Ollama model repository name")
    parser.add_argument("--version", type=str, default="v1", help="Ollama model version tag (e.g. v1, 1.0.0; empty to omit)")
    parser.add_argument("--outtype", choices=["q8_0", "f16", "bf16", "f32"], default="q8_0", help="GGUF tensor precision")
    parser.add_argument("--ollama-quantize", type=str, default=None, help="Optional `ollama create -q` target (e.g. q4_K_M)")

    # Evaluation gate options
    parser.add_argument("--skip-eval", action="store_true", help="Skip held-out test.jsonl evaluation after creating Ollama model")
    parser.add_argument("--min-write-recall", type=float, default=0.95, help="Minimum guarded DB_WRITE recall gate (0..1)")
    parser.add_argument("--min-macro-f1", type=float, default=0.90, help="Minimum guarded Macro F1 gate (0..1)")
    parser.add_argument("--dry-run", action="store_true", help="Validate custom data and print planned commands without training/writing")
    args = parser.parse_args()

    py = sys.executable

    # 1. Build & validate dataset
    build_cmd = [py, str(SCRIPTS_DIR / "build_dataset.py"), "--custom-train-repeat", str(args.custom_train_repeat)]
    if args.custom_file:
        for cf in args.custom_file:
            build_cmd.extend(["--custom-file", str(cf)])
    if args.custom_only:
        build_cmd.append("--custom-only")
    if args.include_seeds:
        build_cmd.append("--include-seeds")
    if args.regenerate_synthetic:
        build_cmd.extend(["--regenerate-synthetic", "--target-total", str(args.target_total)])
    if args.dry_run:
        build_cmd.append("--dry-run")

    _run_step("Step 1/4: Validate & build train/val/test splits", build_cmd, dry_run=False)

    # 2. Fresh base model LoRA fine-tuning
    train_cmd = [
        py, "-u", str(SCRIPTS_DIR / "train_ollama.py"),
        "--model-id", args.base_model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--lora-r", str(args.lora_r),
    ]
    _run_step(f"Step 2/4: Fine-tune fresh base model ({args.base_model})", train_cmd, dry_run=args.dry_run)

    # 3. Export GGUF & register in Ollama
    ollama_ref, _ = resolve_ollama_ref(args.model_name, args.version)
    export_cmd = [
        py, str(SCRIPTS_DIR / "export_ollama_gguf.py"),
        "--outtype", args.outtype,
        "--model-name", args.model_name,
        "--version", args.version,
    ]
    if args.ollama_quantize:
        export_cmd.extend(["--ollama-quantize", args.ollama_quantize])
    _run_step(f"Step 3/4: Export {args.outtype.upper()} GGUF & create Ollama '{ollama_ref}'", export_cmd, dry_run=args.dry_run)

    # 4. Evaluate in Ollama & enforce quality gates
    if not args.skip_eval:
        report_path = BASE_DIR / "models" / "evaluation_report.json"
        eval_cmd = [
            py, "-u", str(SCRIPTS_DIR / "evaluate.py"),
            "--ollama-model", ollama_ref,
            "--output-report", str(report_path),
        ]
        _run_step(f"Step 4/4: Evaluate '{ollama_ref}' on test.jsonl", eval_cmd, dry_run=args.dry_run)

        if not args.dry_run and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            m = report.get("ollama", {})
            w_rec = float(m.get("db_write_recall", 0.0))
            m_f1 = float(m.get("macro_f1", 0.0))
            print(f"\n[GATE CHECK] DB_WRITE Recall={w_rec:.4f} (min {args.min_write_recall:.2f}) | Macro F1={m_f1:.4f} (min {args.min_macro_f1:.2f})")
            if w_rec < args.min_write_recall or m_f1 < args.min_macro_f1:
                raise SystemExit(
                    f"[FAIL] Evaluation gate not met: DB_WRITE recall={w_rec:.4f}, Macro F1={m_f1:.4f}"
                )
            print("[OK] Evaluation quality gates passed!")

    print("\n[DONE] Retrain pipeline complete.")


if __name__ == "__main__":
    main()
