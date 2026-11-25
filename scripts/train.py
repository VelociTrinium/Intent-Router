"""
Unified Training Entrypoint.
Supports:
  python scripts/train.py --backend ollama    (Fine-tunes Qwen2.5-0.5B-Instruct LoRA + merges HF weights)
  python scripts/train.py --backend setfit    (Fine-tunes SetFit sentence transformer classifier)
  python scripts/train.py --backend baseline  (Fits TF-IDF + LogisticRegression benchmark)
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Unified Trainer for Intent Router.")
    parser.add_argument(
        "--backend",
        choices=["ollama", "setfit", "baseline"],
        default="ollama",
        help="Which model architecture to train (default: ollama)"
    )
    args, unknown = parser.parse_known_args()

    if args.backend == "ollama":
        from scripts.train_ollama import main as ollama_main
        sys.argv = [sys.argv[0]] + unknown
        ollama_main()
    elif args.backend == "setfit":
        from scripts.train_setfit import main as setfit_main
        sys.argv = [sys.argv[0]] + unknown
        setfit_main()
    else:
        from scripts.train_baseline import main as baseline_main
        sys.argv = [sys.argv[0]] + unknown
        baseline_main()


if __name__ == "__main__":
    main()
