"""
Trains a few-shot SetFit intent classification model.
Uses sentence-transformers backbone (default: BAAI/bge-small-en-v1.5).
Supports single-turn and multi-turn (chat_history) samples cleanly.
Saves model artifacts and label mappings to models/intent_router/.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

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

from app.intent_spec import (
    format_text_with_history,
    load_jsonl_dataset,
    load_seed_dataset,
)

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models" / "intent_router"
SEED_FILE = DATA_DIR / "seed_prompts.json"


def _to_xy(rows) -> Tuple[List[str], List[str]]:
    texts = [format_text_with_history(r["text"], r.get("chat_history")) for r in rows]
    labels = [r["label"] for r in rows]
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Train SetFit few-shot intent classifier.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Sentence Transformer backbone (default: BAAI/bge-small-en-v1.5)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-iterations", type=int, default=10, help="Contrastive text pairs per sample")
    parser.add_argument("--epochs", type=int, default=1, help="Number of body contrastive epochs")
    parser.add_argument("--push-to-hub", type=str, default=None, metavar="REPO_ID", help="Push model to Hugging Face Hub")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError:
        raise ImportError("setfit and datasets libraries are required. Install with: pip install setfit datasets")

    train_file = PROCESSED_DIR / "train.jsonl"
    val_file = PROCESSED_DIR / "val.jsonl"

    if train_file.exists():
        print(f"[INFO] Loading training dataset from {train_file}...")
        X_train, y_train = _to_xy(load_jsonl_dataset(train_file))
        X_val, y_val = _to_xy(load_jsonl_dataset(val_file)) if val_file.exists() else ([], [])
    else:
        print(f"[WARN] No processed dataset found at {train_file}. Falling back to seed_prompts.json...")
        X_train, y_train = _to_xy(load_seed_dataset(SEED_FILE))
        X_val, y_val = [], []

    print(f"[INFO] Loaded {len(X_train)} training samples and {len(X_val)} validation samples.")
    classes = sorted(list(set(y_train)))
    label_to_id = {label: i for i, label in enumerate(classes)}
    id_to_label = {i: label for i, label in enumerate(classes)}

    train_ds = Dataset.from_dict({
        "text": X_train,
        "label": [label_to_id[l] for l in y_train]
    })

    eval_ds = None
    if X_val:
        eval_ds = Dataset.from_dict({
            "text": X_val,
            "label": [label_to_id[l] for l in y_val]
        })

    print(f"[INFO] Initializing SetFitModel with backbone: {args.model_id}")
    model = SetFitModel.from_pretrained(args.model_id, labels=classes)

    training_args = TrainingArguments(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_iterations=args.num_iterations,
        logging_steps=25
    )

    def compute_accuracy(y_pred, y_test):
        from sklearn.metrics import accuracy_score
        return {"accuracy": float(accuracy_score(y_test, y_pred))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric=compute_accuracy
    )

    print("[INFO] Training SetFit model...")
    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    print(f"[OK] Saved SetFit model weights to: {MODEL_DIR}")

    label_map = {
        "classes": classes,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "backbone": args.model_id
    }
    label_map_path = MODEL_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[OK] Saved label mapping to: {label_map_path}")

    if eval_ds:
        print("[INFO] Evaluating on validation set...")
        try:
            metrics = trainer.evaluate()
            print(f"[INFO] Validation metrics: {metrics}")
        except Exception as e:
            print(f"[WARN] Validation evaluation encountered an error: {e}")

    if args.push_to_hub:
        repo_id = args.push_to_hub.strip()
        print(f"[INFO] Pushing trained SetFit model to Hugging Face Hub: {repo_id}...")
        if not hf_token:
            print("[WARN] No HF_TOKEN found in environment or arguments.")
        try:
            model.push_to_hub(repo_id=repo_id, token=hf_token)
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(label_map_path),
                path_in_repo="label_map.json",
                repo_id=repo_id
            )
            print(f"[OK] Successfully pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[ERROR] Failed to push to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()
