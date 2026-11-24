"""
Trains a few-shot SetFit intent classification model.
Supports fine-tuning sentence transformer representations with contrastive training,
saving local model weights and config, and optional pushing to the Hugging Face Hub.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models" / "intent_router"
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
    parser = argparse.ArgumentParser(description="Train SetFit few-shot intent classifier.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Pretrained Sentence Transformer backbone from Hugging Face"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-iterations", type=int, default=20, help="Number of text pairs generated per sentence")
    parser.add_argument("--epochs", type=int, default=1, help="Number of body epochs")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model weights to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo ID (e.g. username/intent-router)")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError:
        raise ImportError(
            "setfit and datasets libraries are required. Install with: pip install setfit datasets"
        )

    train_file = PROCESSED_DIR / "train.jsonl"
    val_file = PROCESSED_DIR / "val.jsonl"

    if train_file.exists():
        print(f"Loading train dataset from {train_file}...")
        X_train, y_train = load_dataset(train_file)
        X_val, y_val = load_dataset(val_file) if val_file.exists() else ([], [])
    else:
        print(f"Processed data not found at {train_file}. Falling back to seeds...")
        X_train, y_train = load_from_seeds()
        X_val, y_val = [], []

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

    print(f"Initializing SetFitModel with backbone: {args.model_id}")
    model = SetFitModel.from_pretrained(
        args.model_id,
        labels=classes
    )

    training_args = TrainingArguments(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_iterations=args.num_iterations,
        evaluation_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch" if eval_ds else "no",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric="accuracy"
    )

    print("Beginning SetFit few-shot contrastive fine-tuning...")
    trainer.train()

    if eval_ds:
        print("Evaluating model performance on validation set...")
        metrics = trainer.evaluate()
        print(f"Validation Metrics: {metrics}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving fine-tuned SetFit model to: {MODEL_DIR}")
    model.save_pretrained(str(MODEL_DIR))

    label_map_path = MODEL_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "backbone": args.model_id
        }, f, indent=2)
    print(f"[OK] Saved label map to: {label_map_path}")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id must be provided when --push-to-hub is enabled.")
        print(f"Pushing model weights to Hugging Face Hub repository: {args.repo_id}...")
        model.push_to_hub(repo_id=args.repo_id, token=hf_token)
        print("[OK] Successfully pushed model to Hugging Face Hub.")


if __name__ == "__main__":
    main()
