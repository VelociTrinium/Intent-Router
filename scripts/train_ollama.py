"""
Fine-tunes a compact causal LLM (default: Qwen/Qwen2.5-0.5B-Instruct) with LoRA
on single-turn and multi-turn intent routing data so it classifies prompts
into a single token (CHAT, READ, or WRITE) with calibrated next-token probabilities.

Designed for CPU-only Windows or GPU training:
- Dynamic batch padding + length-grouped sampling
- Loss computed strictly on the assistant target token (+ EOS)
- Evaluates validation Accuracy, Macro F1, and DB_WRITE Recall via exact 3-token logits
- Merges LoRA weights into FP32 HuggingFace format at models/ollama_merged/
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.intent_spec import (
    CLASSES,
    OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
    ROUTE_TO_TOKEN,
    build_ollama_messages,
    load_jsonl_dataset,
    load_seed_dataset,
)

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEED_FILE = DATA_DIR / "seed_prompts.json"
LORA_OUT_DIR = BASE_DIR / "models" / "ollama_lora"
MERGED_OUT_DIR = BASE_DIR / "models" / "ollama_merged"


def _extract_token_ids(template_out: Any) -> List[int]:
    if isinstance(template_out, dict) or hasattr(template_out, "input_ids"):
        ids = template_out["input_ids"]
    else:
        ids = template_out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def encode_sample(
    row: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 256,
) -> Dict[str, List[int]]:
    target_token = ROUTE_TO_TOKEN[row["label"]]
    eos_str = tokenizer.eos_token or "<|im_end|>"
    target_ids = tokenizer.encode(target_token + eos_str, add_special_tokens=False)

    history = list(row.get("chat_history") or [])
    prompt_ids: List[int] = []
    while True:
        messages = build_ollama_messages(
            prompt=row["text"],
            chat_history=history,
            include_system=True,
            system_prompt=OLLAMA_CLASSIFIER_SYSTEM_PROMPT,
            max_turns=6,
        )
        raw_out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = _extract_token_ids(raw_out)
        if len(prompt_ids) + len(target_ids) <= max_seq_length or not history:
            break
        # Drop oldest history turn if conversation exceeds max_seq_length
        history = history[1:]

    if len(prompt_ids) + len(target_ids) > max_seq_length:
        keep_prompt = max_seq_length - len(target_ids)
        prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_ids": prompt_ids,
    }


class IntentTokenDataset(TorchDataset):
    def __init__(self, encoded_rows: List[Dict[str, List[int]]]):
        self.rows = encoded_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        r = self.rows[idx]
        return {
            "input_ids": r["input_ids"],
            "attention_mask": r["attention_mask"],
            "labels": r["labels"],
        }


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def resolve_class_token_ids(tokenizer: Any) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for route in CLASSES:
        tok = ROUTE_TO_TOKEN[route]
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected 1 token for '{tok}', got {ids}")
        mapping[route] = ids[0]
    return mapping


def evaluate_logits_on_split(
    model: Any,
    val_rows: List[Dict[str, Any]],
    val_encoded: List[Dict[str, List[int]]],
    class_token_ids: Dict[str, int],
    write_threshold: float = settings.WRITE_THRESHOLD,
    confidence_threshold: float = settings.CONFIDENCE_THRESHOLD,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    ordered_ids = [class_token_ids[c] for c in CLASSES]

    y_true = [r["label"] for r in val_rows]
    y_raw: List[str] = []
    y_guarded: List[str] = []

    with torch.no_grad():
        for enc in val_encoded:
            inp = torch.tensor([enc["prompt_ids"]], dtype=torch.long, device=device)
            out = model(input_ids=inp)
            next_logits = out.logits[0, -1, ordered_ids]
            probs = F.softmax(next_logits.float(), dim=-1).cpu().tolist()
            prob_map = {c: float(p) for c, p in zip(CLASSES, probs)}

            raw_pred = max(prob_map.items(), key=lambda kv: kv[1])[0]
            y_raw.append(raw_pred)

            if prob_map["DB_WRITE"] >= write_threshold:
                y_guarded.append("DB_WRITE")
            elif prob_map[raw_pred] < confidence_threshold:
                y_guarded.append("GENERAL_CHAT")
            else:
                y_guarded.append(raw_pred)

    raw_acc = accuracy_score(y_true, y_raw)
    raw_f1 = f1_score(y_true, y_raw, average="macro", zero_division=0)
    guarded_acc = accuracy_score(y_true, y_guarded)
    guarded_f1 = f1_score(y_true, y_guarded, average="macro", zero_division=0)
    write_rec = recall_score(y_true, y_guarded, labels=["DB_WRITE"], average=None, zero_division=0)[0]

    model.train()
    return {
        "raw_accuracy": round(float(raw_acc), 4),
        "raw_macro_f1": round(float(raw_f1), 4),
        "guarded_accuracy": round(float(guarded_acc), 4),
        "guarded_macro_f1": round(float(guarded_f1), 4),
        "db_write_recall": round(float(write_rec), 4),
    }


def main():
    default_model_id = str(BASE_DIR / "models" / "hf_base" / "Qwen2.5-0.5B-Instruct")
    if not Path(default_model_id).exists():
        default_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-0.5B-Instruct as 1-token Ollama intent classifier.")
    parser.add_argument("--model-id", type=str, default=default_model_id, help="Base HuggingFace causal LM or local directory")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional cap on optimizer steps (-1 = full epochs)")
    parser.add_argument("--cpu-threads", type=int, default=min(12, os.cpu_count() or 8), help="PyTorch CPU intra-op threads")
    parser.add_argument("--lora-dir", type=Path, default=LORA_OUT_DIR, help="Output dir for LoRA adapter")
    parser.add_argument("--merged-dir", type=Path, default=MERGED_OUT_DIR, help="Output dir for merged HF weights")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model

    if not torch.cuda.is_available():
        torch.set_num_threads(args.cpu_threads)
        print(f"[INFO] Running on CPU with {torch.get_num_threads()} PyTorch threads.")

    train_file = PROCESSED_DIR / "train.jsonl"
    val_file = PROCESSED_DIR / "val.jsonl"
    train_rows = load_jsonl_dataset(train_file) if train_file.exists() else load_seed_dataset(SEED_FILE)
    if not train_rows:
        raise RuntimeError(f"No training samples found in {train_file} or {SEED_FILE}. Run `python scripts/build_dataset.py` first.")
    val_rows = load_jsonl_dataset(val_file) if val_file.exists() else []
    print(f"[INFO] Loaded {len(train_rows)} train samples, {len(val_rows)} val samples.")

    print(f"[INFO] Loading fresh base model & tokenizer: {args.model_id}...")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class_token_ids = resolve_class_token_ids(tokenizer)
    print(f"[INFO] ChatML template | single-token class IDs: {class_token_ids}")

    # Wipe stale adapter/merged directories so fresh base training never mixes checkpoints
    run_ckpt_dir = BASE_DIR / "checkpoints" / "ollama_lora_runs"
    for d in (args.lora_dir, args.merged_dir, run_ckpt_dir):
        if d.exists():
            shutil.rmtree(d)
    args.lora_dir.mkdir(parents=True, exist_ok=True)
    args.merged_dir.mkdir(parents=True, exist_ok=True)

    train_encoded = [encode_sample(r, tokenizer, max_seq_length=args.max_seq_length) for r in train_rows]
    val_encoded = [encode_sample(r, tokenizer, max_seq_length=args.max_seq_length) for r in val_rows]

    lengths = [len(x["input_ids"]) for x in train_encoded]
    print(f"[INFO] Tokenized lengths -> min: {min(lengths)}, median: {int(np.median(lengths))}, max: {max(lengths)}")

    use_gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if use_gpu else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = IntentTokenDataset(train_encoded)
    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    class ValEvalCallback(TrainerCallback):
        def on_epoch_end(self, cb_args, state, control, **kwargs):
            nonlocal best_score, best_metrics
            if not val_rows:
                return
            metrics = evaluate_logits_on_split(model, val_rows, val_encoded, class_token_ids)
            print(
                f"\n[VAL epoch={state.epoch:.1f}] "
                f"raw_acc={metrics['raw_accuracy']:.4f} raw_f1={metrics['raw_macro_f1']:.4f} | "
                f"guarded_acc={metrics['guarded_accuracy']:.4f} guarded_f1={metrics['guarded_macro_f1']:.4f} "
                f"db_write_rec={metrics['db_write_recall']:.4f}"
            )
            score = 0.5 * metrics["guarded_macro_f1"] + 0.5 * metrics["db_write_recall"]
            if score >= best_score:
                best_score = score
                best_metrics = metrics
                model.save_pretrained(str(args.lora_dir))
                tokenizer.save_pretrained(str(args.lora_dir))
                print(f"[INFO] Saved new best LoRA adapter (score={score:.4f}) to {args.lora_dir}")

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "checkpoints" / "ollama_lora_runs"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=max(4, int(0.08 * (len(train_ds) // max(1, args.batch_size * args.grad_accum)) * args.epochs)),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        use_cpu=not use_gpu,
        dataloader_num_workers=0,
        train_sampling_strategy="group_by_length",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[ValEvalCallback()],
    )

    t0 = time.perf_counter()
    print("[INFO] Starting LoRA fine-tuning...")
    trainer.train()
    train_sec = time.perf_counter() - t0
    print(f"[OK] Training finished in {train_sec:.1f}s.")

    # Save final adapter if no val split triggered callback
    if not best_metrics:
        model.save_pretrained(str(args.lora_dir))
        tokenizer.save_pretrained(str(args.lora_dir))
        if val_rows:
            best_metrics = evaluate_logits_on_split(model, val_rows, val_encoded, class_token_ids)
    else:
        # Reload best LoRA weights before merging (free training model first to keep RAM low)
        import gc
        from peft import PeftModel
        del trainer, model
        gc.collect()
        print(f"[INFO] Reloading best LoRA checkpoint from {args.lora_dir} before merge...")
        base = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, str(args.lora_dir))

    print(f"[INFO] Merging LoRA weights and saving FP32 model to {args.merged_dir}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(args.merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.merged_dir))

    meta = {
        "base_model": args.model_id,
        "chat_family": "chatml",
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_seconds": round(train_sec, 2),
        "val_metrics": best_metrics,
        "class_token_ids": class_token_ids,
    }
    with open(args.merged_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if run_ckpt_dir.exists():
        shutil.rmtree(run_ckpt_dir, ignore_errors=True)
    print(f"[OK] Saved merged HuggingFace model & metadata to {args.merged_dir}")


if __name__ == "__main__":
    main()
