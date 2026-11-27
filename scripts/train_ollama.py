"""
Fine-tunes a compact causal LLM (default: Qwen/Qwen2.5-0.5B-Instruct) with LoRA
on single-turn and multi-turn intent routing data so it classifies prompts
into a single token (CHAT, READ, WRITE, or ASK) with calibrated next-token probabilities.

Designed for CPU-only Windows or GPU training:
- Dynamic batch padding + length-grouped sampling
- Single decision-token extraction (no full-sequence vocab cross-entropy overhead)
- Hybrid loss: vocab anchor CE (keeps CHAT/READ/WRITE/ASK in top_logprobs) +
  4-class calibrated CE with class-only label smoothing and gentle class weighting
- Batched validation scoring Accuracy, Macro F1, DB_WRITE Recall, CLARIFICATION Recall,
  and 4-class NLL/Brier calibration
- Merges LoRA weights into FP32 HuggingFace format at models/ollama_merged/
"""

import os
import sys
import json
import math
import time
import shutil
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    apply_routing_guardrails,
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
    max_seq_length: int = 512,
    class_token_ids: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    label = row["label"]
    if class_token_ids is not None:
        target_id = class_token_ids[label]
    else:
        target_ids_tmp = tokenizer.encode(ROUTE_TO_TOKEN[label], add_special_tokens=False)
        if len(target_ids_tmp) != 1:
            raise ValueError(f"Expected 1 token for '{ROUTE_TO_TOKEN[label]}', got {target_ids_tmp}")
        target_id = target_ids_tmp[0]

    history = list(row.get("chat_history") or [])
    history_trimmed = False
    prompt_truncated = False
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
        if len(prompt_ids) + 1 <= max_seq_length or not history:
            break
        # Drop oldest history turn if conversation exceeds max_seq_length
        history = history[1:]
        history_trimmed = True

    if len(prompt_ids) + 1 > max_seq_length:
        keep_prompt = max_seq_length - 1
        prompt_ids = prompt_ids[-keep_prompt:]
        prompt_truncated = True

    input_ids = prompt_ids + [target_id]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decision_pos": len(prompt_ids) - 1,
        "target_vocab_id": target_id,
        "target_class_idx": CLASSES.index(label),
        "prompt_ids": prompt_ids,
        "history_trimmed": history_trimmed,
        "prompt_truncated": prompt_truncated,
    }


class IntentTokenDataset(TorchDataset):
    def __init__(self, encoded_rows: List[Dict[str, Any]]):
        self.rows = encoded_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "input_ids": r["input_ids"],
            "attention_mask": r["attention_mask"],
            "decision_pos": r["decision_pos"],
            "target_vocab_id": r["target_vocab_id"],
            "target_class_idx": r["target_class_idx"],
        }


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_mask = [], []
        decision_pos, target_vocab_ids, target_class_idx = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_mask.append(f["attention_mask"] + [0] * pad_len)
            decision_pos.append(f["decision_pos"])
            target_vocab_ids.append(f["target_vocab_id"])
            target_class_idx.append(f["target_class_idx"])
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "decision_pos": torch.tensor(decision_pos, dtype=torch.long),
            "target_vocab_ids": torch.tensor(target_vocab_ids, dtype=torch.long),
            "target_class_idx": torch.tensor(target_class_idx, dtype=torch.long),
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


def compute_class_weights(train_rows: List[Dict[str, Any]]) -> torch.Tensor:
    """Gentle inverse-sqrt class weights normalized to mean 1.0."""
    counts = Counter(r["label"] for r in train_rows)
    raw = [1.0 / math.sqrt(max(1, counts.get(c, 1))) for c in CLASSES]
    mean_w = sum(raw) / len(raw)
    return torch.tensor([w / mean_w for w in raw], dtype=torch.float32)


def evaluate_logits_on_split(
    model: Any,
    val_rows: List[Dict[str, Any]],
    val_encoded: List[Dict[str, Any]],
    class_token_ids: Dict[str, int],
    write_threshold: float = settings.WRITE_THRESHOLD,
    clarify_threshold: float = settings.CLARIFY_THRESHOLD,
    confidence_threshold: float = settings.CONFIDENCE_THRESHOLD,
    fallback_route: str = settings.FALLBACK_ROUTE,
    batch_size: int = 16,
    pad_token_id: int = 0,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    ordered_ids = [class_token_ids[c] for c in CLASSES]
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    # Sort indices by prompt length to minimize padding during batched eval, then restore original order
    sorted_indices = sorted(range(len(val_encoded)), key=lambda i: len(val_encoded[i]["prompt_ids"]))
    prob_rows: List[Optional[List[float]]] = [None] * len(val_encoded)

    with torch.no_grad():
        for start in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[start:start + batch_size]
            prompts = [val_encoded[i]["prompt_ids"] for i in batch_indices]
            max_len = max(len(p) for p in prompts)
            b_ids, b_mask, last_pos = [], [], []
            for p in prompts:
                pad_len = max_len - len(p)
                b_ids.append(p + [pad_token_id] * pad_len)
                b_mask.append([1] * len(p) + [0] * pad_len)
                last_pos.append(len(p) - 1)

            inp = torch.tensor(b_ids, dtype=torch.long, device=device)
            mask = torch.tensor(b_mask, dtype=torch.long, device=device)
            pos = torch.tensor(last_pos, dtype=torch.long, device=device)
            out = model(input_ids=inp, attention_mask=mask)
            b_idx = torch.arange(len(batch_indices), device=device)
            next_logits = out.logits[b_idx, pos, :][:, ordered_ids]
            probs_batch = F.softmax(next_logits.float(), dim=-1).cpu().tolist()
            for idx, probs in zip(batch_indices, probs_batch):
                prob_rows[idx] = probs

    y_true = [r["label"] for r in val_rows]
    y_raw: List[str] = []
    y_guarded: List[str] = []
    nll_sum = 0.0
    brier_sum = 0.0

    for row, probs in zip(val_rows, prob_rows):
        assert probs is not None
        prob_map = {c: float(p) for c, p in zip(CLASSES, probs)}
        guarded_pred, raw_pred, _, _ = apply_routing_guardrails(
            probabilities=prob_map,
            confidence_threshold=confidence_threshold,
            write_threshold=write_threshold,
            clarify_threshold=clarify_threshold,
            fallback_route=fallback_route,
            prompt=row["text"],
        )
        y_raw.append(raw_pred)
        y_guarded.append(guarded_pred)

        true_idx = class_to_idx[row["label"]]
        nll_sum += -math.log(max(probs[true_idx], 1e-12))
        brier_sum += sum((probs[j] - (1.0 if j == true_idx else 0.0)) ** 2 for j in range(len(CLASSES)))

    raw_acc = accuracy_score(y_true, y_raw)
    raw_f1 = f1_score(y_true, y_raw, average="macro", zero_division=0)
    guarded_acc = accuracy_score(y_true, y_guarded)
    guarded_f1 = f1_score(y_true, y_guarded, average="macro", zero_division=0)
    write_rec = recall_score(y_true, y_guarded, labels=["DB_WRITE"], average=None, zero_division=0)[0]
    clarify_rec = recall_score(y_true, y_guarded, labels=["CLARIFICATION_REQUIRED"], average=None, zero_division=0)[0]
    n_samples = max(1, len(val_rows))

    model.train()
    return {
        "raw_accuracy": round(float(raw_acc), 4),
        "raw_macro_f1": round(float(raw_f1), 4),
        "guarded_accuracy": round(float(guarded_acc), 4),
        "guarded_macro_f1": round(float(guarded_f1), 4),
        "db_write_recall": round(float(write_rec), 4),
        "clarification_recall": round(float(clarify_rec), 4),
        "val_nll": round(float(nll_sum / n_samples), 4),
        "val_brier": round(float(brier_sum / n_samples), 4),
    }


def main():
    default_model_id = str(BASE_DIR / "models" / "hf_base" / "Qwen2.5-0.5B-Instruct")
    if not Path(default_model_id).exists():
        default_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-0.5B-Instruct as 1-token 4-class Ollama intent classifier.")
    parser.add_argument("--model-id", type=str, default=default_model_id, help="Base HuggingFace causal LM or local directory")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=-1, help="LoRA alpha (<=0 auto-sets 2 * lora_r)")
    parser.add_argument("--lora-dropout", type=float, default=0.10, help="LoRA dropout rate")
    parser.add_argument("--class-label-smoothing", type=float, default=0.03, help="Label smoothing over the 4 valid classes only")
    parser.add_argument("--vocab-loss-weight", type=float, default=0.40, help="Weight on full-vocab CE anchor loss vs 4-class CE loss")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length (protects multi-turn history + system prompt)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional cap on optimizer steps (-1 = full epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training reproducibility")
    parser.add_argument("--cpu-threads", type=int, default=min(12, os.cpu_count() or 8), help="PyTorch CPU intra-op threads")
    parser.add_argument("--lora-dir", type=Path, default=LORA_OUT_DIR, help="Output dir for LoRA adapter")
    parser.add_argument("--merged-dir", type=Path, default=MERGED_OUT_DIR, help="Output dir for merged HF weights")
    args = parser.parse_args()

    effective_alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_r

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model

    set_seed(args.seed)
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
    ordered_class_ids = [class_token_ids[c] for c in CLASSES]
    class_weights = compute_class_weights(train_rows)
    print(f"[INFO] ChatML template | single-token 4-class IDs: {class_token_ids}")
    print(f"[INFO] Inverse-sqrt class weights: {[round(float(w), 3) for w in class_weights.tolist()]}")

    # Wipe stale adapter/merged directories so fresh base training never mixes checkpoints
    run_ckpt_dir = BASE_DIR / "checkpoints" / "ollama_lora_runs"
    for d in (args.lora_dir, args.merged_dir, run_ckpt_dir):
        if d.exists():
            shutil.rmtree(d)
    args.lora_dir.mkdir(parents=True, exist_ok=True)
    args.merged_dir.mkdir(parents=True, exist_ok=True)

    train_encoded = [encode_sample(r, tokenizer, max_seq_length=args.max_seq_length, class_token_ids=class_token_ids) for r in train_rows]
    val_encoded = [encode_sample(r, tokenizer, max_seq_length=args.max_seq_length, class_token_ids=class_token_ids) for r in val_rows]

    lengths = [len(x["input_ids"]) for x in train_encoded]
    trimmed = sum(1 for x in train_encoded if x["history_trimmed"])
    truncated = sum(1 for x in train_encoded if x["prompt_truncated"])
    print(
        f"[INFO] Tokenized lengths -> min: {min(lengths)}, median: {int(np.median(lengths))}, max: {max(lengths)} "
        f"(history_trimmed={trimmed}, prompt_truncated={truncated}, max_seq_length={args.max_seq_length})"
    )

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
        lora_alpha=effective_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = IntentTokenDataset(train_encoded)
    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    class IntentClassifierTrainer(Trainer):
        def __init__(self, *t_args, **t_kwargs):
            super().__init__(*t_args, **t_kwargs)
            # compute_loss returns a per-batch mean; let Trainer divide by gradient_accumulation_steps
            self.loss_is_scaled_for_ga = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            logits = outputs.logits
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            dec_logits = logits[batch_idx, inputs["decision_pos"], :].float()

            loss_vocab = F.cross_entropy(dec_logits, inputs["target_vocab_ids"])
            four_ids = torch.tensor(ordered_class_ids, dtype=torch.long, device=logits.device)
            four_logits = dec_logits.index_select(dim=-1, index=four_ids)
            weights = class_weights.to(logits.device)
            loss_4 = F.cross_entropy(
                four_logits,
                inputs["target_class_idx"],
                weight=weights,
                label_smoothing=args.class_label_smoothing,
            )
            loss = args.vocab_loss_weight * loss_vocab + (1.0 - args.vocab_loss_weight) * loss_4
            return (loss, outputs) if return_outputs else loss

    class ValEvalCallback(TrainerCallback):
        def on_epoch_end(self, cb_args, state, control, **kwargs):
            nonlocal best_score, best_metrics
            if not val_rows:
                return
            metrics = evaluate_logits_on_split(
                model,
                val_rows,
                val_encoded,
                class_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
            print(
                f"\n[VAL epoch={state.epoch:.1f}] "
                f"raw_acc={metrics['raw_accuracy']:.4f} raw_f1={metrics['raw_macro_f1']:.4f} | "
                f"guarded_acc={metrics['guarded_accuracy']:.4f} guarded_f1={metrics['guarded_macro_f1']:.4f} "
                f"db_write_rec={metrics['db_write_recall']:.4f} clarify_rec={metrics['clarification_recall']:.4f} "
                f"nll={metrics['val_nll']:.4f} brier={metrics['val_brier']:.4f}"
            )
            score = (
                0.50 * metrics["guarded_macro_f1"]
                + 0.25 * metrics["db_write_recall"]
                + 0.25 * metrics["clarification_recall"]
                - 0.005 * metrics["val_nll"]
            )
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
        weight_decay=args.weight_decay,
        warmup_steps=max(4, int(0.08 * (len(train_ds) // max(1, args.batch_size * args.grad_accum)) * args.epochs)),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=args.seed,
        use_cpu=not use_gpu,
        dataloader_num_workers=0,
        train_sampling_strategy="group_by_length",
        remove_unused_columns=False,
    )

    trainer = IntentClassifierTrainer(
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
            best_metrics = evaluate_logits_on_split(
                model,
                val_rows,
                val_encoded,
                class_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
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
        "classes": CLASSES,
        "route_to_token": ROUTE_TO_TOKEN,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": effective_alpha,
        "lora_dropout": args.lora_dropout,
        "class_label_smoothing": args.class_label_smoothing,
        "vocab_loss_weight": args.vocab_loss_weight,
        "seed": args.seed,
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
