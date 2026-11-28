"""
Uploads exported GGUF weights and Ollama Modelfile (`models/ollama_gguf/`)
to a Hugging Face Hub repository using `HF_TOKEN` from `.env` or `--hf-token`.

Usage:
    python scripts/push_to_hf.py --repo-id <your-hf-username>/intent-router-qwen2.5-0.5b-instruct-gguf
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GGUF_DIR = BASE_DIR / "models" / "ollama_gguf"
DEFAULT_MERGED_DIR = BASE_DIR / "models" / "ollama_merged"
DEFAULT_EVAL_REPORT = BASE_DIR / "models" / "evaluation_report.json"


def build_hf_model_card(
    repo_id: str,
    gguf_files: List[Path],
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tag: Optional[str] = None,
) -> str:
    file_list = "\n".join(f"- `{p.name}` ({p.stat().st_size / (1024 * 1024):.1f} MB)" for p in gguf_files)
    primary_gguf = gguf_files[0].name if gguf_files else "intent-router-qwen2.5-0.5b-instruct-v1-q8_0.gguf"
    inferred_ver = (tag or "").strip() or "v1"
    if not tag:
        import re
        m = re.search(r"-(v\d+(?:\.\d+)*)-", primary_gguf)
        if m:
            inferred_ver = m.group(1)
    return f"""---
license: apache-2.0
language:
- en
base_model: {base_model}
tags:
- gguf
- ollama
- intent-classification
- text-classification
- qwen2.5
pipeline_tag: text-classification
---

# {repo_id}

Fine-tuned 4-class NLP intent router (`GENERAL_CHAT`, `DB_READ`, `DB_WRITE`, `CLARIFICATION_REQUIRED`) exported to GGUF for Ollama.

## Included Files
{file_list}
- `Modelfile` (pre-configured with system prompt, ChatML template, `temperature 0`, `num_predict 1`)

## Quickstart with Ollama
```bash
huggingface-cli download {repo_id} {primary_gguf} Modelfile --local-dir .
ollama create intent-router-qwen2.5-0.5b-instruct:{inferred_ver} -f Modelfile
```
"""


def main():
    parser = argparse.ArgumentParser(description="Push exported GGUF + Modelfile to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, metavar="USER/REPO", help="Target Hugging Face repo (e.g. username/intent-router-qwen2.5-0.5b-instruct-gguf)")
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_GGUF_DIR, help="Directory containing .gguf and Modelfile")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (defaults to HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create repository as private if it does not exist")
    parser.add_argument("--tag", type=str, default=None, help="Optional git release tag on Hugging Face Hub (e.g. v1.0.0)")
    parser.add_argument("--tag-message", type=str, default=None, help="Optional tag annotation message")
    parser.add_argument("--commit-message", type=str, default="Upload GGUF intent router and Modelfile")
    parser.add_argument("--dry-run", action="store_true", help="Verify upload artifacts without pushing to Hugging Face")
    args = parser.parse_args()

    if not args.gguf_dir.exists():
        raise FileNotFoundError(f"GGUF directory not found: {args.gguf_dir}. Run export_ollama_gguf.py or retrain_pipeline.py first.")

    gguf_files = sorted(args.gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {args.gguf_dir}.")

    modelfile = args.gguf_dir / "Modelfile"
    if not modelfile.exists():
        raise FileNotFoundError(f"Modelfile not found in {args.gguf_dir}.")

    files_to_upload: List[Path] = [*gguf_files, modelfile]
    for extra in (DEFAULT_MERGED_DIR / "training_metadata.json", DEFAULT_EVAL_REPORT):
        if extra.exists():
            files_to_upload.append(extra)

    card_path = args.gguf_dir / "README.md"
    if not card_path.exists():
        card_path.write_text(build_hf_model_card(args.repo_id, gguf_files, tag=args.tag), encoding="utf-8")
    files_to_upload.append(card_path)

    print("[INFO] Files prepared for Hugging Face Hub:")
    for fp in files_to_upload:
        print(f"  - {fp.name} ({fp.stat().st_size / (1024 * 1024):.2f} MB)")

    if args.dry_run:
        tag_note = f" with tag '{args.tag.strip()}'" if args.tag else ""
        print(f"\n[DRY-RUN] Verified {len(files_to_upload)} artifacts for '{args.repo_id}'{tag_note}. Nothing uploaded.")
        return

    token: Optional[str] = args.hf_token or os.getenv("HF_TOKEN", "").strip() or None
    if not token or token.startswith("your_"):
        raise RuntimeError("Valid HF_TOKEN not found in .env or --hf-token.")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi

    repo_id = args.repo_id.strip()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    for fp in files_to_upload:
        print(f"[INFO] Uploading {fp.name} -> {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=fp.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    if args.tag and args.tag.strip():
        clean_tag = args.tag.strip()
        print(f"[INFO] Creating Hugging Face tag '{clean_tag}' on {repo_id}...")
        api.create_tag(
            repo_id=repo_id,
            tag=clean_tag,
            tag_message=args.tag_message or f"Release {clean_tag}",
            repo_type="model",
            exist_ok=True,
        )
        print(f"[OK] Tagged release: https://huggingface.co/{repo_id}/tree/{clean_tag}")
    print(f"[OK] Uploaded GGUF model to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
