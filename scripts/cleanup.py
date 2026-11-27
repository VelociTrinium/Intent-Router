"""
Standalone, zero-dependency cleanup utility for `intent-router`.

Removes temporary training/epoch checkpoints, LoRA adapters, merged FP32 weights,
and exported GGUF files from the workspace once the model has been imported into
Ollama's local blob store (and/or pushed to Hugging Face).

By default, preserves:
- `models/hf_base/` (reusable base weights so future retrains run offline)
- `models/evaluation_report.json` (lightweight test metrics report)
- `data/seed_prompts.json`, `data/custom/`, `data/raw/`, `data/processed/`
- `.venv/`, `.env`, `docs/`, `app/`, `scripts/`, `tests/`

Usage:
    python scripts/cleanup.py --dry-run
    python scripts/cleanup.py --yes
    python scripts/cleanup.py --include-caches --yes
    python scripts/cleanup.py --include-data --include-base --yes
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_BUILD_TARGETS = (
    "checkpoints",
    "models/ollama_lora",
    "models/ollama_merged",
    "models/ollama_gguf",
    "models/ollama_gguf_archive",
)

PROJECT_CACHE_DIRS = (
    ".pytest_cache",
    "app/__pycache__",
    "scripts/__pycache__",
    "scripts/vendor/conversion/__pycache__",
    "scripts/vendor/gguf-py/gguf/__pycache__",
    "tests/__pycache__",
)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _ensure_within_repo(target: Path) -> Path:
    resolved = target.resolve()
    repo_root = BASE_DIR.resolve()
    if resolved == repo_root or repo_root not in resolved.parents:
        raise ValueError(f"Refusing to delete path outside repo root: {resolved}")
    if ".venv" in resolved.parts or ".git" in resolved.parts:
        raise ValueError(f"Refusing to touch protected directory: {resolved}")
    return resolved


def collect_targets(
    include_caches: bool = False,
    include_data: bool = False,
    include_base: bool = False,
    include_reports: bool = False,
) -> List[Tuple[Path, str]]:
    """Collects existing files/directories to remove along with human labels."""
    candidates: List[Tuple[Path, str]] = []

    for rel in DEFAULT_BUILD_TARGETS:
        p = BASE_DIR / rel
        if p.exists():
            candidates.append((p, rel))

    if include_caches:
        for rel in PROJECT_CACHE_DIRS:
            p = BASE_DIR / rel
            if p.exists():
                candidates.append((p, rel))

    if include_data:
        for sub in ("data/raw", "data/processed"):
            folder = BASE_DIR / sub
            if folder.exists():
                for child in sorted(folder.iterdir()):
                    if child.name == ".gitkeep":
                        continue
                    candidates.append((child, f"{sub}/{child.name}"))

    if include_base:
        p = BASE_DIR / "models/hf_base"
        if p.exists():
            candidates.append((p, "models/hf_base"))

    if include_reports:
        p = BASE_DIR / "models/evaluation_report.json"
        if p.exists():
            candidates.append((p, "models/evaluation_report.json"))

    return candidates


def remove_path(target: Path):
    safe_target = _ensure_within_repo(target)
    if safe_target.is_dir():
        shutil.rmtree(safe_target)
    elif safe_target.exists():
        safe_target.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Remove build/training/export artifacts while keeping Ollama blob models intact."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show paths and reclaimable size without deleting anything",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Delete without interactive confirmation prompt",
    )
    parser.add_argument(
        "--include-caches",
        action="store_true",
        help="Also remove project __pycache__ and .pytest_cache folders (never touches .venv)",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Also remove generated files under data/raw/ and data/processed/ (preserves .gitkeep)",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also remove cached HuggingFace base weights under models/hf_base/ (~953 MB)",
    )
    parser.add_argument(
        "--include-reports",
        action="store_true",
        help="Also remove models/evaluation_report.json",
    )
    args = parser.parse_args()

    targets = collect_targets(
        include_caches=args.include_caches,
        include_data=args.include_data,
        include_base=args.include_base,
        include_reports=args.include_reports,
    )

    if not targets:
        print("[OK] Workspace is already clean. Nothing to remove.")
        return

    sized: List[Tuple[Path, str, int]] = []
    total_bytes = 0
    for path, label in targets:
        sz = _dir_size_bytes(path)
        total_bytes += sz
        sized.append((path, label, sz))

    print("\n[CLEANUP TARGETS]")
    for _, label, sz in sized:
        print(f"  - {label:<42} {sz / (1024 * 1024):>8.2f} MB")
    print(f"  Total reclaimable: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 ** 3):.2f} GB)\n")

    if args.dry_run:
        print("[DRY-RUN] No files were deleted.")
        return

    if not args.yes:
        ans = input("Proceed with deleting the items above? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("[ABORTED] No files were deleted.")
            return

    deleted_bytes = 0
    for path, label, sz in sized:
        remove_path(path)
        deleted_bytes += sz
        print(f"[REMOVED] {label} ({sz / (1024 * 1024):.2f} MB)")

    print(f"\n[OK] Cleanup complete. Reclaimed {deleted_bytes / (1024 * 1024):.2f} MB.")


if __name__ == "__main__":
    main()
