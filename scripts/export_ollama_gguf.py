"""
Exports a merged HuggingFace intent classifier (`models/ollama_merged/`)
to standalone GGUF (default Q8_0 via official llama.cpp `convert_hf_to_gguf.py`),
auto-detects the chat template family (`chatml`, `llama3`, `gemma`), writes a
tailored Ollama Modelfile preserving the 1-token CHAT/READ/WRITE contract, and
registers the model in local Ollama.
"""

import re
import sys
import json
import subprocess
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR.resolve()]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.intent_spec import OLLAMA_CLASSIFIER_SYSTEM_PROMPT

DEFAULT_MERGED_DIR = BASE_DIR / "models" / "ollama_merged"
DEFAULT_OUT_DIR = BASE_DIR / "models" / "ollama_gguf"
CONVERTER_PATH = SCRIPTS_DIR / "vendor" / "convert_hf_to_gguf.py"
LLAMA_CPP_RAW_BASE = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master"

CHATML_TEMPLATE = '''{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}{{ end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}'''

LLAMA3_TEMPLATE = '''{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>{{ else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>{{ end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>

'''

GEMMA_TEMPLATE = '''{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<start_of_turn>user
{{ if and $.System (eq $i 0) }}{{ $.System }}

{{ end }}{{ .Content }}<end_of_turn>
{{ else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last }}<start_of_turn>model
{{ end }}
{{- end }}'''

TEMPLATE_SPECS: Dict[str, Tuple[str, List[str]]] = {
    "chatml": (CHATML_TEMPLATE, ["<|im_end|>", "<|endoftext|>"]),
    "llama3": (LLAMA3_TEMPLATE, ["<|eot_id|>", "<|end_of_text|>"]),
    "gemma": (GEMMA_TEMPLATE, ["<end_of_turn>", "<eos>"]),
}


def detect_template_family(merged_dir: Path) -> str:
    meta_file = merged_dir / "training_metadata.json"
    if meta_file.exists():
        try:
            fam = json.loads(meta_file.read_text(encoding="utf-8")).get("chat_family")
            if fam in TEMPLATE_SPECS:
                return fam
        except Exception:
            pass

    tok_cfg = merged_dir / "tokenizer_config.json"
    if tok_cfg.exists():
        raw = tok_cfg.read_text(encoding="utf-8")
        if "<|eot_id|>" in raw:
            return "llama3"
        if "<end_of_turn>" in raw:
            return "gemma"
    return "chatml"


def ensure_converter(converter_path: Path) -> Path:
    vendor_dir = converter_path.parent
    conv_dir = vendor_dir / "conversion"
    gguf_dir = vendor_dir / "gguf-py" / "gguf"
    conv_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "convert_hf_to_gguf.py",
        "conversion/__init__.py",
        "conversion/base.py",
        "conversion/llama.py",
        "conversion/qwen.py",
        "conversion/gemma.py",
        "gguf-py/gguf/__init__.py",
        "gguf-py/gguf/constants.py",
        "gguf-py/gguf/gguf.py",
        "gguf-py/gguf/gguf_reader.py",
        "gguf-py/gguf/gguf_writer.py",
        "gguf-py/gguf/lazy.py",
        "gguf-py/gguf/metadata.py",
        "gguf-py/gguf/quants.py",
        "gguf-py/gguf/tensor_mapping.py",
        "gguf-py/gguf/utility.py",
        "gguf-py/gguf/vocab.py",
    ]
    for rel_path in required:
        local_path = vendor_dir / rel_path
        if not local_path.exists() or local_path.stat().st_size == 0:
            url = f"{LLAMA_CPP_RAW_BASE}/{rel_path}"
            print(f"[INFO] Downloading {rel_path}...")
            with urllib.request.urlopen(url, timeout=15) as resp:
                local_path.write_bytes(resp.read())
    return converter_path


def write_modelfile(modelfile_path: Path, gguf_filename: str, chat_family: str):
    template, stop_tokens = TEMPLATE_SPECS.get(chat_family, TEMPLATE_SPECS["chatml"])
    stop_lines = "\n".join(f'PARAMETER stop "{tok}"' for tok in stop_tokens)
    content = (
        f"FROM ./{gguf_filename}\n\n"
        f'TEMPLATE """{template}"""\n\n'
        f'SYSTEM """{OLLAMA_CLASSIFIER_SYSTEM_PROMPT}"""\n\n'
        "PARAMETER temperature 0\n"
        "PARAMETER top_k 10\n"
        "PARAMETER num_predict 1\n"
        f"{stop_lines}\n"
    )
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote Modelfile ({chat_family}) to {modelfile_path}")


def resolve_ollama_ref(model_name: str, version: Optional[str] = None) -> Tuple[str, str]:
    """
    Combines `model_name` and optional `version` into an Ollama tag
    (e.g. 'intent-router-qwen2.5-0.5b-instruct:v1') and returns a
    Windows-safe filename stem (no ':').
    """
    clean_name = model_name.strip()
    clean_ver = (version or "").strip().lstrip(":")
    if clean_ver and ":" not in clean_name:
        ollama_ref = f"{clean_name}:{clean_ver}"
    else:
        ollama_ref = clean_name
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", ollama_ref).strip("-")
    return ollama_ref, safe_stem


def main():
    parser = argparse.ArgumentParser(description="Convert merged HF classifier to GGUF and register in Ollama.")
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR, help="Path to merged HF model")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for GGUF & Modelfile")
    parser.add_argument("--outtype", choices=["q8_0", "f16", "bf16", "f32"], default="q8_0", help="GGUF tensor type (default: q8_0)")
    parser.add_argument("--ollama-quantize", type=str, default=None, help="Optional `ollama create -q` target (e.g. q4_K_M)")
    parser.add_argument("--model-name", type=str, default="intent-router-qwen2.5-0.5b-instruct", help="Ollama model repository name")
    parser.add_argument("--version", type=str, default="v1", help="Ollama model version tag (e.g. v1, 1.0.0, latest; empty to omit)")
    parser.add_argument("--skip-ollama-create", action="store_true", help="Only export GGUF + Modelfile without calling ollama create")
    args = parser.parse_args()

    if not args.merged_dir.exists() or not (args.merged_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Merged HuggingFace model not found at {args.merged_dir}. "
            "Run `python scripts/train.py --backend ollama` first."
        )

    ollama_ref, safe_stem = resolve_ollama_ref(args.model_name, args.version)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gguf_filename = f"{safe_stem}-{args.outtype}.gguf"
    gguf_path = args.out_dir / gguf_filename

    converter = ensure_converter(CONVERTER_PATH)
    print(f"[INFO] Converting {args.merged_dir} -> {gguf_path} (outtype={args.outtype})...")
    subprocess.run(
        [sys.executable, str(converter), str(args.merged_dir), "--outfile", str(gguf_path), "--outtype", args.outtype],
        check=True,
    )
    size_mb = gguf_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Exported GGUF ({size_mb:.1f} MB): {gguf_path}")

    chat_family = detect_template_family(args.merged_dir)
    modelfile_path = args.out_dir / "Modelfile"
    write_modelfile(modelfile_path, gguf_filename, chat_family=chat_family)

    if args.skip_ollama_create:
        print("[INFO] --skip-ollama-create set. Skipping `ollama create`.")
        return

    ollama_cmd = ["ollama", "create", ollama_ref, "-f", str(modelfile_path)]
    if args.ollama_quantize:
        ollama_cmd.extend(["-q", args.ollama_quantize])

    print(f"[INFO] Registering model in Ollama: {' '.join(ollama_cmd)}")
    subprocess.run(ollama_cmd, cwd=str(args.out_dir), check=True)
    print(f"[OK] Model '{ollama_ref}' registered in Ollama!")


if __name__ == "__main__":
    main()
