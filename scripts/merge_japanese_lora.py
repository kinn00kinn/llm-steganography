#!/usr/bin/env python3
"""Merge a validated LoRA adapter into the pinned float16 Qwen3-1.7B base."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lsteg.model import ModelManifest  # noqa: E402
from lsteg.training import hash_tree  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config/models/qwen3-1.7b-debug.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/model-cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/models/qwen3-1.7b-japanese-prose-merged-v1",
    )
    parser.add_argument("--evaluation-report", type=Path, default=None)
    parser.add_argument("--allow-unvalidated", action="store_true")
    return parser


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise SystemExit(
            f"missing optional training dependency {name!r}; install "
            "`requirements-training.txt` into the model environment"
        ) from error


def main() -> int:
    args = build_parser().parse_args()
    manifest = ModelManifest.from_path(args.manifest)
    evaluation_report = args.evaluation_report or args.adapter.parent / "evaluation.json"
    if not args.allow_unvalidated:
        try:
            evaluation = json.loads(evaluation_report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(
                "validated merge requires an evaluation report; run evaluate_japanese_lora.py first"
            ) from error
        if not isinstance(evaluation, dict) or evaluation.get("gate_pass") is not True:
            raise SystemExit("LoRA evaluation gate did not pass; refusing to merge")
        adapter_hash = hash_tree(args.adapter)
        if evaluation.get("adapter_sha256") != adapter_hash:
            raise SystemExit("evaluation report does not match this LoRA adapter")
        expected_identity = {
            "base_model_id": manifest.model_id,
            "base_model_revision": manifest.model_revision,
            "tokenizer_id": manifest.tokenizer_id,
            "tokenizer_revision": manifest.tokenizer_revision,
        }
        for key, value in expected_identity.items():
            if evaluation.get(key) != value:
                raise SystemExit(f"evaluation report identity mismatch for {key}")
    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        manifest.model_id,
        revision=manifest.model_revision,
        trust_remote_code=False,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float16,
        device_map={"": "cpu"},
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        manifest.tokenizer_id,
        revision=manifest.tokenizer_revision,
        trust_remote_code=False,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    adapted = peft.PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    artifact_hash = hash_tree(args.output_dir, exclude_names={"MERGED_ARTIFACT.json"})
    metadata = {
        "schema_version": 1,
        "base_model_id": manifest.model_id,
        "base_model_revision": manifest.model_revision,
        "tokenizer_id": manifest.tokenizer_id,
        "tokenizer_revision": manifest.tokenizer_revision,
        "adapter": str(args.adapter),
        "evaluation_report": str(evaluation_report) if evaluation_report.exists() else None,
        "artifact_sha256": artifact_hash,
        "note": (
            "Tokenizer is unchanged from the pinned base; validate this FP16 merged artifact "
            "before stego registration."
        ),
    }
    (args.output_dir / "MERGED_ARTIFACT.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
