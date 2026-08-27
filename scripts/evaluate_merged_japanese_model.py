#!/usr/bin/env python3
"""Validate the FP16 merged Japanese model before stego registration."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lsteg.model import ModelManifest  # noqa: E402
from lsteg.training import (  # noqa: E402
    TrainingRecipe,
    build_completion_training_ids,
    channel_entropy_bits,
    forward_kl_nats,
    full_entropy_nats,
    load_teacher_examples,
    ratio_within_bounds,
    select_kl_rows,
)

CHANNEL_TEMPERATURE = 0.90
CHANNEL_TOP_K = 128
CHANNEL_TOP_P = 0.92


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("merged_model", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config/models/qwen3-1.7b-debug.json",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/model-cache")
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--positions-per-example", type=int, default=4)
    parser.add_argument("--min-gate-scenarios", type=int, default=8)
    parser.add_argument("--min-channel-entropy-ratio", type=float, default=0.75)
    parser.add_argument("--max-channel-entropy-ratio", type=float, default=1.35)
    parser.add_argument("--min-full-entropy-ratio", type=float, default=0.70)
    parser.add_argument("--max-full-entropy-ratio", type=float, default=1.50)
    parser.add_argument("--max-base-kl", type=float, default=0.75)
    parser.add_argument("--allow-worse-nll", action="store_true")
    parser.add_argument("--output-report", type=Path, default=None)
    return parser


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise SystemExit(
            f"missing optional model dependency {name!r}; run `uv sync --extra model`"
        ) from error


def _check_merged_metadata(path: Path, manifest: ModelManifest) -> dict[str, object]:
    metadata_path = path / "MERGED_ARTIFACT.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit("merged model is missing valid MERGED_ARTIFACT.json") from error
    if not isinstance(raw, dict):
        raise SystemExit("merged artifact metadata must be a JSON object")
    expected = {
        "base_model_id": manifest.model_id,
        "base_model_revision": manifest.model_revision,
        "tokenizer_id": manifest.tokenizer_id,
        "tokenizer_revision": manifest.tokenizer_revision,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise SystemExit(f"merged artifact metadata mismatch for {key}")
    return raw


def main() -> int:
    args = build_parser().parse_args()
    if args.max_examples < 1 or args.positions_per_example < 1:
        raise SystemExit("evaluation counts must be positive")
    recipe = TrainingRecipe.from_path(args.recipe)
    manifest = ModelManifest.from_path(args.manifest)
    metadata = _check_merged_metadata(args.merged_model, manifest)
    examples = [item for item in load_teacher_examples(args.dataset) if item.split == "eval"]
    examples = examples[: args.max_examples]
    if not examples:
        raise SystemExit("dataset has no eval examples")
    if args.min_gate_scenarios < 1:
        raise SystemExit("--min-gate-scenarios must be positive")
    eval_scenarios = len({item.scenario_id for item in examples})

    torch = _module("torch")
    transformers = _module("transformers")
    if not torch.cuda.is_available():
        raise SystemExit("merged-model validation requires CUDA")
    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        manifest.tokenizer_id,
        revision=manifest.tokenizer_revision,
        trust_remote_code=False,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    merged_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.merged_model,
        trust_remote_code=False,
        local_files_only=True,
    )

    encoded_examples: list[tuple[list[int], list[int], list[int]]] = []
    for example in examples:
        base_prompt, base_completion = build_completion_training_ids(
            base_tokenizer,
            example.prompt,
            example.completion,
            max_tokens=recipe.max_sequence_tokens,
        )
        merged_prompt, merged_completion = build_completion_training_ids(
            merged_tokenizer,
            example.prompt,
            example.completion,
            max_tokens=recipe.max_sequence_tokens,
        )
        if base_prompt != merged_prompt or base_completion != merged_completion:
            raise SystemExit("merged tokenizer changed the pinned training tokenization")
        rows = select_kl_rows(
            len(base_completion),
            args.positions_per_example,
            seed=recipe.seed + 2,
            example_key=f"merged:{example.scenario_id}:{example.generation_seed}",
        )
        encoded_examples.append((base_prompt, base_completion, rows))

    def load_model(source: str | Path, *, revision: str | None) -> Any:
        kwargs: dict[str, object] = {
            "trust_remote_code": False,
            "local_files_only": True,
            "use_safetensors": True,
            "dtype": compute_dtype,
            "device_map": {"": 0},
        }
        if revision is not None:
            kwargs["revision"] = revision
            kwargs["cache_dir"] = str(args.cache_dir)
        return transformers.AutoModelForCausalLM.from_pretrained(source, **kwargs)

    base_model = load_model(manifest.model_id, revision=manifest.model_revision)
    base_model.eval()
    base_rows: list[Any] = []
    targets_by_example: list[Any] = []
    base_nll = 0.0
    base_entropy = 0.0
    base_channel_entropy = 0.0
    position_count = 0
    with torch.no_grad():
        for prompt_ids, completion_ids, rows in encoded_examples:
            input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
            start = len(prompt_ids)
            positions = torch.arange(
                start - 1,
                start + len(completion_ids) - 1,
                dtype=torch.long,
                device=device,
            )
            row_tensor = torch.tensor(rows, dtype=torch.long, device=device)
            absolute_positions = positions.index_select(0, row_tensor)
            targets = torch.tensor(completion_ids, dtype=torch.long, device=device).index_select(
                0, row_tensor
            )
            logits = base_model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=absolute_positions,
            ).logits[0]
            base_nll += float(
                torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum").cpu()
            )
            base_entropy += float(full_entropy_nats(torch, logits).sum().cpu())
            base_channel_entropy += float(
                channel_entropy_bits(
                    torch,
                    logits,
                    temperature=CHANNEL_TEMPERATURE,
                    top_k=CHANNEL_TOP_K,
                    top_p=CHANNEL_TOP_P,
                )
                .sum()
                .cpu()
            )
            base_rows.append(logits.detach().to("cpu", dtype=torch.float16))
            targets_by_example.append(targets.detach().cpu())
            position_count += len(rows)
    del base_model
    torch.cuda.empty_cache()

    merged_model = load_model(args.merged_model, revision=None)
    merged_model.eval()
    merged_nll = 0.0
    merged_entropy = 0.0
    merged_channel_entropy = 0.0
    kl_total = 0.0
    with torch.no_grad():
        for (prompt_ids, completion_ids, rows), base_logits, targets_cpu in zip(
            encoded_examples, base_rows, targets_by_example, strict=True
        ):
            input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
            start = len(prompt_ids)
            positions = torch.arange(
                start - 1,
                start + len(completion_ids) - 1,
                dtype=torch.long,
                device=device,
            )
            row_tensor = torch.tensor(rows, dtype=torch.long, device=device)
            absolute_positions = positions.index_select(0, row_tensor)
            targets = targets_cpu.to(device=device)
            logits = merged_model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=absolute_positions,
            ).logits[0]
            merged_nll += float(
                torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum").cpu()
            )
            merged_entropy += float(full_entropy_nats(torch, logits).sum().cpu())
            merged_channel_entropy += float(
                channel_entropy_bits(
                    torch,
                    logits,
                    temperature=CHANNEL_TEMPERATURE,
                    top_k=CHANNEL_TOP_K,
                    top_p=CHANNEL_TOP_P,
                )
                .sum()
                .cpu()
            )
            kl_total += float(
                forward_kl_nats(torch, base_logits.to(device=device), logits).sum().cpu()
            )

    base_mean_nll = base_nll / position_count
    merged_mean_nll = merged_nll / position_count
    base_entropy_bits = base_entropy / position_count / math.log(2.0)
    merged_entropy_bits = merged_entropy / position_count / math.log(2.0)
    base_channel_entropy_bits = base_channel_entropy / position_count
    merged_channel_entropy_bits = merged_channel_entropy / position_count
    full_entropy_ratio = merged_entropy_bits / base_entropy_bits
    channel_entropy_ratio = merged_channel_entropy_bits / base_channel_entropy_bits
    report: dict[str, object] = {
        "schema_version": 2,
        "eval_examples": len(examples),
        "eval_scenarios": eval_scenarios,
        "min_gate_scenarios": args.min_gate_scenarios,
        "sampled_positions": position_count,
        "recipe_path": str(args.recipe),
        "recipe": recipe.as_dict(),
        "base_nll": base_mean_nll,
        "merged_nll": merged_mean_nll,
        "base_perplexity": math.exp(min(20.0, base_mean_nll)),
        "merged_perplexity": math.exp(min(20.0, merged_mean_nll)),
        "base_entropy_bits": base_entropy_bits,
        "merged_entropy_bits": merged_entropy_bits,
        "full_entropy_ratio": full_entropy_ratio,
        "channel_temperature": CHANNEL_TEMPERATURE,
        "channel_top_k": CHANNEL_TOP_K,
        "channel_top_p": CHANNEL_TOP_P,
        "base_channel_entropy_bits": base_channel_entropy_bits,
        "merged_channel_entropy_bits": merged_channel_entropy_bits,
        "channel_entropy_ratio": channel_entropy_ratio,
        "base_to_merged_kl_nats": kl_total / position_count,
        "artifact_sha256": metadata.get("artifact_sha256"),
    }
    sufficient_eval_ok = eval_scenarios >= args.min_gate_scenarios
    nll_ok = args.allow_worse_nll or merged_mean_nll <= base_mean_nll
    full_entropy_ok = ratio_within_bounds(
        full_entropy_ratio,
        minimum=args.min_full_entropy_ratio,
        maximum=args.max_full_entropy_ratio,
    )
    channel_entropy_ok = ratio_within_bounds(
        channel_entropy_ratio,
        minimum=args.min_channel_entropy_ratio,
        maximum=args.max_channel_entropy_ratio,
    )
    kl_ok = float(report["base_to_merged_kl_nats"]) <= args.max_base_kl
    report.update(
        {
            "gate_sufficient_eval_scenarios": sufficient_eval_ok,
            "gate_nll_improved": nll_ok,
            "gate_full_entropy_bounded": full_entropy_ok,
            "gate_channel_entropy_bounded": channel_entropy_ok,
            "gate_kl_bounded": kl_ok,
            "gate_pass": (
                sufficient_eval_ok and nll_ok and full_entropy_ok and channel_entropy_ok and kl_ok
            ),
        }
    )
    report_path = args.output_report or args.merged_model / "MERGED_EVALUATION.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"merged evaluation report: {report_path}")
    return 0 if report["gate_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
