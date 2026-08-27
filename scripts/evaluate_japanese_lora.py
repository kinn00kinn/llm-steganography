#!/usr/bin/env python3
"""Compare base and Japanese LoRA distributions on held-out teacher scenarios."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lsteg.model import ModelManifest  # noqa: E402
from lsteg.training import (  # noqa: E402
    TeacherValidationPolicy,
    TrainingRecipe,
    build_completion_training_ids,
    channel_entropy_bits,
    forward_kl_nats,
    full_entropy_nats,
    hash_file,
    hash_tree,
    load_teacher_examples,
    ratio_within_bounds,
    select_kl_rows,
    validate_teacher_completion,
)
from lsteg.training.prompts import JAPANESE_PROSE_SYSTEM_PROMPT  # noqa: E402

CHANNEL_TEMPERATURE = 0.90
CHANNEL_TOP_K = 128
CHANNEL_TOP_P = 0.92


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("adapter", type=Path)
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
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--positions-per-example", type=int, default=8)
    parser.add_argument("--min-gate-scenarios", type=int, default=8)
    parser.add_argument("--generation-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--min-channel-entropy-ratio", type=float, default=0.75)
    parser.add_argument("--max-channel-entropy-ratio", type=float, default=1.35)
    parser.add_argument("--min-full-entropy-ratio", type=float, default=0.70)
    parser.add_argument("--max-full-entropy-ratio", type=float, default=1.50)
    parser.add_argument("--max-base-kl", type=float, default=0.50)
    parser.add_argument("--allow-worse-nll", action="store_true")
    parser.add_argument("--output-report", type=Path, default=None)
    return parser


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise SystemExit(
            f"missing optional training dependency {name!r}; install "
            "`requirements-training.txt` into the model environment"
        ) from error


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": JAPANESE_PROSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tokenizer.encode(prompt_text, add_special_tokens=False))


def _audit_generated(text: str) -> tuple[str, tuple[str, ...]]:
    return validate_teacher_completion(
        text,
        TeacherValidationPolicy(min_chars=80, max_chars=1200, min_japanese_ratio=0.85),
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.max_examples < 1 or args.positions_per_example < 1:
        raise SystemExit("evaluation counts must be positive")
    if args.generation_samples < 0 or args.max_new_tokens < 1:
        raise SystemExit("generation counts must be non-negative and max_new_tokens positive")
    recipe = TrainingRecipe.from_path(args.recipe)
    manifest = ModelManifest.from_path(args.manifest)
    examples = [item for item in load_teacher_examples(args.dataset) if item.split == "eval"]
    examples = examples[: args.max_examples]
    if not examples:
        raise SystemExit("dataset has no eval examples")
    if args.min_gate_scenarios < 1:
        raise SystemExit("--min-gate-scenarios must be positive")
    eval_scenarios = len({item.scenario_id for item in examples})

    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    if not torch.cuda.is_available():
        raise SystemExit("evaluation requires CUDA")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        manifest.tokenizer_id,
        revision=manifest.tokenizer_revision,
        trust_remote_code=False,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        manifest.model_id,
        revision=manifest.model_revision,
        trust_remote_code=False,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        use_safetensors=True,
        quantization_config=quant,
        device_map={"": 0},
    )
    model = peft.PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    model.eval()
    device = torch.device("cuda:0")

    base_nll = 0.0
    adapted_nll = 0.0
    base_entropy = 0.0
    adapted_entropy = 0.0
    base_channel_entropy = 0.0
    adapted_channel_entropy = 0.0
    kl_total = 0.0
    position_count = 0

    with torch.no_grad():
        for example in examples:
            prompt_ids, completion_ids = build_completion_training_ids(
                tokenizer,
                example.prompt,
                example.completion,
                max_tokens=recipe.max_sequence_tokens,
            )
            input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            start = len(prompt_ids)
            prediction_positions = torch.arange(
                start - 1,
                start + len(completion_ids) - 1,
                dtype=torch.long,
                device=device,
            )
            selected_rows = select_kl_rows(
                len(completion_ids),
                args.positions_per_example,
                seed=recipe.seed + 1,
                example_key=f"eval:{example.scenario_id}:{example.generation_seed}",
            )
            row_tensor = torch.tensor(selected_rows, dtype=torch.long, device=device)
            absolute_positions = prediction_positions.index_select(0, row_tensor)
            targets = torch.tensor(completion_ids, dtype=torch.long, device=device).index_select(
                0, row_tensor
            )

            with model.disable_adapter():
                base_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=absolute_positions,
                ).logits[0]
            adapted_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=absolute_positions,
            ).logits[0]
            base_nll += float(
                torch.nn.functional.cross_entropy(
                    base_logits.float(), targets, reduction="sum"
                ).cpu()
            )
            adapted_nll += float(
                torch.nn.functional.cross_entropy(
                    adapted_logits.float(), targets, reduction="sum"
                ).cpu()
            )
            base_entropy += float(full_entropy_nats(torch, base_logits).sum().cpu())
            adapted_entropy += float(full_entropy_nats(torch, adapted_logits).sum().cpu())
            base_channel_entropy += float(
                channel_entropy_bits(
                    torch,
                    base_logits,
                    temperature=CHANNEL_TEMPERATURE,
                    top_k=CHANNEL_TOP_K,
                    top_p=CHANNEL_TOP_P,
                )
                .sum()
                .cpu()
            )
            adapted_channel_entropy += float(
                channel_entropy_bits(
                    torch,
                    adapted_logits,
                    temperature=CHANNEL_TEMPERATURE,
                    top_k=CHANNEL_TOP_K,
                    top_p=CHANNEL_TOP_P,
                )
                .sum()
                .cpu()
            )
            kl_total += float(forward_kl_nats(torch, base_logits, adapted_logits).sum().cpu())
            position_count += len(selected_rows)

    if position_count == 0:
        raise SystemExit("no evaluation positions were selected")

    base_entropy_bits = base_entropy / position_count / math.log(2.0)
    adapted_entropy_bits = adapted_entropy / position_count / math.log(2.0)
    base_channel_entropy_bits = base_channel_entropy / position_count
    adapted_channel_entropy_bits = adapted_channel_entropy / position_count
    full_entropy_ratio = adapted_entropy_bits / base_entropy_bits
    channel_entropy_ratio = adapted_channel_entropy_bits / base_channel_entropy_bits

    report: dict[str, object] = {
        "schema_version": 2,
        "dataset_sha256": hash_file(args.dataset),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": hash_tree(args.adapter),
        "base_model_id": manifest.model_id,
        "base_model_revision": manifest.model_revision,
        "tokenizer_id": manifest.tokenizer_id,
        "tokenizer_revision": manifest.tokenizer_revision,
        "recipe_path": str(args.recipe),
        "recipe": recipe.as_dict(),
        "eval_examples": len(examples),
        "eval_scenarios": eval_scenarios,
        "min_gate_scenarios": args.min_gate_scenarios,
        "sampled_positions": position_count,
        "base_nll": base_nll / position_count,
        "adapter_nll": adapted_nll / position_count,
        "base_perplexity": math.exp(min(20.0, base_nll / position_count)),
        "adapter_perplexity": math.exp(min(20.0, adapted_nll / position_count)),
        "base_entropy_bits": base_entropy_bits,
        "adapter_entropy_bits": adapted_entropy_bits,
        "full_entropy_ratio": full_entropy_ratio,
        "channel_temperature": CHANNEL_TEMPERATURE,
        "channel_top_k": CHANNEL_TOP_K,
        "channel_top_p": CHANNEL_TOP_P,
        "base_channel_entropy_bits": base_channel_entropy_bits,
        "adapter_channel_entropy_bits": adapted_channel_entropy_bits,
        "channel_entropy_ratio": channel_entropy_ratio,
        "base_to_adapter_kl_nats": kl_total / position_count,
    }

    base_rejections: Counter[str] = Counter()
    adapted_rejections: Counter[str] = Counter()
    generation_rows: list[dict[str, object]] = []
    for sample_index, example in enumerate(examples[: args.generation_samples], start=1):
        ids = torch.tensor(
            [_prompt_ids(tokenizer, example.prompt)], dtype=torch.long, device=device
        )
        generation_kwargs = {
            "input_ids": ids,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": CHANNEL_TEMPERATURE,
            "top_k": CHANNEL_TOP_K,
            "top_p": CHANNEL_TOP_P,
            "pad_token_id": tokenizer.eos_token_id,
        }
        torch.manual_seed(recipe.seed + 10_000 + sample_index)
        with model.disable_adapter(), torch.no_grad():
            base_generated = model.generate(**generation_kwargs)
        torch.manual_seed(recipe.seed + 10_000 + sample_index)
        with torch.no_grad():
            adapted_generated = model.generate(**generation_kwargs)
        prefix_len = ids.shape[1]
        base_text = tokenizer.decode(base_generated[0, prefix_len:], skip_special_tokens=True)
        adapted_text = tokenizer.decode(adapted_generated[0, prefix_len:], skip_special_tokens=True)
        _, base_reasons = _audit_generated(base_text)
        _, adapted_reasons = _audit_generated(adapted_text)
        base_rejections.update(base_reasons)
        adapted_rejections.update(adapted_reasons)
        generation_rows.append(
            {
                "sample": sample_index,
                "scenario_id": example.scenario_id,
                "base_reasons": list(base_reasons),
                "adapter_reasons": list(adapted_reasons),
            }
        )
        print(f"\n--- EVAL SAMPLE {sample_index}: BASE ---")
        print(base_text)
        print(f"audit: {list(base_reasons) or ['PASS']}")
        print(f"\n--- EVAL SAMPLE {sample_index}: LORA ---")
        print(adapted_text)
        print(f"audit: {list(adapted_reasons) or ['PASS']}")

    report["generation_samples"] = generation_rows
    report["base_generation_rejections"] = dict(sorted(base_rejections.items()))
    report["adapter_generation_rejections"] = dict(sorted(adapted_rejections.items()))

    sufficient_eval_ok = eval_scenarios >= args.min_gate_scenarios
    nll_ok = args.allow_worse_nll or report["adapter_nll"] <= report["base_nll"]
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
    kl_ok = report["base_to_adapter_kl_nats"] <= args.max_base_kl
    report["gate_sufficient_eval_scenarios"] = sufficient_eval_ok
    report["gate_nll_improved"] = nll_ok
    report["gate_full_entropy_bounded"] = full_entropy_ok
    report["gate_channel_entropy_bounded"] = channel_entropy_ok
    report["gate_kl_bounded"] = kl_ok
    report["gate_pass"] = (
        sufficient_eval_ok and nll_ok and full_entropy_ok and channel_entropy_ok and kl_ok
    )

    report_path = args.output_report or args.adapter.parent / "evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n--- DISTRIBUTION EVALUATION ---")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"evaluation report: {report_path}")
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
