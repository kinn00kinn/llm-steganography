#!/usr/bin/env python3
"""Train the Japanese-prose QLoRA adapter with sparse base-model KL anchoring."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
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
    forward_kl_nats,
    hash_file,
    hash_tree,
    load_teacher_examples,
    select_kl_rows,
    should_optimizer_step,
    validate_teacher_completion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
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
    recipe = TrainingRecipe.from_path(args.recipe)
    manifest = ModelManifest.from_path(args.manifest)
    output_dir = args.output_dir or ROOT / "artifacts/training" / args.recipe.stem
    if manifest.model_id != recipe.base_model_id or manifest.tokenizer_id != recipe.base_model_id:
        raise SystemExit("recipe must use the pinned Qwen3-1.7B model/tokenizer")

    policy = TeacherValidationPolicy(
        min_chars=recipe.min_chars,
        max_chars=recipe.max_chars,
        min_japanese_ratio=recipe.min_japanese_ratio,
    )
    all_examples = load_teacher_examples(args.dataset)
    invalid = [
        item.scenario_id
        for item in all_examples
        if validate_teacher_completion(item.completion, policy)[1]
    ]
    if invalid:
        raise SystemExit(
            f"teacher dataset contains {len(invalid)} invalid examples; audit it first"
        )
    train_examples = [item for item in all_examples if item.split == "train"]
    if args.max_examples is not None:
        if args.max_examples < 1:
            raise SystemExit("--max-examples must be positive")
        train_examples = train_examples[: args.max_examples]
    if not train_examples:
        raise SystemExit("teacher dataset has no train examples")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "train_examples": len(train_examples),
                    "dataset_sha256": hash_file(args.dataset),
                    "recipe": recipe.as_dict(),
                    "model_revision": manifest.model_revision,
                    "tokenizer_revision": manifest.tokenizer_revision,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires CUDA")

    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    torch.cuda.manual_seed_all(recipe.seed)
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        manifest.model_id,
        revision=manifest.model_revision,
        trust_remote_code=False,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        use_safetensors=True,
        quantization_config=quant,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = peft.LoraConfig(
        r=recipe.lora_rank,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        target_modules=recipe.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
        use_dora=False,
    )
    model = peft.get_peft_model(model, lora_config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=recipe.learning_rate)
    device = torch.device("cuda:0")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = hash_file(args.dataset)
    history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(recipe.epochs):
        shuffled = list(train_examples)
        random.Random(recipe.seed + epoch).shuffle(shuffled)
        epoch_ce = 0.0
        epoch_kl = 0.0
        epoch_loss = 0.0
        epoch_count = 0
        accumulated_examples = 0
        for index, example in enumerate(shuffled):
            prompt_ids, completion_ids = build_completion_training_ids(
                tokenizer,
                example.prompt,
                example.completion,
                max_tokens=recipe.max_sequence_tokens,
            )
            start = len(prompt_ids)
            input_ids_list = prompt_ids + completion_ids
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            prediction_positions = torch.arange(
                start - 1,
                start + len(completion_ids) - 1,
                dtype=torch.long,
                device=device,
            )
            targets = torch.tensor(completion_ids, dtype=torch.long, device=device)

            model.train()
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                adapted = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=prediction_positions,
                ).logits[0]
                ce_loss = torch.nn.functional.cross_entropy(
                    adapted.float(),
                    targets,
                    label_smoothing=recipe.label_smoothing,
                )

                kl_loss = adapted.new_zeros((), dtype=torch.float32)
                kl_rows = select_kl_rows(
                    len(completion_ids),
                    recipe.kl_positions_per_example,
                    seed=recipe.seed,
                    example_key=f"{example.scenario_id}:{example.generation_seed}",
                )
                if recipe.kl_weight and kl_rows:
                    row_tensor = torch.tensor(kl_rows, dtype=torch.long, device=device)
                    absolute_positions = prediction_positions.index_select(0, row_tensor)
                    was_training = model.training
                    model.eval()
                    with model.disable_adapter(), torch.no_grad():
                        base = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            logits_to_keep=absolute_positions,
                        ).logits[0]
                    if was_training:
                        model.train()
                    adapted_subset = adapted.index_select(0, row_tensor)
                    kl_loss = forward_kl_nats(torch, base, adapted_subset).mean()
                loss = ce_loss + recipe.kl_weight * kl_loss

            (loss / recipe.gradient_accumulation_steps).backward()
            accumulated_examples += 1
            epoch_count += 1
            epoch_ce += float(ce_loss.detach().cpu())
            epoch_kl += float(kl_loss.detach().cpu())
            epoch_loss += float(loss.detach().cpu())

            should_step = should_optimizer_step(
                accumulated_examples,
                recipe.gradient_accumulation_steps,
                is_last_example=index + 1 == len(shuffled),
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated_examples = 0
                if optimizer_steps % 10 == 0:
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    print(
                        f"epoch={epoch + 1}/{recipe.epochs} step={optimizer_steps} "
                        f"loss={loss.detach().item():.4f} "
                        f"ce={ce_loss.detach().item():.4f} "
                        f"kl={kl_loss.detach().item():.4f} cuda={allocated:.2f}GiB",
                        flush=True,
                    )

        history.append(
            {
                "epoch": epoch + 1,
                "examples": epoch_count,
                "mean_loss": epoch_loss / epoch_count,
                "mean_ce": epoch_ce / epoch_count,
                "mean_kl": epoch_kl / epoch_count,
            }
        )
        checkpoint = output_dir / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint, safe_serialization=True)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_hash = hash_tree(adapter_dir)
    elapsed = time.perf_counter() - started
    metadata = {
        "schema_version": 1,
        "base_model_id": manifest.model_id,
        "base_model_revision": manifest.model_revision,
        "tokenizer_id": manifest.tokenizer_id,
        "tokenizer_revision": manifest.tokenizer_revision,
        "dataset": str(args.dataset),
        "recipe_path": str(args.recipe),
        "dataset_sha256": dataset_hash,
        "adapter_sha256": adapter_hash,
        "recipe": recipe.as_dict(),
        "train_examples": len(train_examples),
        "trainable_parameters": trainable_count,
        "total_parameters_seen_by_peft": total_count,
        "optimizer_steps": optimizer_steps,
        "seconds": elapsed,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    if not math.isfinite(sum(float(row["mean_loss"]) for row in history)):
        raise SystemExit("non-finite training loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
