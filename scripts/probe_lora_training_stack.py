#!/usr/bin/env python3
"""Fail-fast probe for the Windows/CUDA QLoRA training stack and pinned model."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
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
    forward_kl_nats,
    select_kl_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument(
        "--backward-smoke",
        action="store_true",
        help="also inject LoRA and run one CE+KL backward pass",
    )
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
    recipe = TrainingRecipe.from_path(args.recipe)
    if manifest.model_id != recipe.base_model_id:
        raise SystemExit("training recipe/base manifest model mismatch")

    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    _module("bitsandbytes")
    _module("accelerate")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": capability,
        "cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "accelerate": importlib.metadata.version("accelerate"),
        "model_revision": manifest.model_revision,
        "tokenizer_revision": manifest.tokenizer_revision,
    }

    if args.backward_smoke:
        args.load_model = True

    if args.load_model:
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
        report.update(
            {
                "vocab_size": int(model.config.vocab_size),
                "tokenizer_size": len(tokenizer),
                "cuda_allocated_gib": torch.cuda.memory_allocated(0) / 1024**3,
                "cuda_reserved_gib": torch.cuda.memory_reserved(0) / 1024**3,
            }
        )

        if args.backward_smoke:
            model.config.use_cache = False
            model = peft.prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            model = peft.get_peft_model(
                model,
                peft.LoraConfig(
                    r=recipe.lora_rank,
                    lora_alpha=recipe.lora_alpha,
                    lora_dropout=recipe.lora_dropout,
                    target_modules=recipe.target_modules,
                    task_type="CAUSAL_LM",
                    bias="none",
                    use_dora=False,
                ),
            )
            prompt_ids, completion_ids = build_completion_training_ids(
                tokenizer,
                "正午前の研究室で短い日常の出来事を書く。",
                "正午が近づき、私は資料を保存してから友人と学食へ向かった。",
                max_tokens=recipe.max_sequence_tokens,
            )
            device = torch.device("cuda:0")
            start = len(prompt_ids)
            input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
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
                    adapted.float(), targets, label_smoothing=recipe.label_smoothing
                )
                rows = select_kl_rows(
                    len(completion_ids),
                    min(4, recipe.kl_positions_per_example),
                    seed=recipe.seed,
                    example_key="probe",
                )
                kl_loss = adapted.new_zeros((), dtype=torch.float32)
                if rows:
                    row_tensor = torch.tensor(rows, dtype=torch.long, device=device)
                    absolute_positions = prediction_positions.index_select(0, row_tensor)
                    model.eval()
                    with model.disable_adapter(), torch.no_grad():
                        base = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            logits_to_keep=absolute_positions,
                        ).logits[0]
                    model.train()
                    adapted_subset = adapted.index_select(0, row_tensor)
                    kl_loss = forward_kl_nats(torch, base, adapted_subset).mean()
                loss = ce_loss + recipe.kl_weight * kl_loss
            loss.backward()
            finite_gradients = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if not finite_gradients:
                raise SystemExit("LoRA backward smoke produced non-finite gradients")
            report.update(
                {
                    "lora_trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "backward_smoke_ce": float(ce_loss.detach().cpu()),
                    "backward_smoke_kl": float(kl_loss.detach().cpu()),
                    "backward_smoke_loss": float(loss.detach().cpu()),
                    "backward_smoke_finite_gradients": True,
                    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
                }
            )

        del model
        torch.cuda.empty_cache()

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
