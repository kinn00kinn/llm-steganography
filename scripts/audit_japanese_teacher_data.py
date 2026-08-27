#!/usr/bin/env python3
"""Audit Japanese prose distillation JSONL before any GPU training is attempted."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lsteg.training import (  # noqa: E402
    TeacherValidationPolicy,
    TrainingRecipe,
    audit_teacher_examples,
    load_teacher_examples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    recipe = TrainingRecipe.from_path(args.recipe)
    policy = TeacherValidationPolicy(
        min_chars=recipe.min_chars,
        max_chars=recipe.max_chars,
        min_japanese_ratio=recipe.min_japanese_ratio,
    )
    examples = load_teacher_examples(args.dataset)
    numeric_report = audit_teacher_examples(examples, policy)
    teacher_models = Counter(item.teacher_model for item in examples)
    split_scenarios: dict[str, set[str]] = {"train": set(), "eval": set()}
    for item in examples:
        split_scenarios[item.split].add(item.scenario_id)
    overlap = split_scenarios["train"] & split_scenarios["eval"]
    report: dict[str, object] = dict(numeric_report)
    report["scenario_split_overlap"] = len(overlap)
    report["teacher_models"] = dict(sorted(teacher_models.items()))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if int(report["invalid_examples"]) != 0 or int(report["duplicate_completions"]) != 0 or overlap:
        return 2
    if int(report["min_completions_per_scenario"]) < recipe.teacher_completions_per_scenario:
        print(
            "one or more scenarios have fewer accepted completions than the recipe requires",
            file=sys.stderr,
        )
        return 2
    if int(report["eval_examples"]) == 0:
        print("dataset has no eval examples", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
