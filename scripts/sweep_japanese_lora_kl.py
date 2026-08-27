#!/usr/bin/env python3
"""Train/evaluate a one-variable KL-weight sweep for Japanese-prose QLoRA.

The sweep intentionally keeps rank, learning rate, dataset, label smoothing,
and KL row count fixed.  Only ``kl_weight`` changes, so the resulting adapter
comparisons can identify whether stronger base-distribution anchoring fixes the
entropy expansion seen in the v2 smoke adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "kl020": ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json",
    "kl050": ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v3-kl050.json",
    "kl100": ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v3-kl100.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=list(PROFILES),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config/models/qwen3-1.7b-debug.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/model-cache")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/training/japanese-prose-kl-sweep",
    )
    parser.add_argument("--generation-samples", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--positions-per-example", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(command: list[str], *, accepted_codes: frozenset[int] = frozenset({0})) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in accepted_codes:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed.returncode


def _load_report(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read evaluation report: {path}") from error
    if not isinstance(raw, dict):
        raise SystemExit(f"evaluation report root is not an object: {path}")
    return raw


def _positive_float(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SystemExit(f"evaluation report has invalid {key!r}")
    value = float(value)
    if value <= 0.0:
        raise SystemExit(f"evaluation report has non-positive {key!r}")
    return value


def _summary_row(profile: str, recipe_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    full_ratio = _positive_float(report, "full_entropy_ratio")
    channel_ratio = _positive_float(report, "channel_entropy_ratio")
    base_nll = _positive_float(report, "base_nll")
    adapter_nll = _positive_float(report, "adapter_nll")
    return {
        "profile": profile,
        "recipe": str(recipe_path),
        "kl_weight": report.get("recipe", {}).get("kl_weight"),
        "adapter_nll": adapter_nll,
        "nll_delta": adapter_nll - base_nll,
        "full_entropy_ratio": full_ratio,
        "channel_entropy_ratio": channel_ratio,
        "distribution_distance": abs(math.log(full_ratio)) + abs(math.log(channel_ratio)),
        "base_to_adapter_kl_nats": report.get("base_to_adapter_kl_nats"),
        "eval_scenarios": report.get("eval_scenarios"),
        "gate_sufficient_eval_scenarios": report.get("gate_sufficient_eval_scenarios"),
        "gate_full_entropy_bounded": report.get("gate_full_entropy_bounded"),
        "gate_channel_entropy_bounded": report.get("gate_channel_entropy_bounded"),
        "gate_nll_improved": report.get("gate_nll_improved"),
        "gate_kl_bounded": report.get("gate_kl_bounded"),
        "gate_pass": report.get("gate_pass"),
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== KL SWEEP SUMMARY ===")
    header = (
        f"{'profile':<8} {'kl':>5} {'nllΔ':>9} {'Hfull':>8} "
        f"{'Hchan':>8} {'KL':>8} {'dist':>8} {'gate':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        kl = row["kl_weight"]
        nll_delta = float(row["nll_delta"])
        full_ratio = float(row["full_entropy_ratio"])
        channel_ratio = float(row["channel_entropy_ratio"])
        base_kl = float(row["base_to_adapter_kl_nats"])
        distance = float(row["distribution_distance"])
        gate = "PASS" if row["gate_pass"] else "FAIL"
        print(
            f"{row['profile']:<8} {kl:>5.2f} {nll_delta:>+9.4f} "
            f"{full_ratio:>8.3f} {channel_ratio:>8.3f} "
            f"{base_kl:>8.3f} {distance:>8.3f} {gate:>6}"
        )
    print(
        "\nInterpretation: prefer entropy ratios near 1.0 while retaining a negative "
        "NLL delta. Do not treat a smoke dataset with too few eval scenarios as a "
        "release-quality winner even when its distribution metrics look better."
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.generation_samples < 0:
        raise SystemExit("--generation-samples must be non-negative")
    if args.positions_per_example < 1:
        raise SystemExit("--positions-per-example must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        raise SystemExit("--max-examples must be positive")

    plan = {
        "dataset": str(args.dataset),
        "output_root": str(args.output_root),
        "profiles": [
            {"name": profile, "recipe": str(PROFILES[profile])} for profile in args.profiles
        ],
        "skip_training": args.skip_training,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for profile in args.profiles:
        recipe_path = PROFILES[profile]
        output_dir = args.output_root / profile
        adapter_dir = output_dir / "adapter"
        report_path = output_dir / "evaluation.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 88)
        print(f"PROFILE: {profile} ({recipe_path.name})")
        print("=" * 88)

        if not args.skip_training:
            train_command = [
                sys.executable,
                str(ROOT / "scripts/train_japanese_lora.py"),
                str(args.dataset),
                "--manifest",
                str(args.manifest),
                "--recipe",
                str(recipe_path),
                "--cache-dir",
                str(args.cache_dir),
                "--output-dir",
                str(output_dir),
            ]
            if args.max_examples is not None:
                train_command += ["--max-examples", str(args.max_examples)]
            _run(train_command)
        elif not adapter_dir.is_dir():
            raise SystemExit(f"--skip-training requires existing adapter: {adapter_dir}")

        evaluate_command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_japanese_lora.py"),
            str(args.dataset),
            str(adapter_dir),
            "--manifest",
            str(args.manifest),
            "--recipe",
            str(recipe_path),
            "--cache-dir",
            str(args.cache_dir),
            "--generation-samples",
            str(args.generation_samples),
            "--positions-per-example",
            str(args.positions_per_example),
            "--output-report",
            str(report_path),
        ]
        # evaluate_japanese_lora.py intentionally returns 2 when quality gates fail.
        _run(evaluate_command, accepted_codes=frozenset({0, 2}))
        report = _load_report(report_path)
        rows.append(_summary_row(profile, recipe_path, report))

    summary = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "profiles": rows,
    }
    summary_path = args.output_root / "kl-sweep-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_summary(rows)
    print(f"summary report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
