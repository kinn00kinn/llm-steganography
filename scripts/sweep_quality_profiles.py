#!/usr/bin/env python3
"""Run a small, pre-defined quality/capacity sweep on the fixed Qwen3-1.7B model.

Each profile invokes ``compare_quality_first_range.py`` end-to-end, including
compact encryption, stego generation, Unicode transport, exact decode, and a
normal same-base sample. Capacity failures are reported and the sweep continues.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "compare_quality_first_range.py"


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float
    semantic_plan: str


PROFILES = (
    Profile("baseline", 0.90, 128, 0.92, 0.0, "sparse"),
    Profile("guarded", 0.90, 128, 0.92, 0.0, "guarded"),
    Profile("guarded-presence-035", 0.90, 128, 0.92, 0.35, "guarded"),
    Profile("guarded-balanced", 0.85, 96, 0.90, 0.35, "guarded"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--secret", default="確認")
    parser.add_argument("--phase-min-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(profile.name for profile in PROFILES),
        help="run only the named profile; may be repeated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected = [p for p in PROFILES if not args.only or p.name in args.only]
    failures: list[tuple[str, int]] = []

    for index, profile in enumerate(selected, start=1):
        print("\n" + "=" * 88, flush=True)
        print(f"PROFILE {index}/{len(selected)}: {profile.name}", flush=True)
        print(
            f"T={profile.temperature}, top_k={profile.top_k}, top_p={profile.top_p}, "
            f"presence_penalty={profile.presence_penalty}, semantic={profile.semantic_plan}",
            flush=True,
        )
        print("=" * 88, flush=True)
        command = [
            sys.executable,
            str(COMPARE),
            "--key-file",
            str(args.key_file),
            "--secret",
            args.secret,
            "--temperature",
            str(profile.temperature),
            "--top-k",
            str(profile.top_k),
            "--top-p",
            str(profile.top_p),
            "--presence-penalty",
            str(profile.presence_penalty),
            "--semantic-plan",
            profile.semantic_plan,
            "--phase-min-tokens",
            str(args.phase_min_tokens),
            "--max-tokens",
            str(args.max_tokens),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode not in (0, 2):
            return completed.returncode
        if completed.returncode == 2:
            failures.append((profile.name, completed.returncode))

    print("\n" + "=" * 88, flush=True)
    print("SWEEP COMPLETE", flush=True)
    if failures:
        print(
            "capacity-limited profiles: " + ", ".join(name for name, _ in failures),
            flush=True,
        )
    else:
        print("all profiles completed exact round-trip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
