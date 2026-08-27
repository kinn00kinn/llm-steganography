#!/usr/bin/env python3
"""Run the local verification gate before a patch is treated as test-ready.

The default gate is deterministic and does not require the model artifact.
``--full`` adds the complete repository suite. ``--model`` enables the pinned
Qwen/CUDA regression tests. ``--ollama`` runs the local direct-Qwen2.5 determinism/cache probe.
"""

from __future__ import annotations

import argparse
import compileall
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run the complete pytest suite",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="also run pinned local-Qwen/CUDA regression tests",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="also run the local Ollama qwen2.5 direct-backend probe",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="also run the QLoRA dependency/model/backward smoke probe",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print("[1/4] compile source", flush=True)
    if not compileall.compile_dir(ROOT / "src", quiet=1):
        return 1
    if not compileall.compile_dir(ROOT / "scripts", quiet=1):
        return 1

    print("[2/4] deterministic cross-feature gate", flush=True)
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/steg/test_release_gate.py",
            "tests/steg/test_semantic.py",
            "tests/steg/test_tail.py",
            "tests/steg/test_candidate_filter_order.py",
            "tests/steg/test_engine.py",
            "tests/steg/test_frequencies.py",
            "tests/steg/test_ollama_direct.py",
            "tests/payload/test_compact_crypto.py",
            "tests/training",
            "-q",
        ]
    )

    print("[3/4] model-boundary unit tests", flush=True)
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/model/test_incremental.py",
            "tests/model/test_transformers_backend.py",
            "tests/model/test_ollama_api.py",
            "-q",
        ]
    )

    if args.full:
        print("[4/4] full repository suite", flush=True)
        _run([sys.executable, "-m", "pytest", "-q"])
    else:
        print("[4/4] full suite skipped (pass --full to enable)", flush=True)

    if args.model:
        env = os.environ.copy()
        env["LSTEG_RUN_MODEL_TESTS"] = "1"
        print("[extra] pinned Qwen/CUDA integration tests", flush=True)
        _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/model/test_transformers_backend.py",
                "tests/model/test_ollama_api.py",
                "tests/steg/test_tokenizer_transport.py",
                "-q",
            ],
            env=env,
        )

    if args.ollama:
        print("[extra] Ollama qwen2.5 direct-backend probe", flush=True)
        _run(
            [
                sys.executable,
                "scripts/probe_ollama_direct_backend.py",
                "--repeat",
                "3",
                "--steps",
                "8",
            ]
        )

    if args.training:
        print("[extra] QLoRA training-stack backward smoke", flush=True)
        _run(
            [
                sys.executable,
                "scripts/probe_lora_training_stack.py",
                "--backward-smoke",
            ]
        )

    print("release-candidate verification: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
