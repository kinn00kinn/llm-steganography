#!/usr/bin/env python3
"""Generate filtered Japanese prose distillation data from a local Ollama teacher."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lsteg.training import (  # noqa: E402
    TEACHER_DATA_SCHEMA_VERSION,
    TeacherExample,
    TeacherValidationPolicy,
    TrainingRecipe,
    build_scenarios,
    hash_file,
    parse_model_identity,
    validate_teacher_completion,
)
from lsteg.training.prompts import JAPANESE_PROSE_SYSTEM_PROMPT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--completions-per-scenario", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--attempt-multiplier", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/training/japanese-prose-v1.jsonl",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=ROOT / "data/training/japanese-prose-v1.rejected.jsonl",
    )
    return parser


def _get_model_identity(url: str, model: str) -> dict[str, object]:
    request = urllib.request.Request(f"{url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot query Ollama model list: {error}") from error
    try:
        return parse_model_identity(raw, model)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _get_version(url: str) -> str:
    request = urllib.request.Request(f"{url.rstrip('/')}/api/version", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot query Ollama version: {error}") from error
    version = raw.get("version") if isinstance(raw, dict) else None
    return version if isinstance(version, str) else "unknown"


def _post_chat(
    url: str,
    *,
    model: str,
    prompt: str,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JAPANESE_PROSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": 700,
        },
        "keep_alive": "30m",
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ollama chat request failed: {error}") from error
    try:
        content = raw["message"]["content"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Ollama response did not contain message.content") from error
    if not isinstance(content, str):
        raise RuntimeError("Ollama message.content was not text")
    return content


def main() -> int:
    args = build_parser().parse_args()
    recipe = TrainingRecipe.from_path(args.recipe)
    model = args.model or recipe.teacher_model
    scenario_count = args.scenarios or recipe.teacher_scenarios
    completions_per = args.completions_per_scenario or recipe.teacher_completions_per_scenario
    if args.attempt_multiplier < 1:
        raise SystemExit("--attempt-multiplier must be >= 1")

    policy = TeacherValidationPolicy(
        min_chars=recipe.min_chars,
        max_chars=recipe.max_chars,
        min_japanese_ratio=recipe.min_japanese_ratio,
    )
    scenarios = build_scenarios(scenario_count, seed=recipe.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    ollama_version = _get_version(args.ollama_url)
    teacher_identity = _get_model_identity(args.ollama_url, model)
    started = time.perf_counter()
    with (
        args.output.open("w", encoding="utf-8", newline="\n") as output_handle,
        args.rejected_output.open("w", encoding="utf-8", newline="\n") as rejected_handle,
    ):
        for scenario_index, scenario in enumerate(scenarios):
            scenario_accepted = 0
            scenario_completions: set[str] = set()
            max_attempts = completions_per * args.attempt_multiplier
            for attempt in range(max_attempts):
                if scenario_accepted >= completions_per:
                    break
                generation_seed = recipe.seed + scenario_index * 10_000 + attempt
                raw = _post_chat(
                    args.ollama_url,
                    model=model,
                    prompt=scenario.prompt,
                    seed=generation_seed,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                completion, reasons = validate_teacher_completion(raw, policy)
                if completion in scenario_completions:
                    reasons = tuple(sorted({*reasons, "duplicate-completion"}))
                if reasons:
                    rejected += 1
                    rejected_handle.write(
                        json.dumps(
                            {
                                "scenario_id": scenario.scenario_id,
                                "generation_seed": generation_seed,
                                "reasons": list(reasons),
                                "completion": completion,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                example = TeacherExample(
                    schema_version=TEACHER_DATA_SCHEMA_VERSION,
                    scenario_id=scenario.scenario_id,
                    category=scenario.category,
                    split=scenario.split,
                    prompt=scenario.prompt,
                    completion=completion,
                    teacher_model=model,
                    generation_seed=generation_seed,
                )
                output_handle.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
                output_handle.flush()
                scenario_completions.add(completion)
                scenario_accepted += 1
                accepted += 1
            if scenario_accepted < completions_per:
                print(
                    (
                        f"WARNING: {scenario.scenario_id} accepted "
                        f"{scenario_accepted}/{completions_per}"
                    ),
                    file=sys.stderr,
                )
            print(
                f"[{scenario_index + 1}/{len(scenarios)}] accepted={accepted} rejected={rejected}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    metadata = {
        "schema_version": 1,
        "teacher_model": model,
        "teacher_identity": teacher_identity,
        "ollama_version": ollama_version,
        "scenario_count": scenario_count,
        "requested_completions_per_scenario": completions_per,
        "accepted": accepted,
        "rejected": rejected,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "dataset_sha256": hash_file(args.output),
        "seconds": elapsed,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"teacher-data complete: accepted={accepted}, rejected={rejected}, "
        f"seconds={elapsed:.1f}, output={args.output}"
    )
    print(f"metadata: {metadata_path}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
