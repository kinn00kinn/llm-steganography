"""Probe the pinned Qwen2.5-7B GPTQ model before steganographic use."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from lsteg.model import ModelManifest, TransformersBackend, start_incremental_logits
from lsteg.model.interface import RankedIncrementalLogitsSession
from lsteg.steg.frequencies import (
    frequency_table_entropy,
    ranked_logits_to_frequency_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen2.5-7b-gptq-int4-quality.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
DEFAULT_PROMPT = "自然で簡潔な現代日本語で、大学の研究室での平凡な昼休みについて書いてください。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--max-peak-vram-gib", type=float, default=7.5)
    parser.add_argument("--max-median-ms", type=float, default=150.0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _gib(value: int) -> float:
    return value / (1024**3)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    manifest = ModelManifest.from_path(args.manifest)
    started = time.perf_counter()
    backend = TransformersBackend.load(
        manifest,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    load_seconds = time.perf_counter() - started
    torch = backend._torch
    torch.cuda.synchronize(manifest.device)
    resident_gib = _gib(torch.cuda.memory_allocated(manifest.device))
    torch.cuda.reset_peak_memory_stats(manifest.device)

    prompt = backend.render_chat_prompt(args.prompt, enable_thinking=False)
    prompt_ids = backend.tokenize(prompt)

    prefill_started = time.perf_counter()
    first = start_incremental_logits(backend, prompt_ids)
    torch.cuda.synchronize(manifest.device)
    prefill_ms = (time.perf_counter() - prefill_started) * 1000.0
    second = start_incremental_logits(backend, prompt_ids)
    if not isinstance(first, RankedIncrementalLogitsSession):
        raise RuntimeError("Transformers backend did not expose ranked incremental logits")
    if not isinstance(second, RankedIncrementalLogitsSession):
        raise RuntimeError("second session did not expose ranked incremental logits")

    excluded = backend.special_token_ids
    first_ranked = first.top_logits(args.top_k, excluded_token_ids=excluded)
    second_ranked = second.top_logits(args.top_k, excluded_token_ids=excluded)
    deterministic_prefill = first_ranked == second_ranked

    _, initial_table = ranked_logits_to_frequency_table(
        first_ranked,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    initial_entropy = frequency_table_entropy(initial_table)

    session = first
    generated: list[int] = []
    latencies_ms: list[float] = []
    entropies: list[float] = []
    for _ in range(args.steps):
        ranked = session.top_logits(args.top_k, excluded_token_ids=excluded)
        _, table = ranked_logits_to_frequency_table(
            ranked,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        entropies.append(frequency_table_entropy(table))
        token_id = ranked.token_ids[0]
        started = time.perf_counter()
        session.append(token_id)
        torch.cuda.synchronize(manifest.device)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        generated.append(token_id)

    median_ms = statistics.median(latencies_ms)
    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))
    p95_ms = sorted_latencies[p95_index]
    peak_gib = _gib(torch.cuda.max_memory_allocated(manifest.device))
    sample_text = backend.detokenize(generated)

    gate_vram = peak_gib <= args.max_peak_vram_gib
    gate_latency = median_ms <= args.max_median_ms
    gate_pass = deterministic_prefill and gate_vram and gate_latency
    result = {
        "schema_version": 1,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "gptqmodel_version": version("gptqmodel"),
        "runtime": backend.runtime.as_dict(),
        "vocabulary_size": backend.vocabulary_size,
        "special_token_ids": list(excluded),
        "prompt_tokens": len(prompt_ids),
        "load_seconds": load_seconds,
        "resident_allocated_gib": resident_gib,
        "generation_peak_allocated_gib": peak_gib,
        "prefill_ms": prefill_ms,
        "incremental_median_ms": median_ms,
        "incremental_p95_ms": p95_ms,
        "incremental_tokens_per_second": 1000.0 / median_ms,
        "mean_channel_entropy_bits": statistics.mean(entropies),
        "initial_channel_entropy_bits": initial_entropy,
        "deterministic_prefill": deterministic_prefill,
        "gate_vram": gate_vram,
        "gate_latency": gate_latency,
        "gate_pass": gate_pass,
        "sample_text": sample_text,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
