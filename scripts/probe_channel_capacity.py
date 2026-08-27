"""Measure capacity of the *implemented* integer/top-k stego channel.

Unlike ``probe_entropy.py``, this script measures the exact quantized frequency
policy used by ``lsteg.steg``.  It therefore includes temperature, top-k
truncation, integer-frequency quantization, excluded control tokens, and a
Unicode tokenizer-transport check.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path

from lsteg.model import (
    ModelManifest,
    RankedIncrementalLogitsSession,
    TransformersBackend,
    start_incremental_logits,
)
from lsteg.steg import (
    SteganographyConfig,
    filter_no_repeat_ngram_candidates,
    filter_transport_safe_candidates,
    frequency_table_entropy,
    university_lunch_cover_plan,
)
from lsteg.steg.frequencies import (
    logits_to_frequency_table,
    ranked_logits_to_frequency_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
DEFAULT_PROMPT = (
    "大学生活の中で起きた小さな出来事について、自然な一人称の日記を書いてください。"
    "本文だけを書き、同じ表現の繰り返し、読者への指示、文章作成についてのメタ説明、"
    "見出しや箇条書きは避けてください。内容は架空で構いません。"
)
DEFAULT_SYSTEM_PROMPT = (
    "あなたは自然な現代日本語を書く編集者です。具体的な出来事を一貫した視点で描き、"
    "簡潔で読みやすい文章にしてください。"
)
QWEN3_CONTROL_TOKENS = (151643, 151644, 151645, 151668)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus cutoff applied within the top-k candidate pool",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="block exact repeated token n-grams in the active channel (0 disables)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="use --prompt as a raw causal prefix instead of Qwen chat-template input",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--semantic-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the fixed five-stage university diary scaffold",
    )
    parser.add_argument("--phase-min-tokens", type=int, default=40)
    parser.add_argument(
        "--transport-safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "restrict each active alphabet to canonical Unicode transport extensions "
            "(default: enabled)"
        ),
    )
    return parser


def _full_entropy(logits: list[float], temperature: float) -> float:
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    entropy = 0.0
    for weight in weights:
        probability = weight / total
        if probability > 0.0:
            entropy -= probability * math.log2(probability)
    return entropy


def _percentile(values: list[float], fraction: float) -> float:
    """Empirical nearest-rank percentile (p05 of 20 samples includes the minimum)."""
    ordered = sorted(values)
    index = max(0, min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1))
    return ordered[index]


def main() -> int:
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    config = SteganographyConfig(
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        excluded_token_ids=QWEN3_CONTROL_TOKENS,
        enforce_transport_invariance=args.transport_safe,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(
        manifest,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    rng = random.Random(args.seed)
    semantic_plan = (
        university_lunch_cover_plan(min_tokens_per_phase=args.phase_min_tokens)
        if args.semantic_anchors and not args.raw_prompt
        else None
    )
    prompt = (
        semantic_plan.initial_prompt(backend)
        if semantic_plan is not None
        else (
            args.prompt
            if args.raw_prompt
            else backend.render_chat_prompt(
                args.prompt,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                enable_thinking=False,
            )
        )
    )

    raw_capacity_500: list[float] = []
    channel_capacity_500: list[float] = []
    token_per_char: list[float] = []
    transport_failures = 0
    safe_candidate_counts: list[int] = []

    for sample_index in range(args.samples):
        context = backend.tokenize(prompt)
        logit_session = start_incremental_logits(backend, context)
        generated: list[int] = []
        semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
        raw_entropy = 0.0
        channel_entropy = 0.0

        for token_index in range(args.max_new_tokens):
            if semantic_state is not None:
                replacement = semantic_state.maybe_advance(backend, generated, position=token_index)
                if replacement is not None:
                    context = backend.tokenize(replacement)
                    logit_session = start_incremental_logits(backend, context)
            if isinstance(logit_session, RankedIncrementalLogitsSession):
                raw_entropy += logit_session.entropy(temperature=config.temperature)
                ranked = logit_session.top_logits(
                    config.top_k,
                    excluded_token_ids=config.excluded_token_ids,
                )
                top_ids, table = ranked_logits_to_frequency_table(
                    ranked,
                    total=config.frequency_total,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
            else:
                logits = logit_session.next_logits()
                values = list(logits)
                raw_entropy += _full_entropy(values, config.temperature)
                top_ids, table = logits_to_frequency_table(
                    values,
                    top_k=config.top_k,
                    total=config.frequency_total,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    excluded_token_ids=config.excluded_token_ids,
                )
            if config.no_repeat_ngram_size:
                top_ids, table = filter_no_repeat_ngram_candidates(
                    generated,
                    top_ids,
                    table,
                    ngram_size=config.no_repeat_ngram_size,
                )
            if config.enforce_transport_invariance:
                top_ids, table = filter_transport_safe_candidates(
                    backend,
                    generated,
                    top_ids,
                    table,
                )
            safe_candidate_counts.append(len(top_ids))
            channel_entropy += frequency_table_entropy(table)

            draw = rng.randrange(table.total)
            symbol = table.symbol_for(draw)
            token_id = top_ids[symbol]
            context.append(token_id)
            generated.append(token_id)
            if token_index + 1 < args.max_new_tokens:
                logit_session.append(token_id)

        text = backend.detokenize(generated)
        chars = len(text)
        if chars == 0:
            continue

        transport_ok = backend.tokenize(text) == generated
        if not transport_ok:
            transport_failures += 1

        ratio = len(generated) / chars
        token_per_char.append(ratio)
        raw_capacity_500.append(raw_entropy / chars * 500.0)
        channel_capacity_500.append(channel_entropy / chars * 500.0)
        print(
            f"sample {sample_index + 1:3d}: chars={chars:4d} tokens={len(generated):3d} "
            f"raw500={raw_capacity_500[-1]:7.1f} "
            f"channel500={channel_capacity_500[-1]:7.1f} "
            f"transport={'OK' if transport_ok else 'FAIL'}",
            flush=True,
        )

    if not channel_capacity_500:
        print("No non-empty samples generated.")
        return 1

    print("\n--- Implemented channel capacity for 500 Unicode characters ---")
    print(f"samples              : {len(channel_capacity_500)}")
    print(f"temperature          : {config.temperature}")
    print(f"prompt mode          : {'raw' if args.raw_prompt else 'qwen-chat/non-thinking'}")
    print(f"top_k                : {config.top_k}")
    print(f"top_p                : {config.top_p}")
    print(f"no-repeat ngram      : {config.no_repeat_ngram_size}")
    print(f"transport-safe       : {config.enforce_transport_invariance}")
    print(f"semantic anchors     : {semantic_plan is not None}")
    if safe_candidate_counts:
        print(
            f"mean active candidates: {statistics.mean(safe_candidate_counts):.1f} / {config.top_k}"
        )
    print(f"mean token/character : {statistics.mean(token_per_char):.3f}")
    print(f"raw LM mean          : {statistics.mean(raw_capacity_500):.1f} bits")
    print(f"channel mean         : {statistics.mean(channel_capacity_500):.1f} bits")
    for label, fraction in (("p05", 0.05), ("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        print(
            f"channel {label:3s}          : {_percentile(channel_capacity_500, fraction):.1f} bits"
        )
    print(f"transport failures   : {transport_failures}/{len(channel_capacity_500)}")
    return 0 if transport_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
