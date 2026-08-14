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

from lsteg.model import ModelManifest, TransformersBackend
from lsteg.steg import SteganographyConfig, frequency_table_entropy
from lsteg.steg.frequencies import logits_to_frequency_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
DEFAULT_PROMPT = "架空のニュース記事：\n本日午後、東京都内の"  # noqa: RUF001
QWEN3_CONTROL_TOKENS = (151643, 151644, 151645)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
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
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
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
        excluded_token_ids=QWEN3_CONTROL_TOKENS,
    )
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(
        manifest,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    rng = random.Random(args.seed)

    raw_capacity_500: list[float] = []
    channel_capacity_500: list[float] = []
    token_per_char: list[float] = []
    transport_failures = 0

    for sample_index in range(args.samples):
        context = backend.tokenize(args.prompt)
        generated: list[int] = []
        raw_entropy = 0.0
        channel_entropy = 0.0

        for _ in range(args.max_new_tokens):
            logits = backend.next_logits(context)
            values = list(logits)
            raw_entropy += _full_entropy(values, config.temperature)
            top_ids, table = logits_to_frequency_table(
                values,
                top_k=config.top_k,
                total=config.frequency_total,
                temperature=config.temperature,
                excluded_token_ids=config.excluded_token_ids,
            )
            channel_entropy += frequency_table_entropy(table)

            draw = rng.randrange(table.total)
            symbol = table.symbol_for(draw)
            token_id = top_ids[symbol]
            context.append(token_id)
            generated.append(token_id)

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
    print(f"top_k                : {config.top_k}")
    print(f"mean token/character : {statistics.mean(token_per_char):.3f}")
    print(f"raw LM mean          : {statistics.mean(raw_capacity_500):.1f} bits")
    print(f"channel mean         : {statistics.mean(channel_capacity_500):.1f} bits")
    for label, fraction in (("p05", 0.05), ("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        print(
            f"channel {label:3s}          : "
            f"{_percentile(channel_capacity_500, fraction):.1f} bits"
        )
    print(f"transport failures   : {transport_failures}/{len(channel_capacity_500)}")
    return 0 if transport_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
