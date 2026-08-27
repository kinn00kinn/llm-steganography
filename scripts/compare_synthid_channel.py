"""Compare a SynthID-inspired candidate-pool stego sample with its base sampler.

The experimental channel draws multiple iid candidates from the exact active
Qwen distribution.  Duplicate draws become integer counts in a tiny empirical
distribution, and the existing Range Coder embeds the payload through that
distribution.  Averaged over keyed candidate pools, those empirical counts
recover the base model distribution while retaining collision entropy instead
of discarding collided positions as erasures.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from pathlib import Path

from lsteg.model import ModelManifest, TransformersBackend, start_incremental_logits
from lsteg.payload import (
    decode_secure_text_payload,
    derive_keys,
    encode_secure_text_payload,
    read_master_key,
)
from lsteg.steg import (
    SteganographyConfig,
    append_sentence_tail,
    extract_bytes_tournament,
    hide_bytes_tournament,
    university_lunch_micro_cover_plan,
)
from lsteg.steg.engine import _position_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
QWEN3_CONTROL_TOKENS = (151643, 151644, 151645, 151668)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--secret", default="確認")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    # Qwen3's published non-thinking recommendation is T=.7/P=.8/K=20.  Keep
    # those as this quality-first experiment's defaults instead of widening the
    # distribution to chase capacity.
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--pool-bits",
        "--bits-per-token",
        dest="pool_bits",
        type=int,
        default=4,
        help="log2 iid candidate-pool size; default 4 means 16 draws",
    )
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--phase-min-tokens", type=int, default=8)
    parser.add_argument("--tail-tokens", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="off by default so the base sampling distribution is not hard-filtered",
    )
    return parser


def _ordinary_sample(
    backend: TransformersBackend,
    prompt: str,
    token_count: int,
    config: SteganographyConfig,
    *,
    seed: int,
    semantic_plan,
) -> list[int]:
    rng = random.Random(seed)
    context = backend.tokenize(prompt)
    session = start_incremental_logits(backend, context)
    generated: list[int] = []
    semantic_state = semantic_plan.new_state()

    for position in range(token_count):
        replacement = semantic_state.maybe_advance(backend, generated, position=position)
        if replacement is not None:
            context = backend.tokenize(replacement)
            session = start_incremental_logits(backend, context)
        top_ids, table = _position_table(
            backend,
            session,
            context,
            position=position,
            stego_key=b"\x00" * 32,
            config=config,
            cover_token_ids=generated,
        )
        token_id = top_ids[table.symbol_for(rng.randrange(table.total))]
        context.append(token_id)
        generated.append(token_id)
        if position + 1 < token_count:
            session.append(token_id)
    return generated


def main() -> int:
    args = build_parser().parse_args()
    master_key = read_master_key(args.key_file)
    stego_key = derive_keys(master_key).steganography
    backend = TransformersBackend.load(
        ModelManifest.from_path(args.manifest),
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    plan = university_lunch_micro_cover_plan(min_tokens_per_phase=args.phase_min_tokens)
    prompt = plan.initial_prompt(backend)
    config = SteganographyConfig(
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        excluded_token_ids=QWEN3_CONTROL_TOKENS,
        enforce_transport_invariance=True,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    encoded = encode_secure_text_payload(args.secret, master_key)

    started = time.perf_counter()
    payload_tokens = hide_bytes_tournament(
        backend,
        prompt,
        encoded.frame,
        stego_key=stego_key,
        config=config,
        semantic_plan=plan,
        bits_per_token=args.pool_bits,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
    )
    stego = append_sentence_tail(
        backend,
        prompt,
        payload_tokens,
        seed_key=stego_key,
        config=config,
        semantic_plan=plan,
        max_tail_tokens=args.tail_tokens,
    )
    stego_seconds = time.perf_counter() - started

    stego_text = backend.detokenize(stego)
    received = backend.tokenize(stego_text)
    if received != stego:
        raise RuntimeError("STEG text changed token IDs across Unicode transport")
    started = time.perf_counter()
    recovered_frame = extract_bytes_tournament(
        backend,
        prompt,
        received,
        len(encoded.frame),
        stego_key=stego_key,
        config=config,
        semantic_plan=plan,
        bits_per_token=args.pool_bits,
        context_window=args.context_window,
    )
    recovered_secret = decode_secure_text_payload(recovered_frame, master_key)
    decode_seconds = time.perf_counter() - started
    if recovered_secret != args.secret:
        raise RuntimeError("candidate-pool round-trip recovered a different secret")

    started = time.perf_counter()
    normal_body = _ordinary_sample(
        backend,
        prompt,
        len(payload_tokens),
        config,
        seed=args.seed,
        semantic_plan=plan,
    )
    normal_seed = hashlib.sha256(f"normal-tail:{args.seed}".encode()).digest()
    normal = append_sentence_tail(
        backend,
        prompt,
        normal_body,
        seed_key=normal_seed,
        config=config,
        semantic_plan=plan,
        max_tail_tokens=args.tail_tokens,
    )
    normal_seconds = time.perf_counter() - started

    payload_bits = len(encoded.frame) * 8
    effective = payload_bits / len(payload_tokens)
    print("channel: SynthID-inspired empirical candidate pool (experimental)")
    print(
        f"base distribution: T={config.temperature}, top_k={config.top_k}, "
        f"top_p={config.top_p}, no_repeat_ngram={config.no_repeat_ngram_size}"
    )
    print(
        f"candidate pool: {1 << args.pool_bits} iid draws / empirical-count Range Coding / "
        f"context window={args.context_window}"
    )
    print(f"payload: {payload_bits} bits")
    print(
        f"stego : {len(stego_text)} chars / {len(stego)} tokens "
        f"({len(payload_tokens)} payload + {len(stego) - len(payload_tokens)} tail; "
        f"effective {effective:.3f} bit/payload-token; {stego_seconds:.2f} s)"
    )
    normal_text = backend.detokenize(normal)
    print(
        f"normal: {len(normal_text)} chars / {len(normal)} tokens "
        f"({len(normal_body)} body + {len(normal) - len(normal_body)} tail; "
        f"{normal_seconds:.2f} s)"
    )
    print(f"round-trip: PASS ({decode_seconds:.2f} s decode; exact secret match)")
    print("\n--- TOURNAMENT STEGO ---")
    print(stego_text)
    print("\n--- NORMAL SAME-BASE SAMPLE ---")
    print(normal_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
