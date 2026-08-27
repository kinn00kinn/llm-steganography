"""Compare stego cover text against ordinary sampling from the same channel.

The baseline uses the exact same prompt, temperature, top-k/top-p truncation,
integer frequency quantization and Unicode transport filter as the stego path.
If both outputs are similarly awkward, the bottleneck is the cover model/prompt
rather than the arithmetic steganography mapping.
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
    extract_bytes,
    hide_bytes,
    university_lunch_cover_plan,
)
from lsteg.steg.engine import _position_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
QWEN3_CONTROL_TOKENS = (151643, 151644, 151645, 151668)
DEFAULT_SECRET = "確認"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--secret", default=DEFAULT_SECRET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="block exact repeated token n-grams in the active channel (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--verify-round-trip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode the STEGO sample and require exact secret recovery",
    )
    parser.add_argument(
        "--semantic-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--phase-min-tokens", type=int, default=40)
    return parser


def _ordinary_sample(
    backend: TransformersBackend,
    prompt: str,
    token_count: int,
    config: SteganographyConfig,
    *,
    seed: int,
    semantic_plan=None,
) -> list[int]:
    rng = random.Random(seed)
    context = backend.tokenize(prompt)
    session = start_incremental_logits(backend, context)
    generated: list[int] = []
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None

    for position in range(token_count):
        if semantic_state is not None:
            replacement = semantic_state.maybe_advance(backend, generated, position=position)
            if replacement is not None:
                context = backend.tokenize(replacement)
                session = start_incremental_logits(backend, context)
        # Reuse the stego engine's exact candidate construction so NORMAL and
        # STEGO share transport fallbacks and naturalness filters.  The keyed
        # permutation changes interval order only; uniform ordinary sampling
        # still has exactly the same token probabilities.
        top_ids, table = _position_table(
            backend,
            session,
            context,
            position=position,
            stego_key=b"\x00" * 32,
            config=config,
            cover_token_ids=generated,
        )
        symbol = table.symbol_for(rng.randrange(table.total))
        token_id = top_ids[symbol]
        context.append(token_id)
        generated.append(token_id)
        if position + 1 < token_count:
            session.append(token_id)
    return generated


def main() -> int:
    args = build_parser().parse_args()
    master_key = read_master_key(args.key_file)
    stego_key = derive_keys(master_key).steganography
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(manifest, cache_dir=args.cache_dir, local_files_only=True)
    semantic_plan = (
        university_lunch_cover_plan(min_tokens_per_phase=args.phase_min_tokens)
        if args.semantic_anchors
        else None
    )
    if semantic_plan is not None:
        prompt = semantic_plan.initial_prompt(backend)
    else:
        prompt = backend.render_chat_prompt(
            "現在大学生の語り手が、平日の午後に大学キャンパスで体験した小さな出来事を、"
            "自然な一人称の日記として書いてください。本文だけを書いてください。",
            system_prompt="あなたは自然な現代日本語の短い日記を書く編集者です。",
            enable_thinking=False,
        )
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
    stego_payload = hide_bytes(
        backend,
        prompt,
        encoded.frame,
        stego_key=stego_key,
        config=config,
        semantic_plan=semantic_plan,
        max_tokens=2048,
    )
    stego = append_sentence_tail(
        backend,
        prompt,
        stego_payload,
        seed_key=stego_key,
        config=config,
        semantic_plan=semantic_plan,
        max_tail_tokens=args.tail_tokens,
    )
    stego_seconds = time.perf_counter() - started

    round_trip_seconds: float | None = None
    if args.verify_round_trip:
        # Simulate the actual Unicode transport boundary before decoding.
        stego_text = backend.detokenize(stego)
        received_ids = backend.tokenize(stego_text)
        if received_ids != stego:
            raise RuntimeError("STEG sample changed token IDs across Unicode transport")
        started = time.perf_counter()
        recovered_frame = extract_bytes(
            backend,
            prompt,
            received_ids,
            len(encoded.frame),
            stego_key=stego_key,
            config=config,
            semantic_plan=semantic_plan,
        )
        recovered_secret = decode_secure_text_payload(recovered_frame, master_key)
        round_trip_seconds = time.perf_counter() - started
        if recovered_secret != args.secret:
            raise RuntimeError("STEG round-trip recovered a different secret")

    started = time.perf_counter()
    normal_payload = _ordinary_sample(
        backend,
        prompt,
        len(stego_payload),
        config,
        seed=args.seed,
        semantic_plan=semantic_plan,
    )
    normal_seed = hashlib.sha256(f"normal-tail:{args.seed}".encode()).digest()
    normal = append_sentence_tail(
        backend,
        prompt,
        normal_payload,
        seed_key=normal_seed,
        config=config,
        semantic_plan=semantic_plan,
        max_tail_tokens=args.tail_tokens,
    )
    normal_seconds = time.perf_counter() - started

    print(
        f"channel: T={config.temperature}, top_k={config.top_k}, "
        f"top_p={config.top_p}, no_repeat_ngram={config.no_repeat_ngram_size}"
    )
    print(f"payload: {len(encoded.frame) * 8} bits")
    print(f"semantic anchors: {'on' if semantic_plan is not None else 'off'}")
    print(
        f"stego : {len(backend.detokenize(stego))} chars / {len(stego)} tokens "
        f"({len(stego_payload)} payload + {len(stego) - len(stego_payload)} tail; "
        f"{stego_seconds:.2f} s)"
    )
    print(
        f"normal: {len(backend.detokenize(normal))} chars / {len(normal)} tokens "
        f"({len(normal_payload)} body + {len(normal) - len(normal_payload)} tail; "
        f"{normal_seconds:.2f} s)"
    )
    if round_trip_seconds is not None:
        print(f"round-trip: PASS ({round_trip_seconds:.2f} s decode; exact secret match)")
    else:
        print("round-trip: skipped")
    print("\n--- STEGO ---")
    print(backend.detokenize(stego))
    print("\n--- NORMAL SAME-CHANNEL SAMPLE ---")
    print(backend.detokenize(normal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
