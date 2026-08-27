"""Quality-first direct Range-Coder benchmark for a pinned Transformers model.

The experiment keeps the production Range-Coder channel, uses a compact
authenticated payload, and compares STEG against ordinary sampling from the
same quantized model distribution. The semantic scaffold constrains only
state/chronology; exact wording is left to the model to preserve carrier
entropy.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path

from lsteg.model import ModelManifest, TransformersBackend, start_incremental_logits
from lsteg.model.errors import ModelInputError
from lsteg.payload import (
    decode_compact_secure_text_payload,
    derive_keys,
    encode_compact_secure_text_payload,
    encode_secure_text_payload,
    read_master_key,
)
from lsteg.steg import (
    InsufficientCoverCapacityError,
    SteganographyConfig,
    append_sentence_tail,
    extract_bytes,
    hide_bytes,
    university_lunch_guarded_sparse_cover_plan,
    university_lunch_sparse_cover_plan,
)
from lsteg.steg.engine import _position_table
from lsteg.steg.whitening import xor_whiten

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"


@dataclass(frozen=True, slots=True)
class _NormalSample:
    token_ids: list[int]
    context_limited: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--secret", default="確認")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--semantic-plan",
        choices=("sparse", "guarded"),
        default="sparse",
        help="semantic scaffold variant; guarded adds hard invariants from observed failures",
    )
    parser.add_argument("--phase-min-tokens", type=int, default=8)
    parser.add_argument("--tail-tokens", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    return parser


def _ordinary_sample(
    backend: TransformersBackend,
    prompt: str,
    token_count: int,
    config: SteganographyConfig,
    *,
    seed: int,
    semantic_plan,
) -> _NormalSample:
    rng = random.Random(seed)
    try:
        context = backend.tokenize(prompt)
        session = start_incremental_logits(backend, context)
    except ModelInputError:
        return _NormalSample([], True)

    generated: list[int] = []
    semantic_state = semantic_plan.new_state()

    for position in range(token_count):
        try:
            replacement = semantic_state.maybe_advance(
                backend,
                generated,
                position=position,
            )
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
        except ModelInputError:
            return _NormalSample(generated, True)

        token_id = top_ids[table.symbol_for(rng.randrange(table.total))]
        context.append(token_id)
        generated.append(token_id)
        if position + 1 < token_count:
            try:
                session.append(token_id)
            except ModelInputError:
                return _NormalSample(generated, True)
    return _NormalSample(generated, False)


def _print_capacity_failure(
    error: InsufficientCoverCapacityError,
    *,
    prompt_tokens: int,
    context_limit: int,
    legacy_bits: int,
    compact_bits: int,
) -> None:
    print("CAPACITY: FAIL")
    print(f"initial prompt: {prompt_tokens} tokens / model context {context_limit}")
    print(
        f"payload: legacy={legacy_bits} bits / compact={compact_bits} bits "
        f"(saved {legacy_bits - compact_bits} bits)"
    )
    print(str(error))
    diagnostics = error.diagnostics
    if diagnostics is not None and diagnostics.generated_tokens:
        settled_rate = diagnostics.settled_bits / diagnostics.generated_tokens
        print(f"observed settled rate: {settled_rate:.3f} bit/cover-token")
        if diagnostics.mean_table_entropy > 0.0:
            optimistic_tokens = compact_bits / diagnostics.mean_table_entropy
            print(
                "entropy-only lower-bound estimate: "
                f"~{optimistic_tokens:.0f} payload tokens (settlement overhead excluded)"
            )


def main() -> int:
    args = build_parser().parse_args()
    master_key = read_master_key(args.key_file)
    stego_key = derive_keys(master_key).steganography
    backend = TransformersBackend.load(
        ModelManifest.from_path(args.manifest),
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    plan_factory = (
        university_lunch_guarded_sparse_cover_plan
        if args.semantic_plan == "guarded"
        else university_lunch_sparse_cover_plan
    )
    plan = plan_factory(min_tokens_per_phase=args.phase_min_tokens)
    prompt = plan.initial_prompt(backend)
    prompt_tokens = len(backend.tokenize(prompt))
    config = SteganographyConfig(
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        excluded_token_ids=backend.special_token_ids,
        enforce_transport_invariance=True,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    legacy = encode_secure_text_payload(args.secret, master_key)
    compact = encode_compact_secure_text_payload(args.secret, master_key)
    transport_payload = xor_whiten(compact.frame, stego_key)
    compact_bits = len(compact.frame) * 8
    legacy_bits = len(legacy.frame) * 8

    started = time.perf_counter()
    try:
        payload_tokens = hide_bytes(
            backend,
            prompt,
            transport_payload,
            stego_key=stego_key,
            config=config,
            semantic_plan=plan,
            max_tokens=args.max_tokens,
        )
    except InsufficientCoverCapacityError as error:
        _print_capacity_failure(
            error,
            prompt_tokens=prompt_tokens,
            context_limit=backend.manifest.max_context_tokens,
            legacy_bits=legacy_bits,
            compact_bits=compact_bits,
        )
        return 2

    stego = append_sentence_tail(
        backend,
        prompt,
        payload_tokens,
        seed_key=stego_key,
        config=config,
        semantic_plan=plan,
        max_tail_tokens=args.tail_tokens,
    )
    encode_seconds = time.perf_counter() - started

    stego_text = backend.detokenize(stego)
    received = backend.tokenize(stego_text)
    if received != stego:
        raise RuntimeError("STEG text changed token IDs across Unicode transport")

    started = time.perf_counter()
    recovered_transport = extract_bytes(
        backend,
        prompt,
        received,
        len(transport_payload),
        stego_key=stego_key,
        config=config,
        semantic_plan=plan,
    )
    recovered_frame = xor_whiten(recovered_transport, stego_key)
    recovered_secret = decode_compact_secure_text_payload(recovered_frame, master_key)
    decode_seconds = time.perf_counter() - started
    if recovered_secret != args.secret:
        raise RuntimeError("quality-first Range channel recovered a different secret")

    started = time.perf_counter()
    normal_result = _ordinary_sample(
        backend,
        prompt,
        len(payload_tokens),
        config,
        seed=args.seed,
        semantic_plan=plan,
    )
    normal_seed = hashlib.sha256(f"quality-normal-tail:{args.seed}".encode()).digest()
    normal = append_sentence_tail(
        backend,
        prompt,
        normal_result.token_ids,
        seed_key=normal_seed,
        config=config,
        semantic_plan=plan,
        max_tail_tokens=args.tail_tokens,
    )
    normal_seconds = time.perf_counter() - started

    effective = compact_bits / len(payload_tokens)
    normal_text = backend.detokenize(normal)
    print("channel: direct quantized Qwen distribution + Range Coding")
    print(
        f"base distribution: T={config.temperature}, top_k={config.top_k}, "
        f"top_p={config.top_p}, presence_penalty={config.presence_penalty}, "
        f"no_repeat_ngram={config.no_repeat_ngram_size}"
    )
    print(
        f"semantic scaffold: {args.semantic_plan}/current-phase-only "
        f"(min_tokens={args.phase_min_tokens}, final carrier repeats)"
    )
    print(
        f"initial prompt: {prompt_tokens} tokens / "
        f"model context {backend.manifest.max_context_tokens}"
    )
    print(
        f"payload: legacy={legacy_bits} bits / compact={compact_bits} bits "
        f"(saved {legacy_bits - compact_bits} bits, "
        f"{100 * (1 - compact_bits / legacy_bits):.1f}%)"
    )
    print(
        f"stego : {len(stego_text)} chars / {len(stego)} tokens "
        f"({len(payload_tokens)} payload + {len(stego) - len(payload_tokens)} tail; "
        f"effective {effective:.3f} bit/payload-token; {encode_seconds:.2f} s)"
    )
    normal_note = " [context-limited]" if normal_result.context_limited else ""
    print(
        f"normal: {len(normal_text)} chars / {len(normal)} tokens "
        f"({len(normal_result.token_ids)} body + "
        f"{len(normal) - len(normal_result.token_ids)} tail; "
        f"{normal_seconds:.2f} s){normal_note}"
    )
    print(f"round-trip: PASS ({decode_seconds:.2f} s decode; exact secret match)")
    print("\n--- DIRECT RANGE STEGO ---")
    print(stego_text)
    print("\n--- NORMAL SAME-BASE SAMPLE ---")
    print(normal_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
