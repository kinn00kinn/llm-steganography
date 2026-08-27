"""Phase-7 local E2E smoke test using the pinned Qwen cover model.

This script exercises the real shared-key path:

    synthetic secret -> payload framing -> AEAD -> hide_bytes -> Unicode text
    -> tokenize -> extract_bytes -> AEAD verify/decrypt -> restored secret

The encrypted frame byte length is still passed explicitly to ``extract_bytes``;
self-delimiting frame recovery is a later Phase-7 orchestration task.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lsteg.model import ModelManifest, TransformersBackend
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
_QWEN_CONTROL_TOKENS = (151643, 151644, 151645, 151668)
_SYNTHETIC_SECRET = "極秘：今夜の会議は中止、明日10時にカフェで。"  # noqa: RUF001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument(
        "--secret",
        default=_SYNTHETIC_SECRET,
        help="synthetic secret used by the local E2E benchmark",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus cutoff applied within the top-k candidate pool",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="block exact repeated token n-grams in the active channel (0 disables)",
    )
    parser.add_argument(
        "--semantic-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="periodically re-anchor the hidden model context to a fixed 5-stage diary plan",
    )
    parser.add_argument(
        "--phase-min-tokens",
        type=int,
        default=40,
        help="minimum visible tokens before a sentence boundary may advance the plan",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    master_key = read_master_key(args.key_file)
    stego_key = derive_keys(master_key).steganography
    manifest = ModelManifest.from_path(args.manifest)
    backend = TransformersBackend.load(
        manifest,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
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
        excluded_token_ids=_QWEN_CONTROL_TOKENS,
        enforce_transport_invariance=True,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    encoded = encode_secure_text_payload(args.secret, master_key)

    started = time.perf_counter()
    cover_ids = hide_bytes(
        backend,
        prompt,
        encoded.frame,
        stego_key=stego_key,
        config=config,
        semantic_plan=semantic_plan,
        max_tokens=args.max_tokens,
    )
    payload_token_count = len(cover_ids)
    cover_ids = append_sentence_tail(
        backend,
        prompt,
        cover_ids,
        seed_key=stego_key,
        config=config,
        semantic_plan=semantic_plan,
        max_tail_tokens=args.tail_tokens,
    )
    encode_seconds = time.perf_counter() - started
    cover_text = backend.detokenize(cover_ids)

    # Simulate transport through a plain Unicode string.  Token identity must
    # survive this boundary or receiver-side logits diverge immediately.
    received_ids = backend.tokenize(cover_text)
    if received_ids != cover_ids:
        print("FAILED: tokenizer transport changed generated token IDs", file=sys.stderr)
        return 1

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
    decode_seconds = time.perf_counter() - started

    if recovered_secret != args.secret:
        print("FAILED: recovered secret differs from the synthetic input", file=sys.stderr)
        return 1

    print("SUCCESS: Phase-7 secure payload round-trip completed")
    print(f"encrypted payload: {len(encoded.frame) * 8} bits")
    print(
        f"channel: temperature={config.temperature}, top_k={config.top_k}, "
        f"top_p={config.top_p}, no_repeat_ngram={config.no_repeat_ngram_size}"
    )
    print(
        f"semantic anchors: {'on' if semantic_plan is not None else 'off'}"
        + (f" (min {args.phase_min_tokens} tok/phase)" if semantic_plan is not None else "")
    )
    print(
        f"cover: {len(cover_text)} Unicode code points / {len(cover_ids)} tokens "
        f"({payload_token_count} payload + {len(cover_ids) - payload_token_count} tail)"
    )
    print(f"encode: {encode_seconds:.2f} s ({len(cover_ids) / encode_seconds:.2f} cover tok/s)")
    print(
        f"decode: {decode_seconds:.2f} s ({payload_token_count / decode_seconds:.2f} payload tok/s)"
    )
    print("--- cover text ---")
    print(cover_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
