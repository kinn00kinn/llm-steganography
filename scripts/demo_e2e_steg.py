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
from lsteg.steg import SteganographyConfig, extract_bytes, hide_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
DEFAULT_CACHE = PROJECT_ROOT / "artifacts" / "model-cache"
_QWEN_CONTROL_TOKENS = (151643, 151644, 151645)
_SYNTHETIC_SECRET = "極秘：今夜の会議は中止、明日10時にカフェで。"  # noqa: RUF001
_DEFAULT_PROMPT = "架空のニュース記事：\n本日午後、東京都内の"  # noqa: RUF001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
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
    config = SteganographyConfig(
        top_k=args.top_k,
        temperature=args.temperature,
        excluded_token_ids=_QWEN_CONTROL_TOKENS,
    )

    encoded = encode_secure_text_payload(_SYNTHETIC_SECRET, master_key)

    started = time.perf_counter()
    cover_ids = hide_bytes(
        backend,
        _DEFAULT_PROMPT,
        encoded.frame,
        stego_key=stego_key,
        config=config,
        max_tokens=args.max_tokens,
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
        _DEFAULT_PROMPT,
        received_ids,
        len(encoded.frame),
        stego_key=stego_key,
        config=config,
    )
    recovered_secret = decode_secure_text_payload(recovered_frame, master_key)
    decode_seconds = time.perf_counter() - started

    if recovered_secret != _SYNTHETIC_SECRET:
        print("FAILED: recovered secret differs from the synthetic input", file=sys.stderr)
        return 1

    print("SUCCESS: Phase-7 secure payload round-trip completed")
    print(f"encrypted payload: {len(encoded.frame) * 8} bits")
    print(f"cover: {len(cover_text)} Unicode code points / {len(cover_ids)} tokens")
    print(f"encode: {encode_seconds:.2f} s")
    print(f"decode: {decode_seconds:.2f} s")
    print("--- cover text ---")
    print(cover_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
