"""Direct Range-Coder A/B using the installed Ollama qwen2.5:7b model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lsteg.model import OllamaAPIClient, OllamaModelIdentity
from lsteg.payload import (
    decode_compact_secure_text_payload,
    derive_keys,
    encode_compact_secure_text_payload,
    encode_secure_text_payload,
    read_master_key,
)
from lsteg.steg import (
    OllamaDirectConfig,
    extract_bytes_ollama,
    hide_bytes_ollama,
    render_qwen25_raw_prompt,
    sample_ollama_cover,
)
from lsteg.steg.whitening import xor_whiten

DEFAULT_IDENTITY = Path("artifacts/ollama/qwen2.5-7b.identity.json")
SYSTEM_PROMPT = (
    "自然な現代日本語の一人称の日常文を書く。設定、時刻、場所を保ち、"
    "普通の出来事だけを具体的に描写する。恋愛、身体接触、過去回想、"
    "新しい登場人物を勝手に追加しない。"
)
USER_PROMPT = (
    "平日の正午前。大学の研究室で午前中の作業をしている。友人一人に昼食へ誘われ、"
    "学食へ行き、食後は研究室へ戻って作業を再開する。300〜500字程度の自然な日常文。"
)
FIXED_PREFIX = "正午が近づいたころ、私は大学の研究室で午前中の作業を続けていた。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--secret", default="確認")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--identity-file", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--stability-confirmations", type=int, default=2)
    parser.add_argument("--stability-max-queries", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--tail-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.identity_file.exists():
        raise SystemExit(
            f"missing identity lock {args.identity_file}; run "
            "scripts/probe_ollama_direct_backend.py --write-identity first"
        )
    expected_identity = OllamaModelIdentity.from_path(args.identity_file)
    master_key = read_master_key(args.key_file)
    stego_key = derive_keys(master_key).steganography
    legacy = encode_secure_text_payload(args.secret, master_key)
    compact = encode_compact_secure_text_payload(args.secret, master_key)
    transport_payload = xor_whiten(compact.frame, stego_key)
    config = OllamaDirectConfig(
        candidate_limit=args.candidate_limit,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        stability_confirmations=args.stability_confirmations,
        stability_max_queries=args.stability_max_queries,
    )
    raw_prompt_prefix = render_qwen25_raw_prompt(SYSTEM_PROMPT, USER_PROMPT)

    with OllamaAPIClient(model=args.model, base_url=args.ollama_url) as client:
        expected_identity.assert_matches(client.identity())
        started = time.perf_counter()
        cover = hide_bytes_ollama(
            client,
            raw_prompt_prefix,
            FIXED_PREFIX,
            transport_payload,
            stego_key=stego_key,
            config=config,
            max_steps=args.max_steps,
            tail_steps=args.tail_steps,
            tail_seed=args.seed,
        )
        encode_seconds = time.perf_counter() - started

        started = time.perf_counter()
        recovered_transport = extract_bytes_ollama(
            client,
            raw_prompt_prefix,
            FIXED_PREFIX,
            cover.text,
            len(transport_payload),
            stego_key=stego_key,
            config=config,
        )
        recovered = decode_compact_secure_text_payload(
            xor_whiten(recovered_transport, stego_key), master_key
        )
        decode_seconds = time.perf_counter() - started
        if recovered != args.secret:
            raise RuntimeError("direct Ollama Range channel recovered a different secret")

        started = time.perf_counter()
        normal = sample_ollama_cover(
            client,
            raw_prompt_prefix,
            FIXED_PREFIX,
            steps=cover.payload_steps,
            seed=args.seed,
            config=config,
            tail_steps=args.tail_steps,
        )
        normal_seconds = time.perf_counter() - started

    compact_bits = len(compact.frame) * 8
    legacy_bits = len(legacy.frame) * 8
    diagnostics = cover.diagnostics
    print("channel: Ollama qwen2.5:7b top-logprobs + byte-prefix Range Coding")
    print(
        f"distribution: server top_logprobs={config.top_logprobs}, "
        f"safe_limit={config.candidate_limit}, T={config.temperature}, top_p={config.top_p}, "
        f"settle={config.stability_confirmations}/{config.stability_max_queries}"
    )
    print(
        f"payload: legacy={legacy_bits} bits / compact={compact_bits} bits "
        f"(saved {legacy_bits - compact_bits} bits, "
        f"{100 * (1 - compact_bits / legacy_bits):.1f}%)"
    )
    print(
        f"stego : {len(cover.text)} chars / {cover.payload_steps} payload steps "
        f"+ {cover.tail_steps} tail; effective "
        f"{compact_bits / cover.payload_steps:.3f} bit/payload-step; "
        f"{encode_seconds:.2f} s"
    )
    print(
        f"channel diagnostics: entropy={diagnostics.mean_channel_entropy:.3f} bit/step, "
        f"candidates={diagnostics.mean_candidate_count:.2f}, "
        f"wall={diagnostics.mean_wall_ms:.1f} ms/step, "
        f"requests={diagnostics.mean_requests_per_step:.2f}/step, "
        f"prompt_eval_count(median)={diagnostics.median_prompt_eval_count:.1f}, "
        f"prompt_eval={diagnostics.median_prompt_eval_ms:.1f} ms, "
        f"token_eval={diagnostics.median_eval_ms:.1f} ms"
    )
    print(f"normal: {len(normal)} chars; {normal_seconds:.2f} s")
    print(f"round-trip: PASS ({decode_seconds:.2f} s decode; exact secret match)")
    print("\n--- OLLAMA QWEN2.5 DIRECT RANGE STEGO ---")
    print(cover.text)
    print("\n--- OLLAMA QWEN2.5 NORMAL SAME-BASE SAMPLE ---")
    print(normal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
