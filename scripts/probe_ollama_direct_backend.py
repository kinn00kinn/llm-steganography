"""Probe direct Ollama logprobs, settled determinism, and prompt-cache speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

from lsteg.model import OllamaAPIClient, OllamaLogprobResponse
from lsteg.steg import (
    OllamaDirectConfig,
    build_ollama_frequency_table,
    channel_fingerprint,
    render_qwen25_raw_prompt,
    settled_next_token_candidates,
)

DEFAULT_IDENTITY = Path("artifacts/ollama/qwen2.5-7b.identity.json")
SYSTEM_PROMPT = (
    "自然な現代日本語の一人称の日常文を書く。設定、時刻、場所を保ち、"
    "普通の出来事だけを具体的に描写する。恋愛、身体接触、過去回想、"
    "新しい登場人物を勝手に追加しない。"
)
USER_PROMPT = (
    "平日の正午前。大学の研究室で午前中の作業をしている。友人一人に昼食へ誘われ、"
    "学食へ行き、食後は研究室へ戻って作業を再開する。自然な短い日常文として続ける。"
)
FIXED_PREFIX = "正午が近づいたころ、私は大学の研究室で午前中の作業を続けていた。"
QUANTUM_SCAN = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--identity-file", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--write-identity", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--stability-confirmations", type=int, default=2)
    parser.add_argument("--stability-max-queries", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = OllamaDirectConfig(
        candidate_limit=args.candidate_limit,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        stability_confirmations=args.stability_confirmations,
        stability_max_queries=args.stability_max_queries,
    )
    raw_prefix = render_qwen25_raw_prompt(SYSTEM_PROMPT, USER_PROMPT)
    raw_context = raw_prefix + FIXED_PREFIX

    with OllamaAPIClient(model=args.model, base_url=args.ollama_url) as client:
        identity = client.identity()
        print(json.dumps(identity.as_dict(), ensure_ascii=False, indent=2))
        if args.identity_file.exists() and not args.write_identity:
            expected = type(identity).from_path(args.identity_file)
            expected.assert_matches(identity)
            print(f"identity lock: PASS ({args.identity_file})")
        if args.write_identity:
            identity.write(args.identity_file)
            print(f"identity lock written: {args.identity_file}")

        raw_responses = [
            client.next_token_candidates(
                raw_context,
                top_logprobs=config.top_logprobs,
                num_ctx=config.num_ctx,
                keep_alive=config.keep_alive,
            )
            for _ in range(args.repeat)
        ]
        raw_fingerprints = [
            channel_fingerprint(response.candidates, config=config) for response in raw_responses
        ]
        raw_deterministic = len(set(raw_fingerprints)) == 1
        print(
            "single-shot determinism: "
            f"{'PASS' if raw_deterministic else 'FAIL'} ({args.repeat} repeats)"
        )
        print("single-shot fingerprints: " + ", ".join(fp[:12] for fp in raw_fingerprints))
        membership = [_candidate_membership_fingerprint(response) for response in raw_responses]
        print(f"raw top-{config.top_logprobs} membership stable: {len(set(membership)) == 1}")

        quantum_results: dict[str, int] = {}
        for quantum in QUANTUM_SCAN:
            scan_config = replace(config, logprob_quantum=quantum)
            fingerprints = [
                channel_fingerprint(response.candidates, config=scan_config)
                for response in raw_responses
            ]
            quantum_results[f"{quantum:g}"] = len(set(fingerprints))
        print("quantum scan unique fingerprints:")
        print(json.dumps(quantum_results, indent=2))

        settled_fingerprints: list[str] = []
        settled_requests: list[int] = []
        for _ in range(args.repeat):
            response = settled_next_token_candidates(client, raw_context, config=config)
            settled_fingerprints.append(channel_fingerprint(response.candidates, config=config))
            settled_requests.append(response.metrics.request_count)
        settled_deterministic = len(set(settled_fingerprints)) == 1
        print(
            "settled determinism: "
            f"{'PASS' if settled_deterministic else 'FAIL'} ({args.repeat} repeats)"
        )
        print("settled fingerprints: " + ", ".join(fp[:12] for fp in settled_fingerprints))
        print(
            "settled requests/query: "
            f"mean={statistics.fmean(settled_requests):.2f}, max={max(settled_requests)}"
        )
        if not settled_deterministic:
            return 2

        continuation = bytearray()
        wall_ms: list[float] = []
        request_counts: list[int] = []
        prompt_counts_per_request: list[float] = []
        prompt_ms: list[float] = []
        eval_ms: list[float] = []
        candidate_counts: list[int] = []
        entropies: list[float] = []
        from lsteg.steg.frequencies import frequency_table_entropy

        for _ in range(args.steps):
            response = settled_next_token_candidates(
                client,
                raw_prefix + FIXED_PREFIX + continuation.decode("utf-8"),
                config=config,
            )
            candidates, table = build_ollama_frequency_table(response.candidates, config=config)
            index = max(
                range(len(candidates)),
                key=lambda idx: (table.frequency(idx), -idx),
            )
            continuation.extend(candidates[index])
            metrics = response.metrics
            wall_ms.append(metrics.wall_seconds * 1000.0)
            request_counts.append(metrics.request_count)
            prompt_counts_per_request.append(
                metrics.prompt_eval_count / max(metrics.request_count, 1)
            )
            prompt_ms.append(metrics.prompt_eval_duration_ns / 1e6)
            eval_ms.append(metrics.eval_duration_ns / 1e6)
            candidate_counts.append(len(candidates))
            entropies.append(frequency_table_entropy(table))

    steady_slice = slice(2, None) if len(wall_ms) > 2 else slice(None)
    steady_wall = wall_ms[steady_slice]
    steady_requests = request_counts[steady_slice]
    steady_prompt_counts = prompt_counts_per_request[steady_slice]
    steady_prompt_ms = prompt_ms[steady_slice]
    steady_eval_ms = eval_ms[steady_slice]
    summary = {
        "steps": args.steps,
        "mean_candidates": statistics.fmean(candidate_counts),
        "mean_channel_entropy_bits": statistics.fmean(entropies),
        "first_wall_ms": wall_ms[0],
        "steady_median_wall_ms": statistics.median(steady_wall),
        "steady_p95_wall_ms": _percentile(steady_wall, 0.95),
        "steady_mean_requests_per_step": statistics.fmean(steady_requests),
        "steady_median_prompt_eval_count_per_request": statistics.median(steady_prompt_counts),
        "steady_median_prompt_eval_ms_per_step": statistics.median(steady_prompt_ms),
        "steady_median_eval_ms_per_step": statistics.median(steady_eval_ms),
        "cache_effective": statistics.median(steady_prompt_counts) <= 8,
        "sample_text": FIXED_PREFIX + continuation.decode("utf-8"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["cache_effective"]:
        print(
            "WARNING: prompt cache did not look incremental; keep the model loaded, "
            "ensure GPU offload is active, and inspect `ollama ps` before E2E benchmarking."
        )
    return 0


def _candidate_membership_fingerprint(response: OllamaLogprobResponse) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(
        response.candidates,
        key=lambda item: (item.raw_bytes, item.token),
    ):
        digest.update(len(candidate.raw_bytes).to_bytes(4, "big"))
        digest.update(candidate.raw_bytes)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
