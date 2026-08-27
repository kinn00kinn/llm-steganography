from __future__ import annotations

from dataclasses import replace

import pytest

from lsteg.model.ollama_api import (
    OllamaLogprobResponse,
    OllamaRequestMetrics,
    OllamaTokenCandidate,
)
from lsteg.steg.ollama_direct import (
    OllamaDirectConfig,
    OllamaDirectDeterminismError,
    build_ollama_frequency_table,
    channel_fingerprint,
    extract_bytes_ollama,
    hide_bytes_ollama,
    render_qwen25_raw_prompt,
    settled_next_token_candidates,
)

ZERO_METRICS = OllamaRequestMetrics(0.001, 0, 0, 1, 1000, 1, 1000)


class _FakeSource:
    def __init__(self, candidates: tuple[OllamaTokenCandidate, ...]) -> None:
        self.candidates = candidates
        self.prompts: list[str] = []

    def next_token_candidates(
        self,
        raw_prompt: str,
        *,
        top_logprobs: int,
        num_ctx: int,
        keep_alive: str,
    ) -> OllamaLogprobResponse:
        self.prompts.append(raw_prompt)
        assert top_logprobs >= len(self.candidates)
        assert num_ctx > 0
        assert keep_alive
        return OllamaLogprobResponse(self.candidates, ZERO_METRICS)


def _candidates() -> tuple[OllamaTokenCandidate, ...]:
    return (
        OllamaTokenCandidate("あ".encode(), "あ", -0.1),
        OllamaTokenCandidate("い".encode(), "い", -0.2),
        OllamaTokenCandidate("う".encode(), "う", -0.3),
        OllamaTokenCandidate("え".encode(), "え", -0.4),
    )


def _config() -> OllamaDirectConfig:
    return OllamaDirectConfig(
        top_logprobs=4,
        candidate_limit=4,
        temperature=1.0,
        top_p=1.0,
        logprob_quantum=1e-4,
    )


def test_qwen25_raw_prompt_is_prefix_extendable() -> None:
    base = render_qwen25_raw_prompt("system", "user", assistant_prefix="導入。")
    assert base.endswith("<|im_start|>assistant\n導入。")
    assert (base + "続き").startswith(base)


def test_frequency_table_filters_prefix_ambiguous_candidates() -> None:
    candidates = (
        OllamaTokenCandidate(b"a", "a", -0.1),
        OllamaTokenCandidate(b"ab", "ab", -0.2),
        OllamaTokenCandidate(b"b", "b", -0.3),
        OllamaTokenCandidate(b"c", "c", -0.4),
    )
    selected, table = build_ollama_frequency_table(candidates, config=_config())
    assert b"a" in selected
    assert b"ab" not in selected
    assert table.total == 32768
    for left in selected:
        for right in selected:
            if left == right:
                continue
            assert not left.startswith(right)


def test_small_logprob_noise_is_quantized_out() -> None:
    config = _config()
    first = _candidates()
    second = tuple(replace(candidate, logprob=candidate.logprob + 0.00002) for candidate in first)
    assert channel_fingerprint(first, config=config) == channel_fingerprint(second, config=config)


def test_direct_byte_range_round_trip() -> None:
    source = _FakeSource(_candidates())
    config = _config()
    raw_prefix = render_qwen25_raw_prompt("system", "user")
    fixed = "正午。"
    payload = b"OK"
    key = bytes(range(32))

    cover = hide_bytes_ollama(
        source,
        raw_prefix,
        fixed,
        payload,
        stego_key=key,
        config=config,
        max_steps=100,
        tail_steps=0,
    )
    recovered = extract_bytes_ollama(
        source,
        raw_prefix,
        fixed,
        cover.text,
        len(payload),
        stego_key=key,
        config=config,
    )
    assert recovered == payload
    assert cover.text.startswith(fixed)
    assert cover.payload_steps > 0


def test_direct_cover_rejects_wrong_fixed_prefix() -> None:
    source = _FakeSource(_candidates())
    with pytest.raises(Exception, match="fixed visible prefix"):
        extract_bytes_ollama(
            source,
            render_qwen25_raw_prompt("system", "user"),
            "正午。",
            "朝。あいう",
            1,
            stego_key=bytes(range(32)),
            config=_config(),
        )


class _PerPromptSettlingSource:
    def __init__(
        self,
        first: tuple[OllamaTokenCandidate, ...],
        settled: tuple[OllamaTokenCandidate, ...],
    ) -> None:
        self.first = first
        self.settled = settled
        self.counts: dict[str, int] = {}

    def next_token_candidates(
        self,
        raw_prompt: str,
        *,
        top_logprobs: int,
        num_ctx: int,
        keep_alive: str,
    ) -> OllamaLogprobResponse:
        del top_logprobs, num_ctx, keep_alive
        count = self.counts.get(raw_prompt, 0)
        self.counts[raw_prompt] = count + 1
        candidates = self.first if count == 0 else self.settled
        return OllamaLogprobResponse(candidates, ZERO_METRICS)


class _AlternatingSource:
    def __init__(
        self,
        first: tuple[OllamaTokenCandidate, ...],
        second: tuple[OllamaTokenCandidate, ...],
    ) -> None:
        self.first = first
        self.second = second
        self.count = 0

    def next_token_candidates(
        self,
        raw_prompt: str,
        *,
        top_logprobs: int,
        num_ctx: int,
        keep_alive: str,
    ) -> OllamaLogprobResponse:
        del raw_prompt, top_logprobs, num_ctx, keep_alive
        candidates = self.first if self.count % 2 == 0 else self.second
        self.count += 1
        return OllamaLogprobResponse(candidates, ZERO_METRICS)


def test_settled_query_accepts_consecutive_warm_table() -> None:
    first = _candidates()
    settled = tuple(
        replace(candidate, logprob=candidate.logprob - (0.01 if index == 0 else 0.0))
        for index, candidate in enumerate(first)
    )
    source = _PerPromptSettlingSource(first, settled)
    response = settled_next_token_candidates(source, "prompt", config=_config())
    assert channel_fingerprint(response.candidates, config=_config()) == channel_fingerprint(
        settled, config=_config()
    )
    assert response.metrics.request_count == 3


def test_settled_query_fails_closed_when_table_never_stabilizes() -> None:
    first = _candidates()
    second = tuple(
        replace(candidate, logprob=candidate.logprob - (0.01 if index == 0 else 0.0))
        for index, candidate in enumerate(first)
    )
    source = _AlternatingSource(first, second)
    config = replace(_config(), stability_max_queries=4)
    with pytest.raises(OllamaDirectDeterminismError, match="did not settle"):
        settled_next_token_candidates(source, "prompt", config=config)


def test_round_trip_survives_first_query_cache_jitter_per_prefix() -> None:
    first = _candidates()
    settled = tuple(
        replace(candidate, logprob=candidate.logprob - (0.01 if index == 0 else 0.0))
        for index, candidate in enumerate(first)
    )
    config = _config()
    raw_prefix = render_qwen25_raw_prompt("system", "user")
    fixed = "正午。"
    payload = b"OK"
    key = bytes(range(32))

    sender = _PerPromptSettlingSource(first, settled)
    cover = hide_bytes_ollama(
        sender,
        raw_prefix,
        fixed,
        payload,
        stego_key=key,
        config=config,
        max_steps=100,
        tail_steps=0,
    )
    receiver = _PerPromptSettlingSource(first, settled)
    recovered = extract_bytes_ollama(
        receiver,
        raw_prefix,
        fixed,
        cover.text,
        len(payload),
        stego_key=key,
        config=config,
    )
    assert recovered == payload
    assert cover.diagnostics.mean_requests_per_step >= 2.0
