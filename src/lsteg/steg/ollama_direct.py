"""Direct byte-transport Range Coding over Ollama top-logprob candidates.

Unlike the Transformers backend, Ollama's native logprob API exposes token text
and raw bytes rather than stable vocabulary IDs.  This channel therefore uses a
prefix-free set of individually valid UTF-8 candidate byte strings.  Sender and
receiver re-query the exact visible text prefix at every step; tokenizer-ID
transport invariance is not required.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import unicodedata
from dataclasses import dataclass
from struct import Struct
from typing import Protocol

from lsteg.coding.frequencies import MAX_FREQUENCY_TOTAL, FrequencyTable
from lsteg.coding.range_coder import CodedBits, RangeDecoder, RangeEncoder
from lsteg.model.ollama_api import (
    OLLAMA_TOP_LOGPROBS_MAX,
    OllamaLogprobResponse,
    OllamaRequestMetrics,
    OllamaTokenCandidate,
    quantize_logprob,
)
from lsteg.steg.frequencies import frequency_table_entropy
from lsteg.steg.mapping import STEGANOGRAPHY_KEY_SIZE

_BYTE_MAPPING_DOMAIN = b"llm-steganography/v3/ollama-byte-candidate-mapping"
_UINT32 = Struct(">I")
_UINT64 = Struct(">Q")
_FORBIDDEN_SPECIAL_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


class OllamaCandidateSource(Protocol):
    def next_token_candidates(
        self,
        raw_prompt: str,
        *,
        top_logprobs: int,
        num_ctx: int,
        keep_alive: str,
    ) -> OllamaLogprobResponse: ...


@dataclass(frozen=True, slots=True)
class OllamaDirectConfig:
    top_logprobs: int = OLLAMA_TOP_LOGPROBS_MAX
    candidate_limit: int = 16
    temperature: float = 0.90
    top_p: float = 0.92
    frequency_total: int = MAX_FREQUENCY_TOTAL
    logprob_quantum: float = 1e-4
    num_ctx: int = 4096
    keep_alive: str = "30m"
    min_candidates: int = 2
    stability_confirmations: int = 2
    stability_max_queries: int = 4

    def __post_init__(self) -> None:
        if not 2 <= self.top_logprobs <= OLLAMA_TOP_LOGPROBS_MAX:
            raise ValueError("top_logprobs must be between 2 and 20")
        if not 2 <= self.candidate_limit <= self.top_logprobs:
            raise ValueError("candidate_limit must be in [2, top_logprobs]")
        if not 2 <= self.min_candidates <= self.candidate_limit:
            raise ValueError("min_candidates must be in [2, candidate_limit]")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if not 2 <= self.frequency_total <= MAX_FREQUENCY_TOTAL:
            raise ValueError("frequency_total is outside the Range Coder limit")
        if not math.isfinite(self.logprob_quantum) or self.logprob_quantum <= 0.0:
            raise ValueError("logprob_quantum must be finite and positive")
        if self.num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        if not self.keep_alive:
            raise ValueError("keep_alive must not be empty")
        if self.stability_confirmations <= 0:
            raise ValueError("stability_confirmations must be positive")
        if self.stability_max_queries < self.stability_confirmations:
            raise ValueError("stability_max_queries must be >= stability_confirmations")


@dataclass(frozen=True, slots=True)
class OllamaDirectDiagnostics:
    payload_bits: int
    payload_steps: int
    mean_channel_entropy: float
    mean_candidate_count: float
    mean_wall_ms: float
    mean_requests_per_step: float
    median_prompt_eval_count: float
    median_prompt_eval_ms: float
    median_eval_ms: float


@dataclass(frozen=True, slots=True)
class OllamaDirectCover:
    text: str
    payload_steps: int
    tail_steps: int
    diagnostics: OllamaDirectDiagnostics


class OllamaDirectCapacityError(RuntimeError):
    pass


class OllamaDirectCoverError(RuntimeError):
    pass


class OllamaDirectDeterminismError(RuntimeError):
    pass


def settled_next_token_candidates(
    source: OllamaCandidateSource,
    raw_prompt: str,
    *,
    config: OllamaDirectConfig,
) -> OllamaLogprobResponse:
    """Query until the derived channel table is stable on consecutive calls.

    Ollama/llama.cpp prompt-cache reuse on GPU can make the first query for an
    otherwise identical prompt differ slightly from the warmed-cache result.
    Exact steganography cannot tolerate that.  We therefore accept a table only
    after its deterministic channel fingerprint repeats consecutively.  The
    repeated request reuses the same prompt cache, so the confirmation is much
    cheaper than reevaluating the full prefix in the common case.
    """
    previous: str | None = None
    streak = 0
    fingerprints: list[str] = []
    metrics: list[OllamaRequestMetrics] = []
    for _ in range(config.stability_max_queries):
        response = source.next_token_candidates(
            raw_prompt,
            top_logprobs=config.top_logprobs,
            num_ctx=config.num_ctx,
            keep_alive=config.keep_alive,
        )
        metrics.append(response.metrics)
        fingerprint = channel_fingerprint(response.candidates, config=config)
        fingerprints.append(fingerprint)
        if fingerprint == previous:
            streak += 1
        else:
            previous = fingerprint
            streak = 1
        if streak >= config.stability_confirmations:
            return OllamaLogprobResponse(
                response.candidates,
                _aggregate_request_metrics(metrics),
            )
    abbreviated = ", ".join(value[:12] for value in fingerprints)
    raise OllamaDirectDeterminismError(
        "Ollama channel did not settle to a repeated frequency table within "
        f"{config.stability_max_queries} queries: {abbreviated}. "
        "Exact Range-Coder transport is unsafe on this runtime."
    )


def render_qwen25_raw_prompt(
    system_prompt: str,
    user_prompt: str,
    *,
    assistant_prefix: str = "",
) -> str:
    """Render the official Qwen2.5 ChatML generation prefix explicitly.

    Raw rendering makes the next request an exact prefix extension when the
    cover grows, which is the layout Ollama's prompt cache can reuse best.
    """
    for value, name in ((system_prompt, "system_prompt"), (user_prompt, "user_prompt")):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} must be a non-empty str")
        if any(marker in value for marker in _FORBIDDEN_SPECIAL_MARKERS):
            raise ValueError(f"{name} contains a reserved Qwen ChatML marker")
    if not isinstance(assistant_prefix, str):
        raise TypeError("assistant_prefix must be str")
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


def hide_bytes_ollama(
    source: OllamaCandidateSource,
    raw_prompt_prefix: str,
    fixed_cover_prefix: str,
    payload: bytes,
    *,
    stego_key: bytes,
    config: OllamaDirectConfig | None = None,
    max_steps: int = 900,
    tail_steps: int = 24,
    tail_seed: int = 20260817,
) -> OllamaDirectCover:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return OllamaDirectCover(
            fixed_cover_prefix,
            0,
            0,
            OllamaDirectDiagnostics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    _validate_key(stego_key)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    cfg = config or OllamaDirectConfig()
    target_bits = len(payload) * 8
    terminated = CodedBits(payload + b"\x80", target_bits + 1)
    decoder = RangeDecoder(terminated)
    mirror = RangeEncoder()
    continuation = bytearray()
    entropy_sum = 0.0
    candidate_sum = 0
    metrics: list[OllamaRequestMetrics] = []

    for position in range(max_steps):
        raw_context = raw_prompt_prefix + fixed_cover_prefix + continuation.decode("utf-8")
        response = settled_next_token_candidates(
            source,
            raw_context,
            config=cfg,
        )
        candidates, table = build_ollama_frequency_table(response.candidates, config=cfg)
        candidates, table = keyed_byte_candidate_permutation(
            candidates,
            table,
            stego_key=stego_key,
            position=position,
            cover_prefix=bytes(continuation),
        )
        entropy_sum += frequency_table_entropy(table)
        candidate_sum += len(candidates)
        metrics.append(response.metrics)
        symbol = decoder.decode(table)
        chosen = candidates[symbol]
        mirror.encode(table, symbol)
        continuation.extend(chosen)
        if mirror.settled_bit_length >= target_bits:
            if mirror.settled_bits().data[: len(payload)] != payload:
                raise RuntimeError("Range Coder payload-prefix invariant failed")
            payload_steps = position + 1
            break
    else:
        raise OllamaDirectCapacityError(
            f"payload did not settle within max_steps={max_steps}; "
            f"settled={mirror.settled_bit_length}/{target_bits} bits"
        )

    generated_tail = _append_tail(
        source,
        raw_prompt_prefix,
        fixed_cover_prefix,
        continuation,
        config=cfg,
        max_steps=tail_steps,
        seed=tail_seed,
    )
    text = fixed_cover_prefix + continuation.decode("utf-8")
    return OllamaDirectCover(
        text=text,
        payload_steps=payload_steps,
        tail_steps=generated_tail,
        diagnostics=_diagnostics(
            payload_bits=target_bits,
            payload_steps=payload_steps,
            entropy_sum=entropy_sum,
            candidate_sum=candidate_sum,
            metrics=metrics,
        ),
    )


def extract_bytes_ollama(
    source: OllamaCandidateSource,
    raw_prompt_prefix: str,
    fixed_cover_prefix: str,
    cover_text: str,
    payload_size: int,
    *,
    stego_key: bytes,
    config: OllamaDirectConfig | None = None,
) -> bytes:
    if payload_size < 0:
        raise ValueError("payload_size must not be negative")
    if not cover_text.startswith(fixed_cover_prefix):
        raise OllamaDirectCoverError("cover does not start with the fixed visible prefix")
    if payload_size == 0:
        return b""
    _validate_key(stego_key)
    cfg = config or OllamaDirectConfig()
    required_bits = payload_size * 8
    remaining = cover_text[len(fixed_cover_prefix) :].encode("utf-8")
    continuation = bytearray()
    encoder = RangeEncoder()
    position = 0

    while remaining:
        raw_context = raw_prompt_prefix + fixed_cover_prefix + continuation.decode("utf-8")
        response = settled_next_token_candidates(
            source,
            raw_context,
            config=cfg,
        )
        candidates, table = build_ollama_frequency_table(response.candidates, config=cfg)
        candidates, table = keyed_byte_candidate_permutation(
            candidates,
            table,
            stego_key=stego_key,
            position=position,
            cover_prefix=bytes(continuation),
        )
        matching = [
            index for index, candidate in enumerate(candidates) if remaining.startswith(candidate)
        ]
        if len(matching) != 1:
            raise OllamaDirectCoverError(
                f"cover continuation is not uniquely decodable at position {position}"
            )
        symbol = matching[0]
        chosen = candidates[symbol]
        encoder.encode(table, symbol)
        continuation.extend(chosen)
        remaining = remaining[len(chosen) :]
        position += 1
        if encoder.settled_bit_length >= required_bits:
            return encoder.settled_bits().data[:payload_size]

    raise OllamaDirectCoverError(
        f"cover ended after settling {encoder.settled_bit_length}/{required_bits} bits"
    )


def sample_ollama_cover(
    source: OllamaCandidateSource,
    raw_prompt_prefix: str,
    fixed_cover_prefix: str,
    *,
    steps: int,
    seed: int,
    config: OllamaDirectConfig | None = None,
    tail_steps: int = 24,
) -> str:
    cfg = config or OllamaDirectConfig()
    rng = random.Random(seed)
    continuation = bytearray()
    for _ in range(steps):
        raw_context = raw_prompt_prefix + fixed_cover_prefix + continuation.decode("utf-8")
        response = settled_next_token_candidates(
            source,
            raw_context,
            config=cfg,
        )
        candidates, table = build_ollama_frequency_table(response.candidates, config=cfg)
        symbol = table.symbol_for(rng.randrange(table.total))
        continuation.extend(candidates[symbol])
    _append_tail(
        source,
        raw_prompt_prefix,
        fixed_cover_prefix,
        continuation,
        config=cfg,
        max_steps=tail_steps,
        seed=seed ^ 0x5A17,
    )
    return fixed_cover_prefix + continuation.decode("utf-8")


def build_ollama_frequency_table(
    raw_candidates: tuple[OllamaTokenCandidate, ...],
    *,
    config: OllamaDirectConfig,
) -> tuple[list[bytes], FrequencyTable]:
    """Create a stable prefix-free UTF-8 alphabet from Ollama's top candidates."""
    deduplicated: dict[bytes, int] = {}
    tokens: dict[bytes, str] = {}
    for candidate in raw_candidates:
        if not _transport_safe_bytes(candidate.raw_bytes, candidate.token):
            continue
        score = quantize_logprob(candidate.logprob, quantum=config.logprob_quantum)
        previous = deduplicated.get(candidate.raw_bytes)
        if previous is None or score > previous:
            deduplicated[candidate.raw_bytes] = score
            tokens[candidate.raw_bytes] = candidate.token

    ranked = sorted(deduplicated.items(), key=lambda item: (-item[1], item[0]))
    prefix_free: list[tuple[bytes, int]] = []
    for raw_bytes, score in ranked:
        if any(
            raw_bytes.startswith(existing) or existing.startswith(raw_bytes)
            for existing, _ in prefix_free
        ):
            continue
        prefix_free.append((raw_bytes, score))
        if len(prefix_free) >= config.candidate_limit:
            break
    if len(prefix_free) < config.min_candidates:
        raise OllamaDirectCapacityError(
            "Ollama top-logprob set has too few prefix-free UTF-8 candidates: "
            f"{len(prefix_free)} < {config.min_candidates}"
        )

    scaled = [score * config.logprob_quantum / config.temperature for _, score in prefix_free]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    probabilities = [value / sum(weights) for value in weights]
    keep = len(prefix_free)
    if config.top_p < 1.0 and keep > 2:
        cumulative = 0.0
        keep = 0
        for probability in probabilities:
            cumulative += probability
            keep += 1
            if cumulative >= config.top_p:
                break
        keep = max(2, keep)
    prefix_free = prefix_free[:keep]
    weights = weights[:keep]
    frequencies = _quantize_weights(weights, total=config.frequency_total)
    return [raw_bytes for raw_bytes, _ in prefix_free], FrequencyTable(frequencies)


def keyed_byte_candidate_permutation(
    candidates: list[bytes],
    table: FrequencyTable,
    *,
    stego_key: bytes,
    position: int,
    cover_prefix: bytes,
) -> tuple[list[bytes], FrequencyTable]:
    _validate_key(stego_key)
    if position < 0:
        raise ValueError("position must not be negative")
    if len(candidates) != table.symbol_count:
        raise ValueError("candidate/table size mismatch")
    context_digest = hashlib.sha256(cover_prefix).digest()
    ranked: list[tuple[bytes, bytes, int]] = []
    for candidate, frequency in zip(candidates, table.frequencies, strict=True):
        message = (
            _BYTE_MAPPING_DOMAIN
            + _UINT64.pack(position)
            + context_digest
            + _UINT32.pack(len(candidate))
            + candidate
        )
        rank = hmac.digest(stego_key, message, "sha256")
        ranked.append((rank, candidate, frequency))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return (
        [candidate for _, candidate, _ in ranked],
        FrequencyTable([frequency for _, _, frequency in ranked]),
    )


def channel_fingerprint(
    raw_candidates: tuple[OllamaTokenCandidate, ...],
    *,
    config: OllamaDirectConfig,
) -> str:
    candidates, table = build_ollama_frequency_table(raw_candidates, config=config)
    digest = hashlib.sha256()
    for candidate, frequency in zip(candidates, table.frequencies, strict=True):
        digest.update(len(candidate).to_bytes(4, "big"))
        digest.update(candidate)
        digest.update(frequency.to_bytes(4, "big"))
    return digest.hexdigest()


def _append_tail(
    source: OllamaCandidateSource,
    raw_prompt_prefix: str,
    fixed_cover_prefix: str,
    continuation: bytearray,
    *,
    config: OllamaDirectConfig,
    max_steps: int,
    seed: int,
) -> int:
    if max_steps <= 0:
        return 0
    rng = random.Random(seed)
    generated = 0
    for _ in range(max_steps):
        raw_context = raw_prompt_prefix + fixed_cover_prefix + continuation.decode("utf-8")
        response = settled_next_token_candidates(
            source,
            raw_context,
            config=config,
        )
        candidates, table = build_ollama_frequency_table(response.candidates, config=config)
        symbol = table.symbol_for(rng.randrange(table.total))
        chosen = candidates[symbol]
        continuation.extend(chosen)
        generated += 1
        text = chosen.decode("utf-8")
        if any(mark in text for mark in "。！？.!?"):  # noqa: RUF001
            break
    return generated


def _transport_safe_bytes(raw_bytes: bytes, token: str) -> bool:
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    if not text or "\ufffd" in text:
        return False
    if any(marker in text or marker in token for marker in _FORBIDDEN_SPECIAL_MARKERS):
        return False
    for character in text:
        category = unicodedata.category(character)
        if category == "Cc" and character not in {"\n", "\t"}:
            return False
    return True


def _quantize_weights(weights: list[float], *, total: int) -> list[int]:
    count = len(weights)
    if count > total:
        raise ValueError("frequency total is too small for candidate count")
    weight_sum = sum(weights)
    remaining = total - count
    raw = [weight / weight_sum * remaining for weight in weights]
    floors = [int(value) for value in raw]
    remainders = [value - floor for value, floor in zip(raw, floors, strict=True)]
    extra = remaining - sum(floors)
    order = sorted(range(count), key=lambda index: (-remainders[index], index))
    frequencies = [1 + floor for floor in floors]
    for index in order[:extra]:
        frequencies[index] += 1
    return frequencies


def _diagnostics(
    *,
    payload_bits: int,
    payload_steps: int,
    entropy_sum: float,
    candidate_sum: int,
    metrics: list[OllamaRequestMetrics],
) -> OllamaDirectDiagnostics:
    denominator = max(payload_steps, 1)
    prompt_counts = sorted(metric.prompt_eval_count for metric in metrics[1:])
    prompt_ms = sorted(metric.prompt_eval_duration_ns / 1e6 for metric in metrics[1:])
    eval_ms = sorted(metric.eval_duration_ns / 1e6 for metric in metrics[1:])
    request_count = sum(metric.request_count for metric in metrics)
    return OllamaDirectDiagnostics(
        payload_bits=payload_bits,
        payload_steps=payload_steps,
        mean_channel_entropy=entropy_sum / denominator,
        mean_candidate_count=candidate_sum / denominator,
        mean_wall_ms=1000.0 * sum(metric.wall_seconds for metric in metrics) / denominator,
        mean_requests_per_step=request_count / denominator,
        median_prompt_eval_count=_median(prompt_counts),
        median_prompt_eval_ms=_median(prompt_ms),
        median_eval_ms=_median(eval_ms),
    )


def _aggregate_request_metrics(
    metrics: list[OllamaRequestMetrics],
) -> OllamaRequestMetrics:
    if not metrics:
        raise ValueError("metrics must not be empty")
    return OllamaRequestMetrics(
        wall_seconds=sum(metric.wall_seconds for metric in metrics),
        total_duration_ns=sum(metric.total_duration_ns for metric in metrics),
        load_duration_ns=sum(metric.load_duration_ns for metric in metrics),
        prompt_eval_count=sum(metric.prompt_eval_count for metric in metrics),
        prompt_eval_duration_ns=sum(metric.prompt_eval_duration_ns for metric in metrics),
        eval_count=sum(metric.eval_count for metric in metrics),
        eval_duration_ns=sum(metric.eval_duration_ns for metric in metrics),
        request_count=sum(metric.request_count for metric in metrics),
    )


def _median(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (float(values[middle - 1]) + float(values[middle])) / 2.0


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != STEGANOGRAPHY_KEY_SIZE:
        raise ValueError(f"stego_key must be exactly {STEGANOGRAPHY_KEY_SIZE} bytes")
