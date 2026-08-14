"""Steganographic hide/extract engine.

The cover model supplies a next-token distribution.  ``K_stego`` deterministically
permutes token/frequency pairs at every position, and the integer Range Coder
maps payload bits into those secret interval positions.

Two channel APIs are provided:

``hide`` / ``extract``
    Low-level canonical ``CodedBits`` transport retained for coder tests.

``hide_bytes`` / ``extract_bytes``
    Phase-7 byte transport for arbitrary encrypted payload bytes.  The receiver
    currently needs the payload byte length; self-delimiting secure-frame
    extraction is a later orchestration step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from lsteg.coding.frequencies import MAX_FREQUENCY_TOTAL, FrequencyTable
from lsteg.coding.range_coder import STATE_BITS, CodedBits, RangeDecoder, RangeEncoder
from lsteg.model.interface import LanguageModelBackend
from lsteg.steg.frequencies import DEFAULT_TOP_K, FREQUENCY_TOTAL, logits_to_frequency_table
from lsteg.steg.mapping import keyed_candidate_permutation


class CoverTokenError(RuntimeError):
    """A received cover token is not in the active candidate alphabet."""


class InsufficientCoverCapacityError(RuntimeError):
    """The configured token budget ended before the payload was fully embedded."""


@dataclass(frozen=True, slots=True)
class SteganographyConfig:
    """Shared parameters that sender and receiver must reproduce exactly."""

    top_k: int = DEFAULT_TOP_K
    frequency_total: int = FREQUENCY_TOTAL
    temperature: float = 1.0
    excluded_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not (1 <= self.top_k <= self.frequency_total):
            raise ValueError(
                f"top_k must satisfy 1 <= top_k <= frequency_total, "
                f"got top_k={self.top_k}, frequency_total={self.frequency_total}"
            )
        if self.frequency_total > MAX_FREQUENCY_TOTAL:
            raise ValueError(
                f"frequency_total must be <= {MAX_FREQUENCY_TOTAL}, got {self.frequency_total}"
            )
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be a number")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and greater than zero")
        object.__setattr__(self, "temperature", temperature)
        normalized: list[int] = []
        for token_id in self.excluded_token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("excluded token IDs must be integers")
            if token_id < 0:
                raise ValueError("excluded token IDs must not be negative")
            normalized.append(token_id)
        if len(set(normalized)) != len(normalized):
            raise ValueError("excluded token IDs must not contain duplicates")
        object.__setattr__(self, "excluded_token_ids", tuple(normalized))


def hide(
    backend: LanguageModelBackend,
    prompt: str,
    coded_bits: CodedBits,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    max_tokens: int = 4096,
) -> list[int]:
    """Embed a canonical ``CodedBits`` stream into generated cover token IDs."""
    if coded_bits.bit_length == 0:
        return []
    _validate_max_tokens(max_tokens)
    cfg = config or SteganographyConfig()
    token_ids = backend.tokenize(prompt)
    decoder = RangeDecoder(coded_bits)
    cover_tokens: list[int] = []
    needed_bits = coded_bits.bit_length + STATE_BITS

    for position in range(max_tokens):
        top_ids, table = _position_table(
            backend,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
        )
        symbol_index = decoder.decode(table)
        token_id = top_ids[symbol_index]
        token_ids.append(token_id)
        cover_tokens.append(token_id)
        if decoder.input_bits_read >= needed_bits:
            return cover_tokens

    raise InsufficientCoverCapacityError(
        f"coded stream did not finish within max_tokens={max_tokens}"
    )


def extract(
    backend: LanguageModelBackend,
    prompt: str,
    cover_tokens: Sequence[int],
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
) -> CodedBits:
    """Recover a canonical coded-bit representation from received cover tokens."""
    cfg = config or SteganographyConfig()
    token_ids = backend.tokenize(prompt)
    encoder = RangeEncoder()

    for position, token_id in enumerate(cover_tokens):
        top_ids, table = _position_table(
            backend,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
        )
        symbol_index = _symbol_index(top_ids, token_id, position, cfg.top_k)
        encoder.encode(table, symbol_index)
        token_ids.append(token_id)

    return encoder.finish()


def hide_bytes(
    backend: LanguageModelBackend,
    prompt: str,
    payload: bytes,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    max_tokens: int = 4096,
) -> list[int]:
    """Embed arbitrary payload bytes using the LLM's dynamic frequency tables.

    A single termination bit is appended to the arithmetic-decoder input.  A
    mirror encoder tracks how many payload-prefix bits are irrevocably settled;
    generation stops as soon as all payload bits are recoverable.
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return []
    _validate_max_tokens(max_tokens)
    cfg = config or SteganographyConfig()
    target_bits = len(payload) * 8
    terminated = CodedBits(payload + b"\x80", target_bits + 1)
    decoder = RangeDecoder(terminated)
    mirror = RangeEncoder()
    token_ids = backend.tokenize(prompt)
    cover_tokens: list[int] = []

    for position in range(max_tokens):
        top_ids, table = _position_table(
            backend,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
        )
        symbol_index = decoder.decode(table)
        mirror.encode(table, symbol_index)
        token_id = top_ids[symbol_index]
        token_ids.append(token_id)
        cover_tokens.append(token_id)

        if mirror.settled_bit_length >= target_bits:
            recovered_prefix = mirror.settled_bits().data[: len(payload)]
            if recovered_prefix != payload:  # pragma: no cover - coder symmetry invariant
                raise RuntimeError("range mapping failed to preserve the payload prefix")
            return cover_tokens

    raise InsufficientCoverCapacityError(
        f"payload did not settle within max_tokens={max_tokens}"
    )


def extract_bytes(
    backend: LanguageModelBackend,
    prompt: str,
    cover_tokens: Sequence[int],
    payload_size: int,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
) -> bytes:
    """Recover an exact payload prefix from cover tokens.

    ``payload_size`` is intentionally explicit in this Phase-7 primitive.  A
    higher-level secure-frame decoder can later infer it from authenticated
    framing once enough prefix bytes have settled.
    """
    if isinstance(payload_size, bool) or not isinstance(payload_size, int):
        raise TypeError("payload_size must be int")
    if payload_size < 0:
        raise ValueError("payload_size must not be negative")
    if payload_size == 0:
        if cover_tokens:
            raise ValueError("empty payload must use an empty cover token sequence")
        return b""

    cfg = config or SteganographyConfig()
    required_bits = payload_size * 8
    token_ids = backend.tokenize(prompt)
    encoder = RangeEncoder()

    for position, token_id in enumerate(cover_tokens):
        top_ids, table = _position_table(
            backend,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
        )
        symbol_index = _symbol_index(top_ids, token_id, position, cfg.top_k)
        encoder.encode(table, symbol_index)
        token_ids.append(token_id)
        if encoder.settled_bit_length >= required_bits:
            return encoder.settled_bits().data[:payload_size]

    raise InsufficientCoverCapacityError(
        f"cover settled {encoder.settled_bit_length} of {required_bits} required bits"
    )


def _position_table(
    backend: LanguageModelBackend,
    context_token_ids: Sequence[int],
    *,
    position: int,
    stego_key: bytes,
    config: SteganographyConfig,
) -> tuple[list[int], FrequencyTable]:
    logits = backend.next_logits(context_token_ids)
    top_ids, table = logits_to_frequency_table(
        logits,
        top_k=config.top_k,
        total=config.frequency_total,
        temperature=config.temperature,
        excluded_token_ids=config.excluded_token_ids,
    )
    return keyed_candidate_permutation(
        top_ids,
        table,
        stego_key=stego_key,
        position=position,
        context_token_ids=context_token_ids,
    )


def _symbol_index(
    top_ids: Sequence[int],
    token_id: int,
    position: int,
    top_k: int,
) -> int:
    try:
        return top_ids.index(token_id)
    except ValueError as error:
        raise CoverTokenError(
            f"cover token {token_id!r} is not in the top-{top_k} alphabet "
            f"at cover position {position}; sender/receiver artifacts or config differ"
        ) from error


def _validate_max_tokens(max_tokens: int) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be int")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
