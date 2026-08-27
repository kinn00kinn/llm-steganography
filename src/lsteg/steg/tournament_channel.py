"""SynthID-inspired empirical-pool Range-Coder steganography.

This experimental channel keeps the *base* Qwen decoder fixed and adds secret
correlation without directly widening or reweighting that decoder distribution.
At each visible token position it draws ``2**pool_bits`` iid candidates from the
active model distribution using a context-bound keyed PRF.  Candidate
collisions are **not** discarded.  Instead their multiplicities define a small
empirical distribution, for example::

    [A, A, A, B, B, C, D, D] -> {A: 3, B: 2, C: 1, D: 2}

The existing integer Range Coder embeds the (whitened) payload through that
empirical table.  Conditional on a candidate pool, uniformly random source bits
select tokens in proportion to the pool counts.  Averaged over iid pools, the
expected token distribution is therefore the original active LLM distribution.
This is a steganographic adaptation of SynthID-Text's quality-first,
non-distortionary candidate-sampling principle; it is not Google's watermark.

The previous v1 experiment let ``b`` bits select one of ``2**b`` candidate
*slots* and treated duplicated selected tokens as erasures.  Peaked small-model
distributions collide often, so that construction could fall below one useful
bit per token and exhaust Qwen3-1.7B's 2048-token context.  Empirical Range
Coding recovers the entropy represented by those collisions instead of throwing
it away.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import Counter
from collections.abc import Sequence
from struct import Struct

from lsteg.coding.frequencies import FrequencyTable
from lsteg.coding.range_coder import CodedBits, RangeDecoder, RangeEncoder
from lsteg.model.errors import ModelInputError
from lsteg.model.incremental import start_incremental_logits
from lsteg.model.interface import LanguageModelBackend
from lsteg.steg.engine import (
    CoverTokenError,
    InsufficientCoverCapacityError,
    SteganographyConfig,
    _maybe_semantic_reanchor,
    _position_table,
    _validate_max_tokens,
)
from lsteg.steg.semantic import SemanticAnchorPlan

_DOMAIN = b"llm-steganography/v2/empirical-tournament-pool"
_WHITEN_DOMAIN = b"llm-steganography/v2/tournament-payload-whitening"
_UINT32 = Struct(">I")
_UINT64 = Struct(">Q")
_DEFAULT_CONTEXT_WINDOW = 4
_DEFAULT_POOL_BITS = 4


def hide_bytes_tournament(
    backend: LanguageModelBackend,
    prompt: str,
    payload: bytes,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    semantic_plan: SemanticAnchorPlan | None = None,
    bits_per_token: int = _DEFAULT_POOL_BITS,
    context_window: int = _DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = 4096,
) -> list[int]:
    """Embed bytes through iid candidate-pool empirical distributions.

    ``bits_per_token`` is retained as the public API name for compatibility
    with the v1 experiment, but in v2 it means ``log2(candidate_pool_size)``;
    it is no longer a fixed number of payload bits consumed per token.
    Actual capacity is the entropy of the empirical candidate-count table.
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return []
    _validate_channel_parameters(bits_per_token, context_window)
    _validate_max_tokens(max_tokens)
    if not stego_key:
        raise ValueError("stego_key must not be empty")

    cfg = config or SteganographyConfig()
    target_bits = len(payload) * 8
    whitened = _xor_whiten(payload, stego_key)
    terminated = CodedBits(whitened + b"\x80", target_bits + 1)
    decoder = RangeDecoder(terminated)
    mirror = RangeEncoder()

    context_ids = backend.tokenize(prompt)
    session = start_incremental_logits(backend, context_ids)
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
    cover_tokens: list[int] = []

    for position in range(max_tokens):
        context_ids, session = _maybe_semantic_reanchor(
            backend,
            semantic_state,
            cover_tokens,
            position,
            context_ids,
            session,
        )
        base_ids, base_table = _position_table(
            backend,
            session,
            context_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
            cover_token_ids=cover_tokens,
        )
        pool_ids, pool_table = _empirical_pool_table(
            base_ids,
            base_table.frequencies,
            base_table.total,
            stego_key=stego_key,
            position=position,
            cover_prefix=cover_tokens,
            pool_bits=bits_per_token,
            context_window=context_window,
        )

        symbol_index = decoder.decode(pool_table)
        mirror.encode(pool_table, symbol_index)
        token_id = pool_ids[symbol_index]
        context_ids.append(token_id)
        cover_tokens.append(token_id)

        if mirror.settled_bit_length >= target_bits:
            recovered_whitened = mirror.settled_bits().data[: len(payload)]
            if recovered_whitened != whitened:  # pragma: no cover - coder invariant
                raise RuntimeError("empirical tournament mapping changed payload prefix")
            return cover_tokens
        try:
            session.append(token_id)
        except ModelInputError as error:
            raise InsufficientCoverCapacityError(
                "empirical tournament channel reached the model context limit after "
                f"settling {mirror.settled_bit_length} of {target_bits} payload bits "
                f"in {len(cover_tokens)} cover tokens"
            ) from error

    raise InsufficientCoverCapacityError(
        "empirical tournament channel settled "
        f"{mirror.settled_bit_length} of {target_bits} payload bits "
        f"within max_tokens={max_tokens}"
    )


def extract_bytes_tournament(
    backend: LanguageModelBackend,
    prompt: str,
    cover_tokens: Sequence[int],
    payload_size: int,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    semantic_plan: SemanticAnchorPlan | None = None,
    bits_per_token: int = _DEFAULT_POOL_BITS,
    context_window: int = _DEFAULT_CONTEXT_WINDOW,
) -> bytes:
    """Recover bytes emitted by :func:`hide_bytes_tournament`."""
    if isinstance(payload_size, bool) or not isinstance(payload_size, int):
        raise TypeError("payload_size must be int")
    if payload_size < 0:
        raise ValueError("payload_size must not be negative")
    if payload_size == 0:
        if cover_tokens:
            raise ValueError("empty payload must use an empty cover token sequence")
        return b""
    _validate_channel_parameters(bits_per_token, context_window)
    if not stego_key:
        raise ValueError("stego_key must not be empty")

    cfg = config or SteganographyConfig()
    required_bits = payload_size * 8
    encoder = RangeEncoder()
    context_ids = backend.tokenize(prompt)
    session = start_incremental_logits(backend, context_ids)
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
    transport_prefix: list[int] = []

    for position, token_id in enumerate(cover_tokens):
        context_ids, session = _maybe_semantic_reanchor(
            backend,
            semantic_state,
            transport_prefix,
            position,
            context_ids,
            session,
        )
        base_ids, base_table = _position_table(
            backend,
            session,
            context_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
            cover_token_ids=transport_prefix,
        )
        pool_ids, pool_table = _empirical_pool_table(
            base_ids,
            base_table.frequencies,
            base_table.total,
            stego_key=stego_key,
            position=position,
            cover_prefix=transport_prefix,
            pool_bits=bits_per_token,
            context_window=context_window,
        )
        try:
            symbol_index = pool_ids.index(token_id)
        except ValueError as error:
            raise CoverTokenError(
                f"cover token {token_id!r} is not in the keyed empirical candidate pool "
                f"at cover position {position}"
            ) from error

        encoder.encode(pool_table, symbol_index)
        context_ids.append(token_id)
        transport_prefix.append(token_id)
        if encoder.settled_bit_length >= required_bits:
            whitened = encoder.settled_bits().data[:payload_size]
            return _xor_whiten(whitened, stego_key)
        if position + 1 < len(cover_tokens):
            try:
                session.append(token_id)
            except ModelInputError as error:
                raise InsufficientCoverCapacityError(
                    "empirical tournament decoder reached the model context limit after "
                    f"settling {encoder.settled_bit_length} of {required_bits} required bits"
                ) from error

    raise InsufficientCoverCapacityError(
        "empirical tournament cover settled "
        f"{encoder.settled_bit_length} of {required_bits} required bits"
    )


def _empirical_pool_table(
    token_ids: Sequence[int],
    frequencies: Sequence[int],
    total: int,
    *,
    stego_key: bytes,
    position: int,
    cover_prefix: Sequence[int],
    pool_bits: int,
    context_window: int,
) -> tuple[list[int], FrequencyTable]:
    """Draw an iid pool and collapse duplicate slots into integer counts."""
    pool = _candidate_pool(
        token_ids,
        frequencies,
        total,
        stego_key=stego_key,
        position=position,
        cover_prefix=cover_prefix,
        pool_bits=pool_bits,
        context_window=context_window,
    )
    counts = Counter(pool)
    # First-occurrence order is deterministic from the reconstructed pool and
    # keeps this helper independent of tokenizer/token-id semantics.
    unique_ids = list(counts)
    return unique_ids, FrequencyTable([counts[token_id] for token_id in unique_ids])


def _candidate_pool(
    token_ids: Sequence[int],
    frequencies: Sequence[int],
    total: int,
    *,
    stego_key: bytes,
    position: int,
    cover_prefix: Sequence[int],
    pool_bits: int,
    context_window: int,
) -> list[int]:
    if len(token_ids) != len(frequencies):
        raise ValueError("token and frequency counts must match")
    if not token_ids or total <= 0:
        raise ValueError("candidate distribution must not be empty")
    cumulative: list[int] = []
    running = 0
    for frequency in frequencies:
        running += int(frequency)
        cumulative.append(running)
    if running != total:
        raise ValueError("frequency total mismatch")

    pool_size = 1 << pool_bits
    context_digest = _window_digest(cover_prefix, context_window)
    return [
        token_ids[
            _symbol_for_draw(
                cumulative,
                _uniform_draw(
                    stego_key,
                    total,
                    position=position,
                    slot=slot,
                    context_digest=context_digest,
                ),
            )
        ]
        for slot in range(pool_size)
    ]


def _xor_whiten(payload: bytes, stego_key: bytes) -> bytes:
    """XOR payload bytes with a domain-separated HMAC-CTR pseudorandom pad.

    The secure frame contains a small deterministic header before its random
    nonce/ciphertext.  Whitening makes those transport bits pseudorandom from
    the stego mapper's point of view without adding bytes or changing AEAD.
    """
    if not payload:
        return b""
    output = bytearray(len(payload))
    offset = 0
    counter = 0
    while offset < len(payload):
        block = hmac.digest(
            stego_key,
            _WHITEN_DOMAIN + _UINT32.pack(counter),
            "sha256",
        )
        take = min(len(block), len(payload) - offset)
        for index in range(take):
            output[offset + index] = payload[offset + index] ^ block[index]
        offset += take
        counter += 1
    return bytes(output)


def _uniform_draw(
    stego_key: bytes,
    total: int,
    *,
    position: int,
    slot: int,
    context_digest: bytes,
) -> int:
    """Return an unbiased deterministic draw in ``range(total)`` via HMAC."""
    if not stego_key:
        raise ValueError("stego_key must not be empty")
    if total <= 0:
        raise ValueError("total must be positive")
    limit = ((1 << 64) // total) * total
    retry = 0
    while True:
        message = (
            _DOMAIN
            + _UINT64.pack(position)
            + _UINT32.pack(slot)
            + _UINT32.pack(retry)
            + context_digest
        )
        value = int.from_bytes(hmac.digest(stego_key, message, "sha256")[:8], "big")
        if value < limit:
            return value % total
        retry += 1


def _window_digest(token_ids: Sequence[int], context_window: int) -> bytes:
    digest = hashlib.sha256()
    window = token_ids[-context_window:] if context_window else []
    digest.update(_UINT32.pack(len(window)))
    for token_id in window:
        digest.update(_UINT32.pack(int(token_id)))
    return digest.digest()


def _symbol_for_draw(cumulative: Sequence[int], draw: int) -> int:
    # Candidate alphabets are small (typically <= 256), so a simple scan keeps
    # this model-neutral and deterministic without another dependency.
    for index, upper in enumerate(cumulative):
        if draw < upper:
            return index
    raise RuntimeError("draw fell outside cumulative frequency table")


def _validate_channel_parameters(pool_bits: int, context_window: int) -> None:
    if isinstance(pool_bits, bool) or not isinstance(pool_bits, int):
        raise TypeError("bits_per_token must be int")
    if not 1 <= pool_bits <= 8:
        raise ValueError("bits_per_token/pool_bits must satisfy 1 <= value <= 8")
    if isinstance(context_window, bool) or not isinstance(context_window, int):
        raise TypeError("context_window must be int")
    if not 1 <= context_window <= 32:
        raise ValueError("context_window must satisfy 1 <= context_window <= 32")
