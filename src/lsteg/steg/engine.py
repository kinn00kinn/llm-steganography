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
from lsteg.model.errors import ModelInputError
from lsteg.model.incremental import start_incremental_logits
from lsteg.model.interface import (
    IncrementalLogitsSession,
    LanguageModelBackend,
    RankedIncrementalLogitsSession,
)
from lsteg.steg.frequencies import (
    DEFAULT_TOP_K,
    FREQUENCY_TOTAL,
    frequency_table_entropy,
    logits_to_frequency_table,
    ranked_logits_to_frequency_table,
)
from lsteg.steg.mapping import keyed_candidate_permutation
from lsteg.steg.repetition import filter_no_repeat_ngram_candidates
from lsteg.steg.semantic import SemanticAnchorPlan, SemanticAnchorState
from lsteg.steg.transport import (
    NoTransportSafeCandidatesError,
    filter_transport_safe_candidates,
)


class CoverTokenError(RuntimeError):
    """A received cover token is not in the active candidate alphabet."""


@dataclass(frozen=True, slots=True)
class CoverCapacityDiagnostics:
    """Structured evidence attached to byte-channel capacity failures."""

    reason: str
    target_bits: int
    settled_bits: int
    generated_tokens: int
    mean_table_entropy: float
    mean_selected_surprisal: float
    semantic_phase: int | None = None
    semantic_phase_count: int | None = None

    def format(self) -> str:
        phase_detail = ""
        if self.semantic_phase is not None and self.semantic_phase_count is not None:
            phase_detail = f", semantic_phase={self.semantic_phase}/{self.semantic_phase_count}"
        return (
            f"{self.reason}; payload settled {self.settled_bits} of {self.target_bits} bits "
            f"after {self.generated_tokens} cover tokens; "
            f"mean_table_entropy={self.mean_table_entropy:.3f} bit/token, "
            f"mean_selected_surprisal={self.mean_selected_surprisal:.3f} bit/token"
            f"{phase_detail}"
        )


class InsufficientCoverCapacityError(RuntimeError):
    """The configured/model context budget ended before the payload settled."""

    diagnostics: CoverCapacityDiagnostics | None

    def __init__(
        self,
        message: str,
        *,
        diagnostics: CoverCapacityDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class SteganographyConfig:
    """Shared parameters that sender and receiver must reproduce exactly."""

    top_k: int = DEFAULT_TOP_K
    frequency_total: int = FREQUENCY_TOTAL
    temperature: float = 1.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    excluded_token_ids: tuple[int, ...] = ()
    enforce_transport_invariance: bool = False
    no_repeat_ngram_size: int = 0

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
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
            raise TypeError("top_p must be a number")
        top_p = float(self.top_p)
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        object.__setattr__(self, "top_p", top_p)
        if isinstance(self.presence_penalty, bool) or not isinstance(
            self.presence_penalty, (int, float)
        ):
            raise TypeError("presence_penalty must be a number")
        presence_penalty = float(self.presence_penalty)
        if not math.isfinite(presence_penalty) or not 0.0 <= presence_penalty <= 2.0:
            raise ValueError("presence_penalty must satisfy 0 <= presence_penalty <= 2")
        object.__setattr__(self, "presence_penalty", presence_penalty)
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
        if not isinstance(self.enforce_transport_invariance, bool):
            raise TypeError("enforce_transport_invariance must be bool")
        if isinstance(self.no_repeat_ngram_size, bool) or not isinstance(
            self.no_repeat_ngram_size, int
        ):
            raise TypeError("no_repeat_ngram_size must be int")
        if self.no_repeat_ngram_size != 0 and self.no_repeat_ngram_size < 2:
            raise ValueError("no_repeat_ngram_size must be 0 or at least 2")


def hide(
    backend: LanguageModelBackend,
    prompt: str,
    coded_bits: CodedBits,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    semantic_plan: SemanticAnchorPlan | None = None,
    max_tokens: int = 4096,
) -> list[int]:
    """Embed a canonical ``CodedBits`` stream into generated cover token IDs."""
    if coded_bits.bit_length == 0:
        return []
    _validate_max_tokens(max_tokens)
    cfg = config or SteganographyConfig()
    token_ids = backend.tokenize(prompt)
    logit_session = start_incremental_logits(backend, token_ids)
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
    decoder = RangeDecoder(coded_bits)
    cover_tokens: list[int] = []
    needed_bits = coded_bits.bit_length + STATE_BITS

    for position in range(max_tokens):
        token_ids, logit_session = _maybe_semantic_reanchor(
            backend, semantic_state, cover_tokens, position, token_ids, logit_session
        )
        top_ids, table = _position_table(
            backend,
            logit_session,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
            cover_token_ids=cover_tokens,
        )
        symbol_index = decoder.decode(table)
        token_id = top_ids[symbol_index]
        token_ids.append(token_id)
        cover_tokens.append(token_id)
        if decoder.input_bits_read >= needed_bits:
            return cover_tokens
        logit_session.append(token_id)

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
    semantic_plan: SemanticAnchorPlan | None = None,
) -> CodedBits:
    """Recover a canonical coded-bit representation from received cover tokens."""
    cfg = config or SteganographyConfig()
    token_ids = backend.tokenize(prompt)
    logit_session = start_incremental_logits(backend, token_ids)
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
    encoder = RangeEncoder()
    transport_prefix: list[int] = []

    for position, token_id in enumerate(cover_tokens):
        token_ids, logit_session = _maybe_semantic_reanchor(
            backend, semantic_state, transport_prefix, position, token_ids, logit_session
        )
        top_ids, table = _position_table(
            backend,
            logit_session,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
            cover_token_ids=transport_prefix,
        )
        symbol_index = _symbol_index(top_ids, token_id, position, cfg.top_k)
        encoder.encode(table, symbol_index)
        token_ids.append(token_id)
        transport_prefix.append(token_id)
        if position + 1 < len(cover_tokens):
            logit_session.append(token_id)

    return encoder.finish()


def hide_bytes(
    backend: LanguageModelBackend,
    prompt: str,
    payload: bytes,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    semantic_plan: SemanticAnchorPlan | None = None,
    max_tokens: int = 4096,
) -> list[int]:
    """Embed arbitrary payload bytes using the LLM's dynamic frequency tables.

    A single termination bit is appended to the arithmetic-decoder input. A
    mirror encoder tracks how many payload-prefix bits are irrevocably settled;
    generation stops as soon as all payload bits are recoverable. Capacity
    failures carry structured diagnostics instead of leaking raw model-context
    errors.
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
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
    cover_tokens: list[int] = []
    entropy_sum = 0.0
    selected_surprisal_sum = 0.0

    try:
        token_ids = backend.tokenize(prompt)
        logit_session = start_incremental_logits(backend, token_ids)
    except ModelInputError as error:
        raise _capacity_error(
            target_bits=target_bits,
            settled_bits=0,
            generated_tokens=0,
            entropy_sum=0.0,
            selected_surprisal_sum=0.0,
            semantic_state=semantic_state,
            reason=f"initial model context rejected: {error}",
        ) from error

    for position in range(max_tokens):
        try:
            token_ids, logit_session = _maybe_semantic_reanchor(
                backend,
                semantic_state,
                cover_tokens,
                position,
                token_ids,
                logit_session,
            )
        except ModelInputError as error:
            raise _capacity_error(
                target_bits=target_bits,
                settled_bits=mirror.settled_bit_length,
                generated_tokens=len(cover_tokens),
                entropy_sum=entropy_sum,
                selected_surprisal_sum=selected_surprisal_sum,
                semantic_state=semantic_state,
                reason=f"semantic re-anchor context rejected: {error}",
            ) from error

        top_ids, table = _position_table(
            backend,
            logit_session,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
            cover_token_ids=cover_tokens,
        )
        entropy_sum += frequency_table_entropy(table)
        symbol_index = decoder.decode(table)
        probability = table.frequencies[symbol_index] / table.total
        selected_surprisal_sum += -math.log2(probability)
        mirror.encode(table, symbol_index)
        token_id = top_ids[symbol_index]
        token_ids.append(token_id)
        cover_tokens.append(token_id)

        if mirror.settled_bit_length >= target_bits:
            recovered_prefix = mirror.settled_bits().data[: len(payload)]
            if recovered_prefix != payload:  # pragma: no cover - coder symmetry invariant
                raise RuntimeError("range mapping failed to preserve the payload prefix")
            return cover_tokens

        try:
            logit_session.append(token_id)
        except ModelInputError as error:
            raise _capacity_error(
                target_bits=target_bits,
                settled_bits=mirror.settled_bit_length,
                generated_tokens=len(cover_tokens),
                entropy_sum=entropy_sum,
                selected_surprisal_sum=selected_surprisal_sum,
                semantic_state=semantic_state,
                reason=f"model context limit reached: {error}",
            ) from error

    raise _capacity_error(
        target_bits=target_bits,
        settled_bits=mirror.settled_bit_length,
        generated_tokens=len(cover_tokens),
        entropy_sum=entropy_sum,
        selected_surprisal_sum=selected_surprisal_sum,
        semantic_state=semantic_state,
        reason=f"max_tokens={max_tokens} exhausted",
    )


def extract_bytes(
    backend: LanguageModelBackend,
    prompt: str,
    cover_tokens: Sequence[int],
    payload_size: int,
    *,
    stego_key: bytes,
    config: SteganographyConfig | None = None,
    semantic_plan: SemanticAnchorPlan | None = None,
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
    logit_session = start_incremental_logits(backend, token_ids)
    semantic_state = semantic_plan.new_state() if semantic_plan is not None else None
    encoder = RangeEncoder()
    transport_prefix: list[int] = []

    for position, token_id in enumerate(cover_tokens):
        token_ids, logit_session = _maybe_semantic_reanchor(
            backend, semantic_state, transport_prefix, position, token_ids, logit_session
        )
        top_ids, table = _position_table(
            backend,
            logit_session,
            token_ids,
            position=position,
            stego_key=stego_key,
            config=cfg,
            cover_token_ids=transport_prefix,
        )
        symbol_index = _symbol_index(top_ids, token_id, position, cfg.top_k)
        encoder.encode(table, symbol_index)
        token_ids.append(token_id)
        transport_prefix.append(token_id)
        if encoder.settled_bit_length >= required_bits:
            return encoder.settled_bits().data[:payload_size]
        logit_session.append(token_id)

    raise InsufficientCoverCapacityError(
        f"cover settled {encoder.settled_bit_length} of {required_bits} required bits"
    )


def _maybe_semantic_reanchor(
    backend: LanguageModelBackend,
    semantic_state: SemanticAnchorState | None,
    cover_token_ids: Sequence[int],
    position: int,
    context_token_ids: list[int],
    logit_session: IncrementalLogitsSession,
) -> tuple[list[int], IncrementalLogitsSession]:
    """Restart the hidden model context at deterministic semantic boundaries."""
    if semantic_state is None:
        return context_token_ids, logit_session

    replacement_prompt = semantic_state.maybe_advance(backend, cover_token_ids, position=position)
    if replacement_prompt is None:
        return context_token_ids, logit_session
    replacement_ids = backend.tokenize(replacement_prompt)
    return replacement_ids, start_incremental_logits(backend, replacement_ids)


def _position_table(
    backend: LanguageModelBackend,
    logit_session: IncrementalLogitsSession,
    context_token_ids: Sequence[int],
    *,
    position: int,
    stego_key: bytes,
    config: SteganographyConfig,
    cover_token_ids: Sequence[int],
) -> tuple[list[int], FrequencyTable]:
    ranked = None
    logits = None
    if isinstance(logit_session, RankedIncrementalLogitsSession):
        ranked = logit_session.top_logits(
            config.top_k,
            excluded_token_ids=config.excluded_token_ids,
            penalized_token_ids=cover_token_ids,
            presence_penalty=config.presence_penalty,
        )
    else:
        logits = logit_session.next_logits()

    def build_table(top_p: float) -> tuple[list[int], FrequencyTable]:
        if ranked is not None:
            return ranked_logits_to_frequency_table(
                ranked,
                total=config.frequency_total,
                temperature=config.temperature,
                top_p=top_p,
            )
        assert logits is not None
        return logits_to_frequency_table(
            logits,
            top_k=config.top_k,
            total=config.frequency_total,
            temperature=config.temperature,
            top_p=top_p,
            excluded_token_ids=config.excluded_token_ids,
            penalized_token_ids=cover_token_ids,
            presence_penalty=config.presence_penalty,
        )

    top_ids, table = build_table(config.top_p)

    # Unicode transport is a hard correctness constraint; anti-repetition is a
    # soft naturalness heuristic.  Apply the hard filter first so the
    # no-repeat filter can deliberately fail open when *all transport-safe*
    # continuations would repeat an n-gram.  The opposite order can leave only
    # transport-unsafe tokens and incorrectly report an empty alphabet.
    if config.enforce_transport_invariance:
        try:
            top_ids, table = filter_transport_safe_candidates(
                backend,
                cover_token_ids,
                top_ids,
                table,
            )
        except NoTransportSafeCandidatesError:
            # A narrow nucleus can occasionally contain no canonical Unicode
            # continuation even though the configured top-k does.  Widen only
            # this position to the full top-k alphabet; sender and receiver
            # reproduce the same deterministic fallback from the visible
            # prefix.  Transport safety itself is never relaxed.
            if config.top_p >= 1.0:
                raise
            top_ids, table = build_table(1.0)
            top_ids, table = filter_transport_safe_candidates(
                backend,
                cover_token_ids,
                top_ids,
                table,
            )

    if config.no_repeat_ngram_size:
        top_ids, table = filter_no_repeat_ngram_candidates(
            cover_token_ids,
            top_ids,
            table,
            ngram_size=config.no_repeat_ngram_size,
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


def _capacity_error(
    *,
    target_bits: int,
    settled_bits: int,
    generated_tokens: int,
    entropy_sum: float,
    selected_surprisal_sum: float,
    semantic_state: SemanticAnchorState | None,
    reason: str,
) -> InsufficientCoverCapacityError:
    denominator = max(generated_tokens, 1)
    semantic_phase = None
    semantic_phase_count = None
    if semantic_state is not None:
        semantic_phase = semantic_state.phase_index + 1
        semantic_phase_count = len(semantic_state.plan.phases)
    diagnostics = CoverCapacityDiagnostics(
        reason=reason,
        target_bits=target_bits,
        settled_bits=settled_bits,
        generated_tokens=generated_tokens,
        mean_table_entropy=entropy_sum / denominator,
        mean_selected_surprisal=selected_surprisal_sum / denominator,
        semantic_phase=semantic_phase,
        semantic_phase_count=semantic_phase_count,
    )
    return InsufficientCoverCapacityError(diagnostics.format(), diagnostics=diagnostics)


def _validate_max_tokens(max_tokens: int) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be int")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
