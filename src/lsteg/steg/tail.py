"""Natural, non-payload tail generation for a finished stego cover."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from lsteg.coding.frequencies import FrequencyTable
from lsteg.model.errors import ModelInputError
from lsteg.model.incremental import start_incremental_logits
from lsteg.model.interface import (
    LanguageModelBackend,
    Logits,
    RankedIncrementalLogitsSession,
    RankedLogits,
)
from lsteg.steg.engine import SteganographyConfig
from lsteg.steg.frequencies import (
    logits_to_frequency_table,
    ranked_logits_to_frequency_table,
)
from lsteg.steg.repetition import filter_no_repeat_ngram_candidates
from lsteg.steg.semantic import SemanticAnchorPlan
from lsteg.steg.transport import (
    NoTransportSafeCandidatesError,
    filter_transport_safe_candidates,
)

_SENTENCE_ENDINGS = ("。", "！", "？", ".", "!", "?")  # noqa: RUF001


def append_sentence_tail(
    backend: LanguageModelBackend,
    prompt: str,
    cover_tokens: Sequence[int],
    *,
    seed_key: bytes,
    config: SteganographyConfig,
    semantic_plan: SemanticAnchorPlan | None = None,
    max_tail_tokens: int = 64,
) -> list[int]:
    """Continue ordinary sampling until the cover reaches sentence punctuation.

    Tail tokens carry no payload. If the model context or the transport-safe
    candidate set is exhausted after the payload has already settled, tail
    generation fails soft and returns the payload cover unchanged/partially
    tailed rather than turning a successful secret round-trip into an exception.
    """
    if not isinstance(seed_key, bytes) or not seed_key:
        raise ValueError("seed_key must be non-empty bytes")
    if isinstance(max_tail_tokens, bool) or not isinstance(max_tail_tokens, int):
        raise TypeError("max_tail_tokens must be int")
    if max_tail_tokens < 0:
        raise ValueError("max_tail_tokens must not be negative")

    generated = list(cover_tokens)
    payload_token_count = len(generated)
    if _ends_sentence(backend.detokenize(generated)):
        return generated

    try:
        if semantic_plan is None:
            context = backend.tokenize(prompt) + generated
        else:
            phase_index = semantic_plan.current_phase_for_cover(backend, generated)
            tail_prompt = semantic_plan.closure_prompt(
                backend,
                generated,
                phase_index=phase_index,
            )
            context = backend.tokenize(tail_prompt)
        logit_session = start_incremental_logits(backend, context)
    except ModelInputError:
        return generated

    for tail_position in range(max_tail_tokens):
        ranked = None
        logits = None
        if isinstance(logit_session, RankedIncrementalLogitsSession):
            ranked = logit_session.top_logits(
                config.top_k,
                excluded_token_ids=config.excluded_token_ids,
                penalized_token_ids=generated,
                presence_penalty=config.presence_penalty,
            )
        else:
            logits = logit_session.next_logits()

        def build_table(
            top_p: float,
            *,
            ranked_logits: RankedLogits | None = ranked,
            full_logits: Logits | None = logits,
        ) -> tuple[list[int], FrequencyTable]:
            if ranked_logits is not None:
                return ranked_logits_to_frequency_table(
                    ranked_logits,
                    total=config.frequency_total,
                    temperature=config.temperature,
                    top_p=top_p,
                )
            assert full_logits is not None
            return logits_to_frequency_table(
                full_logits,
                top_k=config.top_k,
                total=config.frequency_total,
                temperature=config.temperature,
                top_p=top_p,
                excluded_token_ids=config.excluded_token_ids,
                penalized_token_ids=generated,
                presence_penalty=config.presence_penalty,
            )

        top_ids, table = build_table(config.top_p)

        # Transport is a correctness constraint; anti-repeat is only a quality
        # heuristic. Match the payload channel's ordering and deterministic
        # nucleus fallback so tail generation cannot recreate the old
        # "safe token removed first" failure.
        if config.enforce_transport_invariance:
            try:
                top_ids, table = filter_transport_safe_candidates(
                    backend,
                    generated,
                    top_ids,
                    table,
                )
            except NoTransportSafeCandidatesError:
                if config.top_p >= 1.0:
                    return generated
                top_ids, table = build_table(1.0)
                try:
                    top_ids, table = filter_transport_safe_candidates(
                        backend,
                        generated,
                        top_ids,
                        table,
                    )
                except NoTransportSafeCandidatesError:
                    return generated

        if config.no_repeat_ngram_size:
            top_ids, table = filter_no_repeat_ngram_candidates(
                generated,
                top_ids,
                table,
                ngram_size=config.no_repeat_ngram_size,
            )

        draw = _deterministic_draw(seed_key, generated, tail_position, table.total)
        symbol = table.symbol_for(draw)
        token_id = top_ids[symbol]
        context.append(token_id)
        generated.append(token_id)
        tail_text = backend.detokenize(generated[payload_token_count:])
        if _contains_sentence_ending(tail_text):
            break
        try:
            logit_session.append(token_id)
        except ModelInputError:
            break
    return generated


def _deterministic_draw(
    seed_key: bytes,
    generated: Sequence[int],
    tail_position: int,
    total: int,
) -> int:
    digest = hashlib.sha256()
    digest.update(b"lsteg-natural-tail-v1\x00")
    digest.update(seed_key)
    digest.update(tail_position.to_bytes(4, "big"))
    for token_id in generated[-32:]:
        digest.update(int(token_id).to_bytes(4, "big"))
    return int.from_bytes(digest.digest()[:8], "big") % total


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_ENDINGS)


def _contains_sentence_ending(text: str) -> bool:
    return any(mark in text for mark in _SENTENCE_ENDINGS)
