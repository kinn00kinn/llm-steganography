"""Unicode text-transport constraints for steganographic token alphabets.

A causal language model may emit a token-ID sequence that is *not* the
canonical tokenization of its decoded Unicode text.  Sending only that text
would then change the token IDs at the receiver and desynchronize the Range
Coder.

This module removes candidate tokens that would make the generated cover
prefix non-canonical under ``tokenize(detokenize(...))``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lsteg.coding.frequencies import FrequencyTable
from lsteg.model.interface import LanguageModelBackend


@runtime_checkable
class _BatchTransportBackend(Protocol):
    def transport_safe_candidate_ids(
        self,
        cover_prefix_token_ids: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> list[int]: ...


class NoTransportSafeCandidatesError(RuntimeError):
    """No candidate token preserves the Unicode text transport invariant."""


def filter_transport_safe_candidates(
    backend: LanguageModelBackend,
    cover_prefix_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    table: FrequencyTable,
) -> tuple[list[int], FrequencyTable]:
    """Keep only candidates that preserve canonical tokenization of the cover.

    A candidate ``t`` is safe exactly when::

        tokenize(detokenize(cover_prefix + [t])) == cover_prefix + [t]

    The prompt is deliberately *not* included: only the generated cover text is
    transported between sender and receiver.  Frequencies are subset without
    reallocation, preserving the implemented quantized relative masses before
    the surviving alphabet is renormalized by the Range Coder total.
    """
    if len(candidate_token_ids) != table.symbol_count:
        raise ValueError("candidate token count must equal frequency-table symbol count")

    prefix = list(cover_prefix_token_ids)
    if isinstance(backend, _BatchTransportBackend):
        selected_ids = backend.transport_safe_candidate_ids(prefix, candidate_token_ids)
        safe_set = set(selected_ids)
        safe_ids = []
        safe_frequencies = []
        for token_id, frequency in zip(candidate_token_ids, table.frequencies, strict=True):
            if token_id in safe_set:
                safe_ids.append(token_id)
                safe_frequencies.append(frequency)
    else:
        safe_ids = []
        safe_frequencies = []
        for token_id, frequency in zip(
            candidate_token_ids,
            table.frequencies,
            strict=True,
        ):
            trial_ids = [*prefix, token_id]
            text = backend.detokenize(trial_ids)
            if not text:
                continue
            if backend.tokenize(text) != trial_ids:
                continue
            safe_ids.append(token_id)
            safe_frequencies.append(frequency)

    if not safe_ids:
        raise NoTransportSafeCandidatesError(
            "active top-k alphabet contains no Unicode-transport-safe token"
        )

    return safe_ids, FrequencyTable(safe_frequencies)
