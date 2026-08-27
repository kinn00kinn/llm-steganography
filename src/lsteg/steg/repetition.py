"""Deterministic anti-loop filtering for cover-token candidate alphabets."""

from __future__ import annotations

from collections.abc import Sequence

from lsteg.coding.frequencies import FrequencyTable


def filter_no_repeat_ngram_candidates(
    cover_prefix_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    table: FrequencyTable,
    *,
    ngram_size: int,
) -> tuple[list[int], FrequencyTable]:
    """Remove candidates that would recreate an already-seen token n-gram.

    The filter is intentionally token based rather than word based: sender and
    receiver can reproduce it exactly from the visible cover prefix without a
    language-specific segmenter.  If every active candidate would be blocked,
    the original alphabet is returned so the steganographic channel stays live.
    """
    if len(candidate_token_ids) != table.symbol_count:
        raise ValueError("candidate token count must equal frequency-table symbol count")
    if isinstance(ngram_size, bool) or not isinstance(ngram_size, int):
        raise TypeError("ngram_size must be int")
    if ngram_size == 0:
        return list(candidate_token_ids), table
    if ngram_size < 2:
        raise ValueError("ngram_size must be 0 (disabled) or at least 2")

    prefix = list(cover_prefix_token_ids)
    if len(prefix) + 1 < ngram_size:
        return list(candidate_token_ids), table

    seen = {
        tuple(prefix[start : start + ngram_size]) for start in range(len(prefix) - ngram_size + 1)
    }
    stem = tuple(prefix[-(ngram_size - 1) :])

    kept_ids: list[int] = []
    kept_frequencies: list[int] = []
    for token_id, frequency in zip(candidate_token_ids, table.frequencies, strict=True):
        if (*stem, token_id) in seen:
            continue
        kept_ids.append(token_id)
        kept_frequencies.append(frequency)

    if not kept_ids:
        return list(candidate_token_ids), table
    return kept_ids, FrequencyTable(kept_frequencies)
