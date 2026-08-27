from __future__ import annotations

import pytest

from lsteg.coding.frequencies import FrequencyTable
from lsteg.steg.repetition import filter_no_repeat_ngram_candidates


def test_no_repeat_ngram_blocks_only_candidate_that_recreates_seen_ngram() -> None:
    ids, table = filter_no_repeat_ngram_candidates(
        [1, 2, 3, 1, 2],
        [3, 4, 5],
        FrequencyTable([30, 20, 10]),
        ngram_size=3,
    )
    assert ids == [4, 5]
    assert table.frequencies == (20, 10)


def test_no_repeat_ngram_is_disabled_at_zero() -> None:
    original = FrequencyTable([3, 2])
    ids, table = filter_no_repeat_ngram_candidates([1, 1], [1, 2], original, ngram_size=0)
    assert ids == [1, 2]
    assert table is original


def test_no_repeat_ngram_fails_open_if_every_candidate_is_blocked() -> None:
    original = FrequencyTable([7])
    ids, table = filter_no_repeat_ngram_candidates([1, 1], [1], original, ngram_size=2)
    assert ids == [1]
    assert table is original


def test_no_repeat_ngram_rejects_one() -> None:
    with pytest.raises(ValueError):
        filter_no_repeat_ngram_candidates([1], [2], FrequencyTable([1]), ngram_size=1)
