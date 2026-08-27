"""Tests for the keyed candidate interval permutation."""

from __future__ import annotations

import pytest

from lsteg.coding.frequencies import FrequencyTable
from lsteg.steg.mapping import keyed_candidate_permutation

_KEY_A = bytes(range(32))
_KEY_B = bytes(reversed(range(32)))


def test_permutation_preserves_token_frequency_pairs() -> None:
    token_ids = [10, 11, 12, 13, 14]
    table = FrequencyTable([100, 50, 25, 10, 5])
    mapped_ids, mapped_table = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=3,
        context_token_ids=[1, 2, 3],
    )
    original = dict(zip(token_ids, table.frequencies, strict=True))
    mapped = dict(zip(mapped_ids, mapped_table.frequencies, strict=True))
    assert mapped == original
    assert mapped_table.total == table.total


def test_same_inputs_are_deterministic() -> None:
    token_ids = [10, 11, 12, 13]
    table = FrequencyTable([4, 3, 2, 1])
    ids_1, table_1 = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=7,
        context_token_ids=[100, 200],
    )
    ids_2, table_2 = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=7,
        context_token_ids=[100, 200],
    )
    assert ids_1 == ids_2
    assert table_1.frequencies == table_2.frequencies


def test_key_changes_ordering() -> None:
    token_ids = list(range(64))
    table = FrequencyTable([1] * 64)
    ids_a, _ = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=0,
        context_token_ids=[1, 2, 3],
    )
    ids_b, _ = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_B,
        position=0,
        context_token_ids=[1, 2, 3],
    )
    assert ids_a != ids_b


def test_context_and_position_change_ordering() -> None:
    token_ids = list(range(64))
    table = FrequencyTable([1] * 64)
    base, _ = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=0,
        context_token_ids=[1, 2, 3],
    )
    changed_position, _ = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=1,
        context_token_ids=[1, 2, 3],
    )
    changed_context, _ = keyed_candidate_permutation(
        token_ids,
        table,
        stego_key=_KEY_A,
        position=0,
        context_token_ids=[1, 2, 4],
    )
    assert base != changed_position
    assert base != changed_context


def test_invalid_key_length_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        keyed_candidate_permutation(
            [1, 2],
            FrequencyTable([1, 1]),
            stego_key=b"short",
            position=0,
            context_token_ids=[1],
        )
