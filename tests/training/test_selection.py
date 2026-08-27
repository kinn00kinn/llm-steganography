from __future__ import annotations

import pytest

from lsteg.training.selection import select_kl_rows


def test_kl_selection_is_deterministic_sorted_and_bounded() -> None:
    first = select_kl_rows(100, 8, seed=42, example_key="abc")
    second = select_kl_rows(100, 8, seed=42, example_key="abc")
    assert first == second
    assert len(first) == 8
    assert tuple(sorted(first)) == first
    assert all(0 <= row < 100 for row in first)


def test_kl_selection_uses_all_rows_when_short() -> None:
    assert select_kl_rows(3, 8, seed=1, example_key="short") == (0, 1, 2)


def test_kl_selection_rejects_invalid_count() -> None:
    with pytest.raises(ValueError):
        select_kl_rows(0, 1, seed=1, example_key="x")
    with pytest.raises(ValueError):
        select_kl_rows(3, -1, seed=1, example_key="x")
