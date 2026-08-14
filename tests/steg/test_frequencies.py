"""Unit tests for lsteg.steg.frequencies.logits_to_frequency_table."""

from __future__ import annotations

import random

import pytest

from lsteg.coding.frequencies import MAX_FREQUENCY_TOTAL, FrequencyTable
from lsteg.steg.frequencies import (
    DEFAULT_TOP_K,
    FREQUENCY_TOTAL,
    logits_to_frequency_table,
)


def _uniform_logits(vocab_size: int) -> list[float]:
    return [0.0] * vocab_size


def _peaked_logits(vocab_size: int, peak_token: int = 0, scale: float = 10.0) -> list[float]:
    logits = [0.0] * vocab_size
    logits[peak_token] = scale
    return logits


class TestLogitsToFrequencyTable:
    def test_returns_correct_types(self) -> None:
        top_ids, table = logits_to_frequency_table(_uniform_logits(1000))
        assert isinstance(top_ids, list)
        assert all(isinstance(i, int) for i in top_ids)
        assert isinstance(table, FrequencyTable)

    def test_table_total_equals_frequency_total(self) -> None:
        _, table = logits_to_frequency_table(_uniform_logits(1000))
        assert table.total == FREQUENCY_TOTAL

    def test_table_length_equals_top_k(self) -> None:
        top_k = 128
        top_ids, table = logits_to_frequency_table(_uniform_logits(500), top_k=top_k)
        assert len(top_ids) == top_k
        assert table.symbol_count == top_k

    def test_top_ids_length_equals_top_k(self) -> None:
        top_ids, _ = logits_to_frequency_table(_uniform_logits(1000))
        assert len(top_ids) == DEFAULT_TOP_K

    def test_all_top_ids_are_valid_token_ids(self) -> None:
        vocab_size = 500
        top_ids, _ = logits_to_frequency_table(_uniform_logits(vocab_size), top_k=50)
        assert all(0 <= t < vocab_size for t in top_ids)

    def test_top_ids_are_unique(self) -> None:
        top_ids, _ = logits_to_frequency_table(_uniform_logits(1000))
        assert len(top_ids) == len(set(top_ids))

    def test_no_zero_frequency_in_table(self) -> None:
        _, table = logits_to_frequency_table(_uniform_logits(1000))
        assert all(table.frequency(i) > 0 for i in range(table.symbol_count))

    def test_peaked_logit_selects_correct_top_token(self) -> None:
        """The highest-logit token must appear first in top_ids."""
        vocab_size = 1000
        peak = 42
        top_ids, _ = logits_to_frequency_table(_peaked_logits(vocab_size, peak), top_k=10)
        assert top_ids[0] == peak

    def test_peaked_logit_highest_frequency_goes_to_peak(self) -> None:
        """The highest-probability token must receive the most frequency units."""
        vocab_size = 1000
        peak = 7
        top_ids, table = logits_to_frequency_table(
            _peaked_logits(vocab_size, peak, scale=20.0), top_k=10
        )
        assert top_ids[0] == peak
        # Symbol 0 in the table corresponds to the top token.
        assert table.frequency(0) == max(table.frequency(i) for i in range(table.symbol_count))

    def test_deterministic_on_identical_input(self) -> None:
        logits = [random.gauss(0.0, 1.0) for _ in range(500)]
        ids1, table1 = logits_to_frequency_table(logits, top_k=64)
        ids2, table2 = logits_to_frequency_table(logits, top_k=64)
        assert ids1 == ids2
        assert table1.frequencies == table2.frequencies

    def test_custom_total(self) -> None:
        _, table = logits_to_frequency_table(_uniform_logits(500), top_k=16, total=256)
        assert table.total == 256

    def test_top_k_equals_vocab_size(self) -> None:
        vocab_size = 100
        top_ids, table = logits_to_frequency_table(_uniform_logits(vocab_size), top_k=vocab_size)
        assert len(top_ids) == vocab_size
        assert table.total == FREQUENCY_TOTAL

    def test_top_k_of_one(self) -> None:
        top_ids, table = logits_to_frequency_table(_uniform_logits(100), top_k=1, total=1)
        assert len(top_ids) == 1
        assert table.total == 1
        assert table.frequency(0) == 1

    def test_raises_on_top_k_exceeding_vocab(self) -> None:
        with pytest.raises(ValueError, match="vocabulary size"):
            logits_to_frequency_table(_uniform_logits(10), top_k=20)

    def test_raises_on_total_exceeding_max(self) -> None:
        with pytest.raises(ValueError, match="32768"):
            logits_to_frequency_table(_uniform_logits(100), total=MAX_FREQUENCY_TOTAL + 1)

    def test_raises_on_top_k_greater_than_total(self) -> None:
        with pytest.raises(ValueError, match="top_k must satisfy"):
            logits_to_frequency_table(_uniform_logits(100), top_k=100, total=50)

    def test_tie_breaking_by_ascending_token_id(self) -> None:
        """Exactly equal logits must produce stable ordering by token ID."""
        logits = [0.0] * 10
        top_ids, _ = logits_to_frequency_table(logits, top_k=5, total=10)
        assert top_ids == sorted(top_ids)

    def test_frequency_allocation_sum_invariant_random(self) -> None:
        rng = random.Random(2024)
        for _ in range(50):
            vocab_size = rng.randint(10, 300)
            top_k = rng.randint(1, min(vocab_size, 64))
            total = rng.randint(top_k, MAX_FREQUENCY_TOTAL)
            logits = [rng.gauss(0.0, 2.0) for _ in range(vocab_size)]
            _, table = logits_to_frequency_table(logits, top_k=top_k, total=total)
            assert table.total == total


class TestSteganographyConfig:
    def test_default_config(self) -> None:
        from lsteg.steg.engine import SteganographyConfig

        cfg = SteganographyConfig()
        assert cfg.top_k == DEFAULT_TOP_K
        assert cfg.frequency_total == FREQUENCY_TOTAL

    def test_invalid_top_k(self) -> None:
        from lsteg.steg.engine import SteganographyConfig

        with pytest.raises(ValueError, match="top_k must satisfy"):
            SteganographyConfig(top_k=0)

    def test_invalid_frequency_total(self) -> None:
        from lsteg.steg.engine import SteganographyConfig

        with pytest.raises(ValueError, match="frequency_total must be"):
            SteganographyConfig(frequency_total=MAX_FREQUENCY_TOTAL + 1)
