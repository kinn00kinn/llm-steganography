from __future__ import annotations

import pytest

from lsteg.coding import (
    MAX_FREQUENCY_TOTAL,
    FrequencyTable,
    InvalidFrequencyTableError,
    UnavailableSymbolError,
)


def test_cumulative_intervals_cover_the_total_without_gaps() -> None:
    table = FrequencyTable([40, 0, 30, 20, 10])

    assert table.total == 100
    assert table.symbol_count == 5
    assert table.frequencies == (40, 0, 30, 20, 10)
    assert table.interval(0) == (0, 40)
    assert table.interval(2) == (40, 70)
    assert table.interval(3) == (70, 90)
    assert table.interval(4) == (90, 100)

    assert [table.symbol_for(value) for value in range(100)] == (
        [0] * 40 + [2] * 30 + [3] * 20 + [4] * 10
    )


@pytest.mark.parametrize("frequencies", [[], [0], [0, 0], [1, -1]])
def test_invalid_frequency_values_are_rejected(frequencies: list[int]) -> None:
    with pytest.raises(InvalidFrequencyTableError):
        FrequencyTable(frequencies)


@pytest.mark.parametrize("frequency", [True, 1.0, "1"])
def test_frequencies_must_be_plain_integers(frequency: object) -> None:
    with pytest.raises(TypeError, match="integers"):
        FrequencyTable([frequency])  # type: ignore[list-item]


def test_frequency_total_is_bounded() -> None:
    assert FrequencyTable([MAX_FREQUENCY_TOTAL]).total == MAX_FREQUENCY_TOTAL

    with pytest.raises(InvalidFrequencyTableError, match=str(MAX_FREQUENCY_TOTAL)):
        FrequencyTable([MAX_FREQUENCY_TOTAL, 1])


def test_uniform_table_contract() -> None:
    table = FrequencyTable.uniform(256)

    assert table.total == 256
    assert table.symbol_count == 256
    assert all(frequency == 1 for frequency in table.frequencies)


@pytest.mark.parametrize("count", [0, -1, MAX_FREQUENCY_TOTAL + 1])
def test_uniform_table_rejects_invalid_size(count: int) -> None:
    with pytest.raises(InvalidFrequencyTableError):
        FrequencyTable.uniform(count)


def test_zero_frequency_and_out_of_range_symbols_are_unavailable() -> None:
    table = FrequencyTable([1, 0, 1])

    with pytest.raises(UnavailableSymbolError, match="zero"):
        table.interval(1)
    with pytest.raises(UnavailableSymbolError, match="outside"):
        table.interval(3)


@pytest.mark.parametrize("value", [-1, 2])
def test_cumulative_lookup_is_bounded(value: int) -> None:
    with pytest.raises(ValueError, match="cumulative"):
        FrequencyTable([1, 1]).symbol_for(value)
