"""Validated cumulative integer frequency tables."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from lsteg.coding.errors import InvalidFrequencyTableError, UnavailableSymbolError

MAX_FREQUENCY_TOTAL = 1 << 15


class FrequencyTable:
    """An immutable symbol table containing only bounded integer frequencies."""

    __slots__ = ("_cumulative", "_frequencies", "_total")

    def __init__(self, frequencies: Sequence[int]) -> None:
        if len(frequencies) == 0:
            raise InvalidFrequencyTableError("frequency table must contain at least one symbol")

        validated: list[int] = []
        cumulative = [0]
        for frequency in frequencies:
            if isinstance(frequency, bool) or not isinstance(frequency, int):
                msg = "frequencies must be integers"
                raise TypeError(msg)
            if frequency < 0:
                raise InvalidFrequencyTableError("frequencies must not be negative")
            validated.append(frequency)
            cumulative.append(cumulative[-1] + frequency)

        total = cumulative[-1]
        if total == 0:
            raise InvalidFrequencyTableError("frequency total must be positive")
        if total > MAX_FREQUENCY_TOTAL:
            raise InvalidFrequencyTableError(
                f"frequency total exceeds {MAX_FREQUENCY_TOTAL}: got {total}"
            )

        self._frequencies = tuple(validated)
        self._cumulative = tuple(cumulative)
        self._total = total

    @classmethod
    def uniform(cls, symbol_count: int) -> FrequencyTable:
        """Create a table assigning frequency one to every symbol."""
        if isinstance(symbol_count, bool) or not isinstance(symbol_count, int):
            msg = "symbol_count must be int"
            raise TypeError(msg)
        if not 1 <= symbol_count <= MAX_FREQUENCY_TOTAL:
            raise InvalidFrequencyTableError(
                f"symbol_count must be between 1 and {MAX_FREQUENCY_TOTAL}"
            )
        return cls((1,) * symbol_count)

    @property
    def symbol_count(self) -> int:
        return len(self._frequencies)

    @property
    def total(self) -> int:
        return self._total

    @property
    def frequencies(self) -> tuple[int, ...]:
        return self._frequencies

    def frequency(self, symbol: int) -> int:
        self._validate_symbol_index(symbol)
        return self._frequencies[symbol]

    def interval(self, symbol: int) -> tuple[int, int]:
        """Return the half-open cumulative interval for a usable symbol."""
        self._validate_symbol_index(symbol)
        if self._frequencies[symbol] == 0:
            raise UnavailableSymbolError(f"symbol has zero frequency: {symbol}")
        return self._cumulative[symbol], self._cumulative[symbol + 1]

    def symbol_for(self, cumulative_value: int) -> int:
        """Find the unique positive-frequency symbol containing a cumulative value."""
        if not 0 <= cumulative_value < self._total:
            raise ValueError(f"cumulative value must be in [0, {self._total})")
        symbol = bisect_right(self._cumulative, cumulative_value) - 1
        if self._frequencies[symbol] == 0:  # pragma: no cover - bisect invariant
            raise RuntimeError("cumulative lookup selected a zero-frequency symbol")
        return symbol

    def _validate_symbol_index(self, symbol: int) -> None:
        if isinstance(symbol, bool) or not isinstance(symbol, int):
            msg = "symbol must be int"
            raise TypeError(msg)
        if not 0 <= symbol < len(self._frequencies):
            raise UnavailableSymbolError(f"symbol is outside the frequency table: {symbol}")
