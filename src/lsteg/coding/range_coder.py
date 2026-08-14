"""Bitwise 32-bit integer Range Encoder and Decoder."""

from __future__ import annotations

from dataclasses import dataclass, field

from lsteg.coding.errors import MalformedRangeFrameError
from lsteg.coding.frequencies import FrequencyTable

STATE_BITS = 32
_FULL_RANGE = 1 << STATE_BITS
_HALF_RANGE = _FULL_RANGE >> 1
_QUARTER_RANGE = _HALF_RANGE >> 1
_THREE_QUARTER_RANGE = _QUARTER_RANGE * 3
_STATE_MASK = _FULL_RANGE - 1


@dataclass(frozen=True, slots=True)
class CodedBits:
    """A canonical MSB-first bit string and its meaningful bit length."""

    data: bytes = field(repr=False)
    bit_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            msg = "coded bit data must be bytes"
            raise TypeError(msg)
        if isinstance(self.bit_length, bool) or not isinstance(self.bit_length, int):
            msg = "bit_length must be int"
            raise TypeError(msg)
        if self.bit_length < 0:
            raise MalformedRangeFrameError("bit length must not be negative")
        expected_bytes = (self.bit_length + 7) // 8
        if len(self.data) != expected_bytes:
            raise MalformedRangeFrameError(
                f"coded byte length mismatch: expected {expected_bytes}, got {len(self.data)}"
            )
        padding_bits = expected_bytes * 8 - self.bit_length
        if padding_bits and self.data[-1] & ((1 << padding_bits) - 1):
            raise MalformedRangeFrameError("unused coded padding bits must be zero")


class RangeEncoder:
    """Incrementally encode symbols using integer cumulative frequencies."""

    __slots__ = ("_finished", "_high", "_low", "_pending_underflow", "_writer")

    def __init__(self) -> None:
        self._low = 0
        self._high = _STATE_MASK
        self._pending_underflow = 0
        self._writer = _BitWriter()
        self._finished = False

    def encode(self, table: FrequencyTable, symbol: int) -> None:
        """Narrow the current interval for one symbol and emit settled bits."""
        if self._finished:
            raise RuntimeError("range encoder is already finished")
        cumulative_low, cumulative_high = table.interval(symbol)
        interval_size = self._high - self._low + 1
        previous_low = self._low
        self._low = previous_low + cumulative_low * interval_size // table.total
        self._high = previous_low + cumulative_high * interval_size // table.total - 1
        self._renormalize()

    def finish(self) -> CodedBits:
        """Emit the shortest canonical suffix selecting the final interval."""
        if self._finished:
            raise RuntimeError("range encoder is already finished")
        self._finished = True
        self._pending_underflow += 1
        if self._low < _QUARTER_RANGE:
            self._emit_with_underflow(0)
        else:
            self._emit_with_underflow(1)
        return self._writer.finish()

    @property
    def settled_bit_length(self) -> int:
        """Return the number of prefix bits irrevocably selected so far."""
        return self._writer.bit_length

    def settled_bits(self) -> CodedBits:
        """Snapshot settled prefix bits without finalizing the interval."""
        return self._writer.snapshot()

    def _renormalize(self) -> None:
        while True:
            if self._high < _HALF_RANGE:
                self._emit_with_underflow(0)
            elif self._low >= _HALF_RANGE:
                self._emit_with_underflow(1)
                self._low -= _HALF_RANGE
                self._high -= _HALF_RANGE
            elif self._low >= _QUARTER_RANGE and self._high < _THREE_QUARTER_RANGE:
                self._pending_underflow += 1
                self._low -= _QUARTER_RANGE
                self._high -= _QUARTER_RANGE
            else:
                return
            self._low = (self._low << 1) & _STATE_MASK
            self._high = ((self._high << 1) & _STATE_MASK) | 1

    def _emit_with_underflow(self, bit: int) -> None:
        self._writer.write(bit)
        inverse = bit ^ 1
        for _ in range(self._pending_underflow):
            self._writer.write(inverse)
        self._pending_underflow = 0


class RangeDecoder:
    """Incrementally recover symbols from an integer range-coded bit string."""

    __slots__ = ("_code", "_high", "_low", "_reader")

    def __init__(self, coded: CodedBits) -> None:
        self._low = 0
        self._high = _STATE_MASK
        self._reader = _BitReader(coded)
        self._code = 0
        for _ in range(STATE_BITS):
            self._code = (self._code << 1) | self._reader.read_or_zero()

    def decode(self, table: FrequencyTable) -> int:
        """Decode one symbol with the same frequency table used by the encoder."""
        interval_size = self._high - self._low + 1
        offset = self._code - self._low
        cumulative_value = ((offset + 1) * table.total - 1) // interval_size
        symbol = table.symbol_for(cumulative_value)
        cumulative_low, cumulative_high = table.interval(symbol)

        previous_low = self._low
        self._low = previous_low + cumulative_low * interval_size // table.total
        self._high = previous_low + cumulative_high * interval_size // table.total - 1
        self._renormalize()
        return symbol

    def _renormalize(self) -> None:
        while True:
            if self._high < _HALF_RANGE:
                pass
            elif self._low >= _HALF_RANGE:
                self._low -= _HALF_RANGE
                self._high -= _HALF_RANGE
                self._code -= _HALF_RANGE
            elif self._low >= _QUARTER_RANGE and self._high < _THREE_QUARTER_RANGE:
                self._low -= _QUARTER_RANGE
                self._high -= _QUARTER_RANGE
                self._code -= _QUARTER_RANGE
            else:
                return
            self._low = (self._low << 1) & _STATE_MASK
            self._high = ((self._high << 1) & _STATE_MASK) | 1
            self._code = ((self._code << 1) & _STATE_MASK) | self._reader.read_or_zero()


class _BitWriter:
    __slots__ = ("_bit_length", "_buffer", "_current", "_used")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current = 0
        self._used = 0
        self._bit_length = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | bit
        self._used += 1
        self._bit_length += 1
        if self._used == 8:
            self._buffer.append(self._current)
            self._current = 0
            self._used = 0

    @property
    def bit_length(self) -> int:
        return self._bit_length

    def snapshot(self) -> CodedBits:
        data = bytes(self._buffer)
        if self._used:
            data += bytes((self._current << (8 - self._used),))
        return CodedBits(data, self._bit_length)

    def finish(self) -> CodedBits:
        return self.snapshot()


class _BitReader:
    __slots__ = ("_coded", "_position")

    def __init__(self, coded: CodedBits) -> None:
        self._coded = coded
        self._position = 0

    def read_or_zero(self) -> int:
        if self._position >= self._coded.bit_length:
            self._position += 1
            return 0
        byte_index, bit_index = divmod(self._position, 8)
        self._position += 1
        return (self._coded.data[byte_index] >> (7 - bit_index)) & 1
