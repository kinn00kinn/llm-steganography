"""Finite symbol and byte round-trips over the integer Range Coder."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from lsteg.coding.errors import InsufficientRangeDataError, MalformedRangeFrameError
from lsteg.coding.framing import MAX_SYMBOL_COUNT, build_range_frame, parse_range_frame
from lsteg.coding.frequencies import FrequencyTable
from lsteg.coding.range_coder import CodedBits, RangeDecoder, RangeEncoder

type FrequencyProvider = Callable[[int, Sequence[int]], FrequencyTable]
type FrequencySource = FrequencyTable | FrequencyProvider

BYTE_FREQUENCIES = FrequencyTable.uniform(256)


def encode_symbols(symbols: Sequence[int], frequencies: FrequencySource) -> bytes:
    """Encode a finite symbol sequence using fixed or context-dependent tables."""
    if len(symbols) > MAX_SYMBOL_COUNT:
        raise ValueError(f"symbol sequence exceeds {MAX_SYMBOL_COUNT} symbols")
    if len(symbols) == 0:
        return build_range_frame(symbol_count=0, coded=CodedBits(b"", 0))

    encoder = RangeEncoder()
    prefix: list[int] = []
    for index, symbol in enumerate(symbols):
        if isinstance(symbol, bool) or not isinstance(symbol, int):
            msg = "symbols must be integers"
            raise TypeError(msg)
        table = _resolve_table(frequencies, index, prefix)
        encoder.encode(table, symbol)
        prefix.append(symbol)
    return build_range_frame(symbol_count=len(symbols), coded=encoder.finish())


def decode_symbols(frame: bytes, frequencies: FrequencySource) -> tuple[int, ...]:
    """Decode the exact number of symbols declared by a finite range frame."""
    parsed = parse_range_frame(frame)
    if parsed.symbol_count == 0:
        return ()

    decoder = RangeDecoder(parsed.coded)
    symbols: list[int] = []
    for index in range(parsed.symbol_count):
        table = _resolve_table(frequencies, index, symbols)
        symbols.append(decoder.decode(table))
    return tuple(symbols)


def encode_bytes(data: bytes, frequencies: FrequencySource = BYTE_FREQUENCIES) -> bytes:
    """Encode bytes as symbols 0..255 using an externally reproducible model."""
    if not isinstance(data, bytes):
        msg = "data must be bytes"
        raise TypeError(msg)
    return encode_symbols(data, frequencies)


def decode_bytes(frame: bytes, frequencies: FrequencySource = BYTE_FREQUENCIES) -> bytes:
    """Decode a finite range frame and require every symbol to be one byte."""
    symbols = decode_symbols(frame, frequencies)
    try:
        return bytes(symbols)
    except ValueError as error:
        raise MalformedRangeFrameError("decoded symbol is outside the byte alphabet") from error


def map_bytes_to_symbols(
    payload: bytes,
    frequencies: FrequencySource,
    *,
    max_symbols: int = MAX_SYMBOL_COUNT,
) -> tuple[int, ...]:
    """Arithmetic-decode payload bits into symbols for a steganographic channel."""
    if not isinstance(payload, bytes):
        msg = "payload must be bytes"
        raise TypeError(msg)
    if isinstance(max_symbols, bool) or not isinstance(max_symbols, int):
        msg = "max_symbols must be int"
        raise TypeError(msg)
    if not 0 <= max_symbols <= MAX_SYMBOL_COUNT:
        raise ValueError(f"max_symbols must be between 0 and {MAX_SYMBOL_COUNT}")
    if not payload:
        return ()
    if isinstance(frequencies, FrequencyTable):
        positive_symbols = sum(frequency > 0 for frequency in frequencies.frequencies)
        if positive_symbols < 2:
            raise InsufficientRangeDataError("frequency table cannot carry payload bits")

    target_bits = len(payload) * 8
    # A trailing one bit avoids the ambiguous all-zero arithmetic-code boundary.
    # It is channel termination only; the receiver returns exactly target_bits.
    terminated = CodedBits(payload + b"\x80", target_bits + 1)
    decoder = RangeDecoder(terminated)
    mirror = RangeEncoder()
    symbols: list[int] = []
    while mirror.settled_bit_length < target_bits:
        if len(symbols) >= max_symbols:
            raise InsufficientRangeDataError(f"payload did not settle within {max_symbols} symbols")
        table = _resolve_table(frequencies, len(symbols), symbols)
        symbol = decoder.decode(table)
        mirror.encode(table, symbol)
        symbols.append(symbol)

    recovered_prefix = mirror.settled_bits().data[: len(payload)]
    if recovered_prefix != payload:  # pragma: no cover - coder symmetry invariant
        raise RuntimeError("range mapping failed to preserve the payload prefix")
    return tuple(symbols)


def recover_bytes_from_symbols(
    symbols: Sequence[int],
    payload_size: int,
    frequencies: FrequencySource,
) -> bytes:
    """Arithmetic-encode symbols and recover an exact, externally sized payload prefix."""
    if isinstance(payload_size, bool) or not isinstance(payload_size, int):
        msg = "payload_size must be int"
        raise TypeError(msg)
    if payload_size < 0:
        raise ValueError("payload_size must not be negative")
    if len(symbols) > MAX_SYMBOL_COUNT:
        raise ValueError(f"symbol sequence exceeds {MAX_SYMBOL_COUNT} symbols")
    if payload_size == 0:
        if len(symbols) != 0:
            raise InsufficientRangeDataError("empty payload must use an empty symbol sequence")
        return b""

    required_bits = payload_size * 8
    encoder = RangeEncoder()
    prefix: list[int] = []
    for index, symbol in enumerate(symbols):
        table = _resolve_table(frequencies, index, prefix)
        encoder.encode(table, symbol)
        prefix.append(symbol)

    if encoder.settled_bit_length < required_bits:
        raise InsufficientRangeDataError(
            f"symbol stream settled {encoder.settled_bit_length} of {required_bits} required bits"
        )
    return encoder.settled_bits().data[:payload_size]


def _resolve_table(
    source: FrequencySource,
    index: int,
    prefix: Sequence[int],
) -> FrequencyTable:
    if isinstance(source, FrequencyTable):
        return source
    if not callable(source):
        msg = "frequencies must be a FrequencyTable or callable provider"
        raise TypeError(msg)
    table = source(index, prefix)
    if not isinstance(table, FrequencyTable):
        msg = "frequency provider must return FrequencyTable"
        raise TypeError(msg)
    return table
