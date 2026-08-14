"""Finite-message framing for integer range-coded bits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct

from lsteg.coding.errors import MalformedRangeFrameError, UnsupportedRangeVersionError
from lsteg.coding.range_coder import CodedBits

RANGE_FRAME_MAGIC = b"LRNG"
RANGE_FRAME_VERSION = 1
MAX_SYMBOL_COUNT = 1 << 20
MAX_CODED_BITS = MAX_SYMBOL_COUNT * 32 + 2

# magic, version, coder, symbol count, meaningful coded bits
_HEADER = Struct(">4sBBII")
RANGE_FRAME_HEADER_SIZE = _HEADER.size


class RangeCoderAlgorithm(IntEnum):
    """Range-coder algorithms supported by finite frame version 1."""

    INTEGER_E1_E2_E3_32 = 1


@dataclass(frozen=True, slots=True)
class RangeFrame:
    """A parsed finite range-code message."""

    version: int
    algorithm: RangeCoderAlgorithm
    symbol_count: int
    coded: CodedBits


def build_range_frame(*, symbol_count: int, coded: CodedBits) -> bytes:
    """Serialize symbol count and exact coded-bit length with no ambiguity."""
    if isinstance(symbol_count, bool) or not isinstance(symbol_count, int):
        msg = "symbol_count must be int"
        raise TypeError(msg)
    if not 0 <= symbol_count <= MAX_SYMBOL_COUNT:
        raise ValueError(f"symbol_count must be between 0 and {MAX_SYMBOL_COUNT}")
    if coded.bit_length > MAX_CODED_BITS:
        raise ValueError(f"coded bit length exceeds {MAX_CODED_BITS}")
    if symbol_count == 0 and coded.bit_length != 0:
        raise ValueError("empty symbol sequence must have an empty coded bit string")
    if symbol_count > 0 and coded.bit_length == 0:
        raise ValueError("non-empty symbol sequence must contain coded bits")

    header = _HEADER.pack(
        RANGE_FRAME_MAGIC,
        RANGE_FRAME_VERSION,
        int(RangeCoderAlgorithm.INTEGER_E1_E2_E3_32),
        symbol_count,
        coded.bit_length,
    )
    return header + coded.data


def parse_range_frame(frame: bytes) -> RangeFrame:
    """Parse, bound, and canonicalize a finite range-code frame."""
    if not isinstance(frame, bytes):
        msg = "frame must be bytes"
        raise TypeError(msg)
    if len(frame) < RANGE_FRAME_HEADER_SIZE:
        raise MalformedRangeFrameError("range frame is truncated")

    magic, version, algorithm_id, symbol_count, bit_length = _HEADER.unpack_from(frame)
    if magic != RANGE_FRAME_MAGIC:
        raise MalformedRangeFrameError("range frame has an invalid magic value")
    if version != RANGE_FRAME_VERSION:
        raise UnsupportedRangeVersionError(f"unsupported range frame version: {version}")
    try:
        algorithm = RangeCoderAlgorithm(algorithm_id)
    except ValueError as error:
        raise UnsupportedRangeVersionError(
            f"unsupported range coder algorithm: {algorithm_id}"
        ) from error
    if symbol_count > MAX_SYMBOL_COUNT:
        raise MalformedRangeFrameError(f"range frame exceeds {MAX_SYMBOL_COUNT} symbols")
    if bit_length > MAX_CODED_BITS:
        raise MalformedRangeFrameError(f"range frame exceeds {MAX_CODED_BITS} coded bits")
    if symbol_count == 0 and bit_length != 0:
        raise MalformedRangeFrameError("empty range frame must not contain coded bits")
    if symbol_count > 0 and bit_length == 0:
        raise MalformedRangeFrameError("non-empty range frame has no coded bits")

    body = frame[RANGE_FRAME_HEADER_SIZE:]
    try:
        coded = CodedBits(body, bit_length)
    except MalformedRangeFrameError:
        raise
    return RangeFrame(
        version=version,
        algorithm=algorithm,
        symbol_count=symbol_count,
        coded=coded,
    )
