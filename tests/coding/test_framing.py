from __future__ import annotations

import pytest

from lsteg.coding import (
    MAX_CODED_BITS,
    MAX_SYMBOL_COUNT,
    RANGE_FRAME_HEADER_SIZE,
    RANGE_FRAME_VERSION,
    CodedBits,
    MalformedRangeFrameError,
    RangeCoderAlgorithm,
    UnsupportedRangeVersionError,
    build_range_frame,
    parse_range_frame,
)


def test_empty_frame_contract() -> None:
    frame = build_range_frame(symbol_count=0, coded=CodedBits(b"", 0))
    parsed = parse_range_frame(frame)

    assert RANGE_FRAME_HEADER_SIZE == 14
    assert frame == b"LRNG\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    assert parsed.version == RANGE_FRAME_VERSION
    assert parsed.algorithm is RangeCoderAlgorithm.INTEGER_E1_E2_E3_32
    assert parsed.symbol_count == 0
    assert parsed.coded == CodedBits(b"", 0)


def test_coded_bits_require_exact_bytes_and_zero_padding() -> None:
    assert CodedBits(b"\x80", 1).bit_length == 1

    with pytest.raises(MalformedRangeFrameError, match="byte length"):
        CodedBits(b"", 1)
    with pytest.raises(MalformedRangeFrameError, match="padding"):
        CodedBits(b"\x81", 1)


@pytest.mark.parametrize("frame", [b"", b"LRNG", b"LRNG\x01\x01"])
def test_truncated_header_is_rejected(frame: bytes) -> None:
    with pytest.raises(MalformedRangeFrameError, match="truncated"):
        parse_range_frame(frame)


def test_invalid_magic_is_rejected() -> None:
    frame = bytearray(build_range_frame(symbol_count=0, coded=CodedBits(b"", 0)))
    frame[0] ^= 0xFF

    with pytest.raises(MalformedRangeFrameError, match="magic"):
        parse_range_frame(bytes(frame))


def test_unknown_version_is_rejected() -> None:
    frame = bytearray(build_range_frame(symbol_count=0, coded=CodedBits(b"", 0)))
    frame[4] = RANGE_FRAME_VERSION + 1

    with pytest.raises(UnsupportedRangeVersionError, match="version"):
        parse_range_frame(bytes(frame))


def test_unknown_algorithm_is_rejected() -> None:
    frame = bytearray(build_range_frame(symbol_count=0, coded=CodedBits(b"", 0)))
    frame[5] = 0xFF

    with pytest.raises(UnsupportedRangeVersionError, match="algorithm"):
        parse_range_frame(bytes(frame))


def test_declared_bit_length_must_match_body() -> None:
    frame = bytearray(build_range_frame(symbol_count=1, coded=CodedBits(b"\x00", 1)))
    frame[13] = 9

    with pytest.raises(MalformedRangeFrameError, match="byte length"):
        parse_range_frame(bytes(frame))


def test_symbol_count_and_bit_length_are_bounded() -> None:
    frame = bytearray(build_range_frame(symbol_count=0, coded=CodedBits(b"", 0)))
    frame[6:10] = (MAX_SYMBOL_COUNT + 1).to_bytes(4, "big")

    with pytest.raises(MalformedRangeFrameError, match="symbols"):
        parse_range_frame(bytes(frame))

    frame[6:10] = (1).to_bytes(4, "big")
    frame[10:14] = (MAX_CODED_BITS + 1).to_bytes(4, "big")
    with pytest.raises(MalformedRangeFrameError, match="coded bits"):
        parse_range_frame(bytes(frame))


def test_empty_and_nonempty_fields_must_be_consistent() -> None:
    empty = bytearray(build_range_frame(symbol_count=0, coded=CodedBits(b"", 0)))
    empty[6:10] = (1).to_bytes(4, "big")

    with pytest.raises(MalformedRangeFrameError, match="no coded bits"):
        parse_range_frame(bytes(empty))
