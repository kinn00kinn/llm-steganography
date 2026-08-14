"""Deterministic integer frequency and Range Coding primitives."""

from lsteg.coding.codec import (
    BYTE_FREQUENCIES,
    FrequencyProvider,
    FrequencySource,
    decode_bytes,
    decode_symbols,
    encode_bytes,
    encode_symbols,
    map_bytes_to_symbols,
    recover_bytes_from_symbols,
)
from lsteg.coding.errors import (
    CodingError,
    InsufficientRangeDataError,
    InvalidFrequencyTableError,
    MalformedRangeFrameError,
    UnavailableSymbolError,
    UnsupportedRangeVersionError,
)
from lsteg.coding.framing import (
    MAX_CODED_BITS,
    MAX_SYMBOL_COUNT,
    RANGE_FRAME_HEADER_SIZE,
    RANGE_FRAME_VERSION,
    RangeCoderAlgorithm,
    RangeFrame,
    build_range_frame,
    parse_range_frame,
)
from lsteg.coding.frequencies import MAX_FREQUENCY_TOTAL, FrequencyTable
from lsteg.coding.range_coder import STATE_BITS, CodedBits, RangeDecoder, RangeEncoder

__all__ = [
    "BYTE_FREQUENCIES",
    "MAX_CODED_BITS",
    "MAX_FREQUENCY_TOTAL",
    "MAX_SYMBOL_COUNT",
    "RANGE_FRAME_HEADER_SIZE",
    "RANGE_FRAME_VERSION",
    "STATE_BITS",
    "CodedBits",
    "CodingError",
    "FrequencyProvider",
    "FrequencySource",
    "FrequencyTable",
    "InsufficientRangeDataError",
    "InvalidFrequencyTableError",
    "MalformedRangeFrameError",
    "RangeCoderAlgorithm",
    "RangeDecoder",
    "RangeEncoder",
    "RangeFrame",
    "UnavailableSymbolError",
    "UnsupportedRangeVersionError",
    "build_range_frame",
    "decode_bytes",
    "decode_symbols",
    "encode_bytes",
    "encode_symbols",
    "map_bytes_to_symbols",
    "parse_range_frame",
    "recover_bytes_from_symbols",
]
