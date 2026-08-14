"""Versioned binary framing for normalized text payloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct

from lsteg.payload.errors import MalformedPayloadError, UnsupportedPayloadVersionError

TEXT_FRAME_MAGIC = b"LSTG"
TEXT_FRAME_VERSION = 1

# magic, version, compression, decoded UTF-8 bytes, stored body bytes
_HEADER = Struct(">4sBBHH")
TEXT_FRAME_HEADER_SIZE = _HEADER.size
_MAX_UINT16 = (1 << 16) - 1


class CompressionMethod(IntEnum):
    """Compression methods supported by text frame version 1."""

    RAW = 0
    ZLIB = 1


@dataclass(frozen=True, slots=True)
class TextFrame:
    """A parsed text frame before decompression and UTF-8 decoding."""

    version: int
    compression: CompressionMethod
    decoded_size: int
    body: bytes


def build_text_frame(
    *,
    compression: CompressionMethod,
    decoded_size: int,
    body: bytes,
) -> bytes:
    """Build a canonical version-1 text frame."""
    if not 0 <= decoded_size <= _MAX_UINT16:
        msg = f"decoded_size must be between 0 and {_MAX_UINT16}"
        raise ValueError(msg)
    if len(body) > _MAX_UINT16:
        msg = f"body must be at most {_MAX_UINT16} bytes"
        raise ValueError(msg)

    header = _HEADER.pack(
        TEXT_FRAME_MAGIC,
        TEXT_FRAME_VERSION,
        int(compression),
        decoded_size,
        len(body),
    )
    return header + body


def parse_text_frame(frame: bytes) -> TextFrame:
    """Parse structural fields and reject ambiguous or unsupported frames."""
    if not isinstance(frame, bytes):
        msg = "frame must be bytes"
        raise TypeError(msg)
    if len(frame) < TEXT_FRAME_HEADER_SIZE:
        raise MalformedPayloadError("text payload frame is truncated")

    magic, version, compression_id, decoded_size, body_size = _HEADER.unpack_from(frame)
    if magic != TEXT_FRAME_MAGIC:
        raise MalformedPayloadError("text payload frame has an invalid magic value")
    if version != TEXT_FRAME_VERSION:
        raise UnsupportedPayloadVersionError(f"unsupported text payload version: {version}")

    try:
        compression = CompressionMethod(compression_id)
    except ValueError as error:
        raise MalformedPayloadError(
            f"unsupported text payload compression method: {compression_id}"
        ) from error

    expected_frame_size = TEXT_FRAME_HEADER_SIZE + body_size
    if len(frame) != expected_frame_size:
        raise MalformedPayloadError(
            f"text payload frame length mismatch: expected {expected_frame_size}, got {len(frame)}"
        )

    return TextFrame(
        version=version,
        compression=compression,
        decoded_size=decoded_size,
        body=frame[TEXT_FRAME_HEADER_SIZE:],
    )
