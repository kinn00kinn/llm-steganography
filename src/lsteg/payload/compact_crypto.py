"""Compact authenticated text payload for quality-first steganography experiments.

Version 2 removes the duplicated 10-byte text header and 10-byte secure header
used by the conservative Phase-2 framing.  It keeps an authenticated, versioned
2-byte metadata word, a random 96-bit nonce, and the standard 128-bit
AES-GCM-SIV authentication tag.

This format is additive: the existing XChaCha20-Poly1305 v1 framing remains the
compatibility/default protocol elsewhere in the project.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCMSIV

from lsteg.payload.codec import MAX_SECRET_UTF8_BYTES, normalize_secret
from lsteg.payload.errors import AuthenticationError, MalformedPayloadError
from lsteg.payload.keys import derive_keys, validate_master_key

COMPACT_SECURE_VERSION = 1
COMPACT_NONCE_SIZE = 12
COMPACT_TAG_SIZE = 16
COMPACT_META_SIZE = 2
COMPACT_FIXED_OVERHEAD = COMPACT_META_SIZE + COMPACT_NONCE_SIZE + COMPACT_TAG_SIZE

# 2-bit version | 1-bit compression | 9-bit decoded byte length.  The high
# four bits are reserved and must remain zero.  MAX_SECRET_UTF8_BYTES == 400.
_VERSION_SHIFT = 10
_COMPRESSION_SHIFT = 9
_DECODED_SIZE_MASK = (1 << 9) - 1
_RESERVED_MASK = 0xF000
_COMPRESSION_RAW = 0
_COMPRESSION_DEFLATE = 1


@dataclass(frozen=True, slots=True)
class CompactSecurePayloadMetrics:
    """Safe-to-report size measurements for the compact secure frame."""

    code_points: int
    raw_bytes: int
    stored_bytes: int
    frame_bytes: int
    compressed: bool

    @property
    def frame_bits(self) -> int:
        return self.frame_bytes * 8

    @property
    def overhead_bytes(self) -> int:
        return self.frame_bytes - self.stored_bytes


@dataclass(frozen=True, slots=True)
class EncodedCompactSecureTextPayload:
    """Compact authenticated payload plus non-secret metrics."""

    frame: bytes = field(repr=False)
    normalized_text: str = field(repr=False)
    metrics: CompactSecurePayloadMetrics


def encode_compact_secure_text_payload(
    secret_text: str,
    master_key: bytes,
) -> EncodedCompactSecureTextPayload:
    """Normalize, optionally raw-DEFLATE, and encrypt a bounded text secret."""
    normalized = normalize_secret(secret_text)
    raw = normalized.encode("utf-8")
    compressed = _raw_deflate(raw)
    if len(compressed) < len(raw):
        compression = _COMPRESSION_DEFLATE
        body = compressed
    else:
        compression = _COMPRESSION_RAW
        body = raw

    metadata = _build_metadata(compression=compression, decoded_size=len(raw))
    nonce = os.urandom(COMPACT_NONCE_SIZE)
    key = derive_keys(validate_master_key(master_key)).encryption
    ciphertext = AESGCMSIV(key).encrypt(nonce, body, metadata)
    frame = metadata + nonce + ciphertext
    return EncodedCompactSecureTextPayload(
        frame=frame,
        normalized_text=normalized,
        metrics=CompactSecurePayloadMetrics(
            code_points=len(normalized),
            raw_bytes=len(raw),
            stored_bytes=len(body),
            frame_bytes=len(frame),
            compressed=compression == _COMPRESSION_DEFLATE,
        ),
    )


def decode_compact_secure_text_payload(frame: bytes, master_key: bytes) -> str:
    """Authenticate and decode a compact secure text payload."""
    if not isinstance(frame, bytes):
        raise TypeError("frame must be bytes")
    if len(frame) < COMPACT_FIXED_OVERHEAD:
        raise MalformedPayloadError("compact secure payload frame is truncated")

    metadata = frame[:COMPACT_META_SIZE]
    version, compression, decoded_size = _parse_metadata(metadata)
    if version != COMPACT_SECURE_VERSION:
        raise MalformedPayloadError(f"unsupported compact payload version: {version}")

    nonce_start = COMPACT_META_SIZE
    ciphertext_start = nonce_start + COMPACT_NONCE_SIZE
    nonce = frame[nonce_start:ciphertext_start]
    ciphertext = frame[ciphertext_start:]
    if len(ciphertext) < COMPACT_TAG_SIZE:
        raise MalformedPayloadError("compact secure payload ciphertext is truncated")

    key = derive_keys(validate_master_key(master_key)).encryption
    try:
        body = AESGCMSIV(key).decrypt(nonce, ciphertext, metadata)
    except InvalidTag as error:
        raise AuthenticationError("compact secure payload authentication failed") from error

    if compression == _COMPRESSION_RAW:
        raw = body
        if len(raw) != decoded_size:
            raise MalformedPayloadError("compact raw payload length does not match metadata")
    elif compression == _COMPRESSION_DEFLATE:
        if len(body) >= decoded_size:
            raise MalformedPayloadError("compact compressed payload is not smaller than raw text")
        raw = _raw_inflate(body, expected_size=decoded_size)
    else:  # pragma: no cover - metadata parser validates this
        raise MalformedPayloadError("unsupported compact compression method")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MalformedPayloadError("compact payload body is not valid UTF-8") from error
    if normalize_secret(text) != text:
        raise MalformedPayloadError("compact payload text is not canonical NFC")
    return text


def _build_metadata(*, compression: int, decoded_size: int) -> bytes:
    if compression not in (_COMPRESSION_RAW, _COMPRESSION_DEFLATE):
        raise ValueError("unsupported compact compression method")
    if not 0 <= decoded_size <= MAX_SECRET_UTF8_BYTES:
        raise ValueError("decoded_size exceeds compact payload bound")
    value = (
        (COMPACT_SECURE_VERSION << _VERSION_SHIFT)
        | (compression << _COMPRESSION_SHIFT)
        | decoded_size
    )
    return value.to_bytes(COMPACT_META_SIZE, "big")


def _parse_metadata(metadata: bytes) -> tuple[int, int, int]:
    if len(metadata) != COMPACT_META_SIZE:
        raise MalformedPayloadError("compact metadata must be exactly two bytes")
    value = int.from_bytes(metadata, "big")
    if value & _RESERVED_MASK:
        raise MalformedPayloadError("compact payload reserved metadata bits are non-zero")
    version = (value >> _VERSION_SHIFT) & 0x3
    compression = (value >> _COMPRESSION_SHIFT) & 0x1
    decoded_size = value & _DECODED_SIZE_MASK
    if decoded_size > MAX_SECRET_UTF8_BYTES:
        raise MalformedPayloadError("compact payload decoded size exceeds protocol bound")
    return version, compression, decoded_size


def _raw_deflate(raw: bytes) -> bytes:
    compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
    return compressor.compress(raw) + compressor.flush()


def _raw_inflate(body: bytes, *, expected_size: int) -> bytes:
    decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    try:
        raw = decompressor.decompress(body, expected_size + 1)
        if len(raw) > expected_size or decompressor.unconsumed_tail:
            raise MalformedPayloadError("compact payload expands beyond declared length")
        raw += decompressor.flush(expected_size - len(raw) + 1)
    except zlib.error as error:
        raise MalformedPayloadError("compact payload body is not valid raw DEFLATE") from error
    if len(raw) != expected_size or not decompressor.eof:
        raise MalformedPayloadError("compact payload decoded length mismatch")
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise MalformedPayloadError("compact payload contains trailing compressed data")
    return raw
