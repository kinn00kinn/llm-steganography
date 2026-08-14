"""Normalize text and convert it to and from a bounded binary payload."""

from __future__ import annotations

import unicodedata
import zlib
from dataclasses import dataclass

from lsteg.payload.errors import (
    InvalidSecretTextError,
    MalformedPayloadError,
    SecretTooLongError,
)
from lsteg.payload.framing import (
    CompressionMethod,
    build_text_frame,
    parse_text_frame,
)

MAX_SECRET_CODE_POINTS = 100
MAX_SECRET_UTF8_BYTES = MAX_SECRET_CODE_POINTS * 4
_ZLIB_LEVEL = 9


@dataclass(frozen=True, slots=True)
class PayloadMetrics:
    """Size measurements captured at each Phase-1 payload boundary."""

    code_points: int
    raw_bytes: int
    stored_bytes: int
    frame_bytes: int
    compression: CompressionMethod

    @property
    def raw_bits(self) -> int:
        return self.raw_bytes * 8

    @property
    def stored_bits(self) -> int:
        return self.stored_bytes * 8

    @property
    def frame_bits(self) -> int:
        return self.frame_bytes * 8

    @property
    def bytes_saved(self) -> int:
        return self.raw_bytes - self.stored_bytes

    @property
    def compression_ratio(self) -> float | None:
        if self.raw_bytes == 0:
            return None
        return self.stored_bytes / self.raw_bytes


@dataclass(frozen=True, slots=True)
class EncodedTextPayload:
    """A canonical text payload frame and its non-secret size metadata."""

    frame: bytes
    normalized_text: str
    metrics: PayloadMetrics


def normalize_secret(secret_text: str) -> str:
    """Normalize a secret to NFC and enforce the version-1 text limits."""
    if not isinstance(secret_text, str):
        msg = "secret_text must be str"
        raise TypeError(msg)

    normalized = unicodedata.normalize("NFC", secret_text)
    code_points = len(normalized)
    if code_points > MAX_SECRET_CODE_POINTS:
        raise SecretTooLongError(
            f"normalized secret exceeds {MAX_SECRET_CODE_POINTS} code points: got {code_points}"
        )

    try:
        encoded = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise InvalidSecretTextError("secret contains an unpaired Unicode surrogate") from error

    if len(encoded) > MAX_SECRET_UTF8_BYTES:
        # This is implied by the code-point bound for valid Unicode, but keeping the
        # protocol limit explicit prevents future changes from weakening the decoder.
        raise InvalidSecretTextError(
            f"normalized secret exceeds {MAX_SECRET_UTF8_BYTES} UTF-8 bytes"
        )
    return normalized


def encode_text_payload(secret_text: str) -> EncodedTextPayload:
    """Encode a secret into the canonical Phase-1 text payload frame."""
    normalized = normalize_secret(secret_text)
    raw = normalized.encode("utf-8")
    compression, body = _select_storage(raw)
    frame = build_text_frame(
        compression=compression,
        decoded_size=len(raw),
        body=body,
    )
    metrics = PayloadMetrics(
        code_points=len(normalized),
        raw_bytes=len(raw),
        stored_bytes=len(body),
        frame_bytes=len(frame),
        compression=compression,
    )
    return EncodedTextPayload(frame=frame, normalized_text=normalized, metrics=metrics)


def decode_text_payload(frame: bytes) -> str:
    """Decode and validate a canonical Phase-1 text payload frame."""
    parsed = parse_text_frame(frame)
    if parsed.decoded_size > MAX_SECRET_UTF8_BYTES:
        raise MalformedPayloadError(f"decoded payload exceeds {MAX_SECRET_UTF8_BYTES} UTF-8 bytes")
    if len(parsed.body) > MAX_SECRET_UTF8_BYTES:
        raise MalformedPayloadError(f"stored payload exceeds {MAX_SECRET_UTF8_BYTES} bytes")

    if parsed.compression is CompressionMethod.RAW:
        if len(parsed.body) != parsed.decoded_size:
            raise MalformedPayloadError("raw payload length does not match decoded length")
        raw = parsed.body
    else:
        if len(parsed.body) >= parsed.decoded_size:
            raise MalformedPayloadError("compressed payload is not smaller than its raw form")
        raw = _decompress_zlib(parsed.body, expected_size=parsed.decoded_size)

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MalformedPayloadError("payload body is not valid UTF-8") from error

    normalized = unicodedata.normalize("NFC", text)
    if text != normalized:
        raise MalformedPayloadError("payload text is not in canonical NFC form")
    if len(text) > MAX_SECRET_CODE_POINTS:
        raise MalformedPayloadError(f"decoded payload exceeds {MAX_SECRET_CODE_POINTS} code points")
    return text


def _select_storage(raw: bytes) -> tuple[CompressionMethod, bytes]:
    compressed = zlib.compress(raw, level=_ZLIB_LEVEL)
    if len(compressed) < len(raw):
        return CompressionMethod.ZLIB, compressed
    return CompressionMethod.RAW, raw


def _decompress_zlib(body: bytes, *, expected_size: int) -> bytes:
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS)
    try:
        raw = decompressor.decompress(body, expected_size + 1)
        if len(raw) > expected_size or decompressor.unconsumed_tail:
            raise MalformedPayloadError("compressed payload expands beyond its declared length")

        remaining_limit = expected_size - len(raw) + 1
        raw += decompressor.flush(remaining_limit)
    except zlib.error as error:
        raise MalformedPayloadError("payload body is not a valid zlib stream") from error

    if len(raw) > expected_size:
        raise MalformedPayloadError("compressed payload expands beyond its declared length")
    if not decompressor.eof:
        raise MalformedPayloadError("compressed payload stream is truncated")
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise MalformedPayloadError("compressed payload contains trailing data")
    if len(raw) != expected_size:
        raise MalformedPayloadError(
            f"decoded payload length mismatch: expected {expected_size}, got {len(raw)}"
        )
    return raw
