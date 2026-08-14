from __future__ import annotations

import zlib

import pytest

from lsteg.payload import (
    MAX_SECRET_CODE_POINTS,
    MAX_SECRET_UTF8_BYTES,
    CompressionMethod,
    MalformedPayloadError,
    UnsupportedPayloadVersionError,
    decode_text_payload,
    encode_text_payload,
)
from lsteg.payload.framing import (
    TEXT_FRAME_HEADER_SIZE,
    TEXT_FRAME_MAGIC,
    TEXT_FRAME_VERSION,
    build_text_frame,
    parse_text_frame,
)


def test_frame_header_contract() -> None:
    raw = "秘密".encode()
    frame = build_text_frame(
        compression=CompressionMethod.RAW,
        decoded_size=len(raw),
        body=raw,
    )

    assert TEXT_FRAME_HEADER_SIZE == 10
    assert frame[:4] == TEXT_FRAME_MAGIC
    assert frame[4] == TEXT_FRAME_VERSION
    assert parse_text_frame(frame).body == raw


@pytest.mark.parametrize("decoded_size", [-1, 1 << 16])
def test_frame_builder_rejects_unrepresentable_decoded_size(decoded_size: int) -> None:
    with pytest.raises(ValueError, match="decoded_size"):
        build_text_frame(
            compression=CompressionMethod.RAW,
            decoded_size=decoded_size,
            body=b"",
        )


def test_frame_builder_rejects_unrepresentable_body_size() -> None:
    with pytest.raises(ValueError, match="body"):
        build_text_frame(
            compression=CompressionMethod.RAW,
            decoded_size=0,
            body=b"x" * (1 << 16),
        )


def test_frame_parser_requires_bytes() -> None:
    with pytest.raises(TypeError, match="bytes"):
        parse_text_frame(bytearray(TEXT_FRAME_HEADER_SIZE))  # type: ignore[arg-type]


@pytest.mark.parametrize("frame", [b"", b"LSTG", b"LSTG\x01\x00\x00"])
def test_truncated_header_is_rejected(frame: bytes) -> None:
    with pytest.raises(MalformedPayloadError, match="truncated"):
        decode_text_payload(frame)


def test_invalid_magic_is_rejected() -> None:
    frame = bytearray(encode_text_payload("hello").frame)
    frame[0] ^= 0xFF

    with pytest.raises(MalformedPayloadError, match="magic"):
        decode_text_payload(bytes(frame))


def test_unknown_version_is_rejected() -> None:
    frame = bytearray(encode_text_payload("hello").frame)
    frame[4] = TEXT_FRAME_VERSION + 1

    with pytest.raises(UnsupportedPayloadVersionError, match="version"):
        decode_text_payload(bytes(frame))


def test_unknown_compression_method_is_rejected() -> None:
    frame = bytearray(encode_text_payload("hello").frame)
    frame[5] = 0xFF

    with pytest.raises(MalformedPayloadError, match="compression"):
        decode_text_payload(bytes(frame))


@pytest.mark.parametrize("change", ["truncate", "append"])
def test_body_length_mismatch_is_rejected(change: str) -> None:
    frame = encode_text_payload("payload").frame
    damaged = frame[:-1] if change == "truncate" else frame + b"x"

    with pytest.raises(MalformedPayloadError, match="length mismatch"):
        decode_text_payload(damaged)


def test_raw_length_mismatch_is_rejected() -> None:
    frame = build_text_frame(
        compression=CompressionMethod.RAW,
        decoded_size=2,
        body=b"x",
    )

    with pytest.raises(MalformedPayloadError, match="raw payload length"):
        decode_text_payload(frame)


def test_invalid_raw_utf8_is_rejected() -> None:
    frame = build_text_frame(
        compression=CompressionMethod.RAW,
        decoded_size=1,
        body=b"\xff",
    )

    with pytest.raises(MalformedPayloadError, match="UTF-8"):
        decode_text_payload(frame)


def test_non_nfc_raw_text_is_rejected() -> None:
    raw = "e\u0301".encode()
    frame = build_text_frame(
        compression=CompressionMethod.RAW,
        decoded_size=len(raw),
        body=raw,
    )

    with pytest.raises(MalformedPayloadError, match="NFC"):
        decode_text_payload(frame)


def test_decoded_code_point_limit_is_enforced() -> None:
    raw = b"a" * (MAX_SECRET_CODE_POINTS + 1)
    body = zlib.compress(raw, level=9)
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=len(raw),
        body=body,
    )

    with pytest.raises(MalformedPayloadError, match="code points"):
        decode_text_payload(frame)


def test_decoded_utf8_byte_limit_is_enforced_before_decompression() -> None:
    raw = b"a" * (MAX_SECRET_UTF8_BYTES + 1)
    body = zlib.compress(raw, level=9)
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=len(raw),
        body=body,
    )

    with pytest.raises(MalformedPayloadError, match="UTF-8 bytes"):
        decode_text_payload(frame)


def test_stored_byte_limit_is_enforced() -> None:
    body = b"x" * (MAX_SECRET_UTF8_BYTES + 1)
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=MAX_SECRET_UTF8_BYTES,
        body=body,
    )

    with pytest.raises(MalformedPayloadError, match="stored payload"):
        decode_text_payload(frame)


@pytest.mark.parametrize(
    "body",
    [
        b"not-zlib",
        zlib.compress(b"a" * 300, level=9)[:-1],
        zlib.compress(b"a" * 300, level=9) + b"trailing",
    ],
)
def test_invalid_zlib_stream_is_rejected(body: bytes) -> None:
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=300,
        body=body,
    )

    with pytest.raises(MalformedPayloadError):
        decode_text_payload(frame)


def test_compression_bomb_is_bounded_by_declared_size() -> None:
    body = zlib.compress(b"a" * 1_000, level=9)
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=MAX_SECRET_UTF8_BYTES,
        body=body,
    )

    with pytest.raises(MalformedPayloadError, match="declared length"):
        decode_text_payload(frame)


def test_compressed_length_mismatch_is_rejected() -> None:
    body = zlib.compress(b"a" * 300, level=9)
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=299,
        body=body,
    )

    with pytest.raises(MalformedPayloadError, match="declared length"):
        decode_text_payload(frame)


def test_noncanonical_compression_choice_is_rejected() -> None:
    body = zlib.compress(b"a", level=9)
    frame = build_text_frame(
        compression=CompressionMethod.ZLIB,
        decoded_size=1,
        body=body,
    )

    with pytest.raises(MalformedPayloadError, match="not smaller"):
        decode_text_payload(frame)
