from __future__ import annotations

import random
import unicodedata

import pytest

from lsteg.payload import (
    MAX_SECRET_CODE_POINTS,
    MAX_SECRET_UTF8_BYTES,
    CompressionMethod,
    InvalidSecretTextError,
    SecretTooLongError,
    decode_text_payload,
    encode_text_payload,
    normalize_secret,
)
from lsteg.payload.framing import TEXT_FRAME_HEADER_SIZE


@pytest.mark.parametrize(
    "secret",
    [
        "",
        "hello",
        "こんにちは",
        "大学生活についての日記です。",
        "🔐📝🌸",
        "改行も\n保持します。",
        "null文字\x00も文字列の一部です",
    ],
)
def test_text_payload_round_trip(secret: str) -> None:
    encoded = encode_text_payload(secret)

    assert decode_text_payload(encoded.frame) == unicodedata.normalize("NFC", secret)


def test_normalization_happens_before_code_point_limit() -> None:
    decomposed = "e\u0301" * MAX_SECRET_CODE_POINTS

    encoded = encode_text_payload(decomposed)

    assert encoded.normalized_text == "é" * MAX_SECRET_CODE_POINTS
    assert encoded.metrics.code_points == MAX_SECRET_CODE_POINTS
    assert decode_text_payload(encoded.frame) == encoded.normalized_text


def test_exact_code_point_and_utf8_limits_are_accepted() -> None:
    secret = "😀" * MAX_SECRET_CODE_POINTS

    encoded = encode_text_payload(secret)

    assert encoded.metrics.code_points == MAX_SECRET_CODE_POINTS
    assert encoded.metrics.raw_bytes == MAX_SECRET_UTF8_BYTES
    assert decode_text_payload(encoded.frame) == secret


def test_secret_over_code_point_limit_is_rejected() -> None:
    with pytest.raises(SecretTooLongError, match="101"):
        encode_text_payload("秘" * (MAX_SECRET_CODE_POINTS + 1))


def test_unpaired_surrogate_is_rejected() -> None:
    with pytest.raises(InvalidSecretTextError, match="surrogate"):
        encode_text_payload("\ud800")


def test_non_string_secret_is_rejected() -> None:
    with pytest.raises(TypeError, match="str"):
        normalize_secret(b"not text")  # type: ignore[arg-type]


def test_short_incompressible_secret_uses_raw_storage() -> None:
    encoded = encode_text_payload("短文")

    assert encoded.metrics.compression is CompressionMethod.RAW
    assert encoded.metrics.raw_bytes == len("短文".encode())
    assert encoded.metrics.stored_bytes == encoded.metrics.raw_bytes
    assert encoded.metrics.frame_bytes == TEXT_FRAME_HEADER_SIZE + encoded.metrics.stored_bytes


def test_repetitive_secret_uses_zlib_storage() -> None:
    encoded = encode_text_payload("あ" * MAX_SECRET_CODE_POINTS)

    assert encoded.metrics.compression is CompressionMethod.ZLIB
    assert encoded.metrics.stored_bytes < encoded.metrics.raw_bytes
    assert encoded.metrics.bytes_saved == (encoded.metrics.raw_bytes - encoded.metrics.stored_bytes)
    assert encoded.metrics.compression_ratio is not None
    assert encoded.metrics.compression_ratio < 1


def test_empty_secret_metrics_do_not_divide_by_zero() -> None:
    metrics = encode_text_payload("").metrics

    assert metrics.compression is CompressionMethod.RAW
    assert metrics.raw_bytes == 0
    assert metrics.raw_bits == 0
    assert metrics.stored_bits == 0
    assert metrics.frame_bits == TEXT_FRAME_HEADER_SIZE * 8
    assert metrics.compression_ratio is None


def test_encoding_is_deterministic_in_the_pinned_runtime() -> None:
    secret = "同じ入力から同じフレームを生成する。" * 4

    first = encode_text_payload(secret)
    second = encode_text_payload(secret)

    assert first == second


def test_one_thousand_seeded_unicode_round_trips() -> None:
    random_source = random.Random(20260814)
    alphabet = (
        "あ",
        "い",
        "漢",
        "字",
        "。",
        "、",
        "A",
        "z",
        "0",
        " ",
        "\n",
        "😀",
        "é",
        "e\u0301",
        "は\u3099",
    )

    for _ in range(1_000):
        secret = "".join(
            random_source.choice(alphabet)
            for _ in range(random_source.randint(0, MAX_SECRET_CODE_POINTS))
        )
        encoded = encode_text_payload(secret)
        expected = unicodedata.normalize("NFC", secret)

        assert encoded.normalized_text == expected
        assert decode_text_payload(encoded.frame) == expected
        assert encoded.metrics.frame_bits == len(encoded.frame) * 8
