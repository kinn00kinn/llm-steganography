from __future__ import annotations

import random
import unicodedata

import pytest
from nacl.secret import Aead

from lsteg.payload import (
    MASTER_KEY_SIZE,
    MAX_SECRET_CODE_POINTS,
    MAX_SECURE_FRAME_SIZE,
    AuthenticationError,
    InvalidMasterKeyError,
    MalformedPayloadError,
    PayloadCodecError,
    UnsupportedPayloadAlgorithmError,
    UnsupportedPayloadVersionError,
    decode_secure_text_payload,
    encode_secure_text_payload,
    encrypt_payload_frame,
    generate_master_key,
)
from lsteg.payload.crypto import MAX_INNER_FRAME_SIZE
from lsteg.payload.secure_framing import (
    SECURE_FRAME_HEADER_SIZE,
    SECURE_FRAME_MAGIC,
    SECURE_FRAME_VERSION,
    SECURE_NONCE_SIZE,
    SECURE_TAG_SIZE,
    EncryptionAlgorithm,
    parse_secure_frame,
)


@pytest.mark.parametrize(
    "secret",
    [
        "",
        "hello",
        "秘密",
        "e\u0301を含む文章",
        "😀" * MAX_SECRET_CODE_POINTS,
        "あ" * MAX_SECRET_CODE_POINTS,
    ],
)
def test_secure_text_round_trip(secret: str) -> None:
    key = generate_master_key()
    encoded = encode_secure_text_payload(secret, key)

    assert decode_secure_text_payload(encoded.frame, key) == unicodedata.normalize("NFC", secret)


def test_secure_envelope_contract_and_metrics() -> None:
    encoded = encode_secure_text_payload("公開用サンプル", bytes(range(MASTER_KEY_SIZE)))
    parsed = parse_secure_frame(encoded.frame)

    assert SECURE_FRAME_HEADER_SIZE == 10
    assert SECURE_NONCE_SIZE == Aead.NONCE_SIZE == 24
    assert SECURE_TAG_SIZE == Aead.MACBYTES == 16
    assert encoded.frame[:4] == SECURE_FRAME_MAGIC
    assert parsed.version == SECURE_FRAME_VERSION
    assert parsed.algorithm is EncryptionAlgorithm.XCHACHA20_POLY1305
    assert len(parsed.nonce) == SECURE_NONCE_SIZE
    assert len(parsed.ciphertext) == parsed.plaintext_size + SECURE_TAG_SIZE
    assert encoded.secure_metrics.inner_frame_bytes == parsed.plaintext_size
    assert encoded.secure_metrics.secure_frame_bytes == len(encoded.frame)
    assert encoded.secure_metrics.secure_frame_bits == len(encoded.frame) * 8
    assert encoded.secure_metrics.overhead_bytes == 50
    assert len(encoded.frame) <= MAX_SECURE_FRAME_SIZE


def test_random_nonce_changes_each_encryption() -> None:
    key = generate_master_key()
    nonces = {
        parse_secure_frame(encode_secure_text_payload("同じ秘密文", key).frame).nonce
        for _ in range(256)
    }

    assert len(nonces) == 256


def test_wrong_key_and_ciphertext_tampering_have_same_public_failure() -> None:
    correct_key = generate_master_key()
    frame = encode_secure_text_payload("秘密", correct_key).frame
    damaged = bytearray(frame)
    damaged[-1] ^= 0x01

    failures: list[AuthenticationError] = []
    for candidate, key in [
        (frame, generate_master_key()),
        (bytes(damaged), correct_key),
    ]:
        with pytest.raises(AuthenticationError) as captured:
            decode_secure_text_payload(candidate, key)
        failures.append(captured.value)

    assert {str(error) for error in failures} == {"secure payload authentication failed"}


@pytest.mark.parametrize("offset", [SECURE_FRAME_HEADER_SIZE, -1])
def test_one_bit_nonce_or_ciphertext_tampering_is_rejected(offset: int) -> None:
    key = generate_master_key()
    damaged = bytearray(encode_secure_text_payload("tamper test", key).frame)
    damaged[offset] ^= 0x01

    with pytest.raises(AuthenticationError, match="authentication failed"):
        decode_secure_text_payload(bytes(damaged), key)


def test_authenticated_header_tampering_is_rejected() -> None:
    key = generate_master_key()
    damaged = bytearray(encode_secure_text_payload("header", key).frame)
    damaged[7] ^= 0x01

    with pytest.raises(PayloadCodecError):
        decode_secure_text_payload(bytes(damaged), key)


def test_unknown_secure_version_is_rejected_before_decryption() -> None:
    frame = bytearray(encode_secure_text_payload("version", generate_master_key()).frame)
    frame[4] = SECURE_FRAME_VERSION + 1

    with pytest.raises(UnsupportedPayloadVersionError, match="version"):
        decode_secure_text_payload(bytes(frame), generate_master_key())


def test_unknown_algorithm_is_rejected_before_decryption() -> None:
    frame = bytearray(encode_secure_text_payload("algorithm", generate_master_key()).frame)
    frame[5] = 0xFF

    with pytest.raises(UnsupportedPayloadAlgorithmError, match="algorithm"):
        decode_secure_text_payload(bytes(frame), generate_master_key())


@pytest.mark.parametrize("size", [0, 1, SECURE_FRAME_HEADER_SIZE + SECURE_NONCE_SIZE])
def test_truncated_secure_frame_is_rejected(size: int) -> None:
    with pytest.raises(MalformedPayloadError, match="truncated"):
        decode_secure_text_payload(b"x" * size, generate_master_key())


def test_oversized_inner_frame_is_rejected_before_encryption() -> None:
    with pytest.raises(MalformedPayloadError, match="inner payload"):
        encrypt_payload_frame(b"x" * (MAX_INNER_FRAME_SIZE + 1), generate_master_key())


def test_invalid_master_key_is_rejected_without_echoing_it() -> None:
    invalid_key = b"not-a-valid-master-key"

    with pytest.raises(InvalidMasterKeyError) as captured:
        encode_secure_text_payload("秘密", invalid_key)
    assert invalid_key.hex() not in str(captured.value)


def test_secure_payload_repr_does_not_expose_secret_or_ciphertext() -> None:
    encoded = encode_secure_text_payload("reprに出さない秘密", generate_master_key())

    representation = repr(encoded)
    assert "reprに出さない秘密" not in representation
    assert encoded.frame.hex() not in representation


def test_five_hundred_seeded_secure_round_trips() -> None:
    source = random.Random(20260814)
    key = generate_master_key()
    alphabet = ("あ", "漢", "。", "A", "0", " ", "\n", "😀", "é", "e\u0301")

    for _ in range(500):
        secret = "".join(
            source.choice(alphabet) for _ in range(source.randint(0, MAX_SECRET_CODE_POINTS))
        )
        encoded = encode_secure_text_payload(secret, key)
        restored = decode_secure_text_payload(encoded.frame, key)
        assert restored == unicodedata.normalize("NFC", secret)
