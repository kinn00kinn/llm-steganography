import pytest

from lsteg.payload.compact_crypto import (
    COMPACT_FIXED_OVERHEAD,
    decode_compact_secure_text_payload,
    encode_compact_secure_text_payload,
)
from lsteg.payload.errors import AuthenticationError
from lsteg.payload.keys import generate_master_key


def test_compact_payload_round_trip_short_japanese() -> None:
    key = generate_master_key()
    encoded = encode_compact_secure_text_payload("確認", key)
    assert decode_compact_secure_text_payload(encoded.frame, key) == "確認"
    assert len(encoded.frame) == COMPACT_FIXED_OVERHEAD + len("確認".encode())


def test_compact_payload_round_trip_compressed_japanese() -> None:
    key = generate_master_key()
    text = "研究室で昼食について友人と話した。" * 8
    encoded = encode_compact_secure_text_payload(text[:100], key)
    assert decode_compact_secure_text_payload(encoded.frame, key) == text[:100]
    assert encoded.metrics.compressed


def test_compact_payload_wrong_key_fails_authentication() -> None:
    encoded = encode_compact_secure_text_payload("確認", generate_master_key())
    with pytest.raises(AuthenticationError):
        decode_compact_secure_text_payload(encoded.frame, generate_master_key())


def test_compact_metadata_is_authenticated() -> None:
    key = generate_master_key()
    encoded = encode_compact_secure_text_payload("確認", key)
    tampered = bytes([encoded.frame[0], encoded.frame[1] ^ 1]) + encoded.frame[2:]
    with pytest.raises((AuthenticationError, ValueError)):
        decode_compact_secure_text_payload(tampered, key)
