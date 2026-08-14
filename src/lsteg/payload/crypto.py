"""Authenticated encryption around the canonical text payload frame."""

from __future__ import annotations

from dataclasses import dataclass, field

from nacl.exceptions import CryptoError
from nacl.secret import Aead
from nacl.utils import random as random_bytes

from lsteg.payload.codec import (
    MAX_SECRET_UTF8_BYTES,
    EncodedTextPayload,
    PayloadMetrics,
    decode_text_payload,
    encode_text_payload,
)
from lsteg.payload.errors import AuthenticationError, MalformedPayloadError
from lsteg.payload.framing import TEXT_FRAME_HEADER_SIZE
from lsteg.payload.keys import derive_keys, validate_master_key
from lsteg.payload.secure_framing import (
    SECURE_FRAME_HEADER_SIZE,
    SECURE_NONCE_SIZE,
    SECURE_TAG_SIZE,
    build_secure_frame,
    build_secure_frame_header,
    parse_secure_frame,
)

MAX_INNER_FRAME_SIZE = TEXT_FRAME_HEADER_SIZE + MAX_SECRET_UTF8_BYTES
MAX_SECURE_FRAME_SIZE = (
    SECURE_FRAME_HEADER_SIZE + SECURE_NONCE_SIZE + MAX_INNER_FRAME_SIZE + SECURE_TAG_SIZE
)


@dataclass(frozen=True, slots=True)
class SecurePayloadMetrics:
    """Non-secret size measurements for the Phase-2 secure envelope."""

    inner_frame_bytes: int
    secure_frame_bytes: int

    @property
    def inner_frame_bits(self) -> int:
        return self.inner_frame_bytes * 8

    @property
    def secure_frame_bits(self) -> int:
        return self.secure_frame_bytes * 8

    @property
    def overhead_bytes(self) -> int:
        return self.secure_frame_bytes - self.inner_frame_bytes


@dataclass(frozen=True, slots=True)
class EncodedSecureTextPayload:
    """An encrypted text payload and safe-to-report size metadata."""

    frame: bytes = field(repr=False)
    normalized_text: str = field(repr=False)
    text_metrics: PayloadMetrics
    secure_metrics: SecurePayloadMetrics


def encrypt_payload_frame(payload_frame: bytes, master_key: bytes) -> bytes:
    """Encrypt an already-framed payload with a fresh random XChaCha nonce."""
    if not isinstance(payload_frame, bytes):
        msg = "payload_frame must be bytes"
        raise TypeError(msg)
    if len(payload_frame) > MAX_INNER_FRAME_SIZE:
        raise MalformedPayloadError(
            f"inner payload exceeds {MAX_INNER_FRAME_SIZE} bytes: got {len(payload_frame)}"
        )

    encryption_key = derive_keys(validate_master_key(master_key)).encryption
    nonce = random_bytes(SECURE_NONCE_SIZE)
    expected_ciphertext_size = len(payload_frame) + SECURE_TAG_SIZE
    header = build_secure_frame_header(
        plaintext_size=len(payload_frame),
        ciphertext_size=expected_ciphertext_size,
    )
    encrypted = Aead(encryption_key).encrypt(payload_frame, header, nonce)
    ciphertext = bytes(encrypted.ciphertext)
    if len(ciphertext) != expected_ciphertext_size:  # pragma: no cover - library invariant
        raise RuntimeError("AEAD returned an unexpected ciphertext length")
    return build_secure_frame(
        plaintext_size=len(payload_frame),
        nonce=nonce,
        ciphertext=ciphertext,
    )


def decrypt_payload_frame(secure_frame: bytes, master_key: bytes) -> bytes:
    """Authenticate and decrypt a secure frame without exposing failure details."""
    parsed = parse_secure_frame(secure_frame)
    if parsed.plaintext_size > MAX_INNER_FRAME_SIZE:
        raise MalformedPayloadError(
            f"secure payload plaintext exceeds {MAX_INNER_FRAME_SIZE} bytes"
        )

    encryption_key = derive_keys(validate_master_key(master_key)).encryption
    try:
        plaintext = Aead(encryption_key).decrypt(
            parsed.ciphertext,
            parsed.authenticated_header,
            parsed.nonce,
        )
    except CryptoError as error:
        raise AuthenticationError("secure payload authentication failed") from error
    if len(plaintext) != parsed.plaintext_size:  # pragma: no cover - AEAD invariant
        raise MalformedPayloadError("decrypted payload length does not match its header")
    return bytes(plaintext)


def encode_secure_text_payload(secret_text: str, master_key: bytes) -> EncodedSecureTextPayload:
    """Normalize, compress, frame, and authenticate a bounded secret text."""
    encoded: EncodedTextPayload = encode_text_payload(secret_text)
    secure_frame = encrypt_payload_frame(encoded.frame, master_key)
    return EncodedSecureTextPayload(
        frame=secure_frame,
        normalized_text=encoded.normalized_text,
        text_metrics=encoded.metrics,
        secure_metrics=SecurePayloadMetrics(
            inner_frame_bytes=len(encoded.frame),
            secure_frame_bytes=len(secure_frame),
        ),
    )


def decode_secure_text_payload(secure_frame: bytes, master_key: bytes) -> str:
    """Authenticate, decrypt, and decode a secure text payload."""
    return decode_text_payload(decrypt_payload_frame(secure_frame, master_key))
