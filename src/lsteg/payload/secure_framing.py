"""Versioned framing for authenticated encrypted payloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct

from lsteg.payload.errors import (
    MalformedPayloadError,
    UnsupportedPayloadAlgorithmError,
    UnsupportedPayloadVersionError,
)

SECURE_FRAME_MAGIC = b"LSEC"
SECURE_FRAME_VERSION = 1
SECURE_NONCE_SIZE = 24
SECURE_TAG_SIZE = 16

# magic, version, algorithm, plaintext bytes, ciphertext+tag bytes
_HEADER = Struct(">4sBBHH")
SECURE_FRAME_HEADER_SIZE = _HEADER.size
_MAX_UINT16 = (1 << 16) - 1


class EncryptionAlgorithm(IntEnum):
    """AEAD algorithms supported by secure frame version 1."""

    XCHACHA20_POLY1305 = 1


@dataclass(frozen=True, slots=True)
class SecureFrame:
    """A parsed secure frame before AEAD authentication and decryption."""

    version: int
    algorithm: EncryptionAlgorithm
    plaintext_size: int
    authenticated_header: bytes
    nonce: bytes
    ciphertext: bytes


def build_secure_frame_header(*, plaintext_size: int, ciphertext_size: int) -> bytes:
    """Build the authenticated version-1 envelope header."""
    if not 0 <= plaintext_size <= _MAX_UINT16:
        msg = f"plaintext_size must be between 0 and {_MAX_UINT16}"
        raise ValueError(msg)
    if not 0 <= ciphertext_size <= _MAX_UINT16:
        msg = f"ciphertext_size must be between 0 and {_MAX_UINT16}"
        raise ValueError(msg)
    if ciphertext_size != plaintext_size + SECURE_TAG_SIZE:
        msg = "ciphertext_size must equal plaintext_size plus the authentication tag"
        raise ValueError(msg)
    return _HEADER.pack(
        SECURE_FRAME_MAGIC,
        SECURE_FRAME_VERSION,
        int(EncryptionAlgorithm.XCHACHA20_POLY1305),
        plaintext_size,
        ciphertext_size,
    )


def build_secure_frame(*, plaintext_size: int, nonce: bytes, ciphertext: bytes) -> bytes:
    """Combine an authenticated header, nonce, and ciphertext without ambiguity."""
    if not isinstance(nonce, bytes):
        msg = "nonce must be bytes"
        raise TypeError(msg)
    if not isinstance(ciphertext, bytes):
        msg = "ciphertext must be bytes"
        raise TypeError(msg)
    if len(nonce) != SECURE_NONCE_SIZE:
        msg = f"nonce must be exactly {SECURE_NONCE_SIZE} bytes"
        raise ValueError(msg)

    header = build_secure_frame_header(
        plaintext_size=plaintext_size,
        ciphertext_size=len(ciphertext),
    )
    return header + nonce + ciphertext


def parse_secure_frame(frame: bytes) -> SecureFrame:
    """Parse and bound a secure frame before attempting authentication."""
    if not isinstance(frame, bytes):
        msg = "frame must be bytes"
        raise TypeError(msg)
    minimum_size = SECURE_FRAME_HEADER_SIZE + SECURE_NONCE_SIZE + SECURE_TAG_SIZE
    if len(frame) < minimum_size:
        raise MalformedPayloadError("secure payload frame is truncated")

    magic, version, algorithm_id, plaintext_size, ciphertext_size = _HEADER.unpack_from(frame)
    if magic != SECURE_FRAME_MAGIC:
        raise MalformedPayloadError("secure payload frame has an invalid magic value")
    if version != SECURE_FRAME_VERSION:
        raise UnsupportedPayloadVersionError(f"unsupported secure payload version: {version}")
    try:
        algorithm = EncryptionAlgorithm(algorithm_id)
    except ValueError as error:
        raise UnsupportedPayloadAlgorithmError(
            f"unsupported secure payload algorithm: {algorithm_id}"
        ) from error

    if ciphertext_size != plaintext_size + SECURE_TAG_SIZE:
        raise MalformedPayloadError("secure payload ciphertext length is inconsistent")
    expected_size = SECURE_FRAME_HEADER_SIZE + SECURE_NONCE_SIZE + ciphertext_size
    if len(frame) != expected_size:
        raise MalformedPayloadError(
            f"secure payload frame length mismatch: expected {expected_size}, got {len(frame)}"
        )

    nonce_start = SECURE_FRAME_HEADER_SIZE
    ciphertext_start = nonce_start + SECURE_NONCE_SIZE
    return SecureFrame(
        version=version,
        algorithm=algorithm,
        plaintext_size=plaintext_size,
        authenticated_header=frame[:SECURE_FRAME_HEADER_SIZE],
        nonce=frame[nonce_start:ciphertext_start],
        ciphertext=frame[ciphertext_start:],
    )
