"""Deterministic keyed transport whitening for Range-Coder source bytes."""

from __future__ import annotations

import hashlib
import hmac

_DOMAIN = b"llm-steganography/v2/range-source-whitening"


def xor_whiten(data: bytes, key: bytes) -> bytes:
    """XOR ``data`` with an HMAC-SHA256 stream; applying twice recovers it."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not key:
        raise ValueError("key must not be empty")
    output = bytearray(len(data))
    offset = 0
    counter = 0
    while offset < len(data):
        block = hmac.new(
            key,
            _DOMAIN + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        take = min(len(block), len(data) - offset)
        for index in range(take):
            output[offset + index] = data[offset + index] ^ block[index]
        offset += take
        counter += 1
    return bytes(output)
