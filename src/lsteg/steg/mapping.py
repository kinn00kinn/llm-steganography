"""Keyed, deterministic candidate ordering for the steganographic channel.

The language model determines *which* tokens are plausible and the integer
frequency policy determines each token's interval width.  The shared stego key
only permutes those token/frequency pairs, so the marginal token probability is
unchanged while the interval ordering is secret.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from struct import Struct

from lsteg.coding.frequencies import FrequencyTable

STEGANOGRAPHY_KEY_SIZE = 32
CANDIDATE_MAPPING_VERSION = 1
_DOMAIN = b"llm-steganography/v1/candidate-mapping"
_UINT32 = Struct(">I")
_UINT64 = Struct(">Q")


def keyed_candidate_permutation(
    token_ids: Sequence[int],
    table: FrequencyTable,
    *,
    stego_key: bytes,
    position: int,
    context_token_ids: Sequence[int],
) -> tuple[list[int], FrequencyTable]:
    """Permute token/frequency pairs using a context-bound HMAC ordering.

    The frequency assigned to a token travels with that token.  Therefore this
    operation changes only the secret interval ordering; it does not change the
    probability mass assigned to any candidate token.
    """
    _validate_key(stego_key)
    if isinstance(position, bool) or not isinstance(position, int):
        raise TypeError("position must be int")
    if position < 0:
        raise ValueError("position must not be negative")
    if len(token_ids) != table.symbol_count:
        raise ValueError(
            "token_ids length must equal frequency table symbol count: "
            f"got {len(token_ids)} and {table.symbol_count}"
        )

    context_digest = _context_digest(context_token_ids)
    ranked: list[tuple[bytes, int, int]] = []
    for token_id, frequency in zip(token_ids, table.frequencies, strict=True):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token IDs must be integers")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError(f"token ID is outside uint32 range: {token_id}")
        message = (
            _DOMAIN
            + bytes((CANDIDATE_MAPPING_VERSION,))
            + _UINT64.pack(position)
            + context_digest
            + _UINT32.pack(token_id)
        )
        rank = hmac.digest(stego_key, message, "sha256")
        ranked.append((rank, token_id, frequency))

    # token_id is an explicit deterministic collision tie-breaker.
    ranked.sort(key=lambda item: (item[0], item[1]))
    permuted_ids = [token_id for _, token_id, _ in ranked]
    permuted_frequencies = [frequency for _, _, frequency in ranked]
    return permuted_ids, FrequencyTable(permuted_frequencies)


def _context_digest(token_ids: Sequence[int]) -> bytes:
    digest = hashlib.sha256()
    digest.update(_UINT64.pack(len(token_ids)))
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("context token IDs must be integers")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError(f"context token ID is outside uint32 range: {token_id}")
        digest.update(_UINT32.pack(token_id))
    return digest.digest()


def _validate_key(stego_key: bytes) -> None:
    if not isinstance(stego_key, bytes):
        raise TypeError("stego_key must be bytes")
    if len(stego_key) != STEGANOGRAPHY_KEY_SIZE:
        raise ValueError(f"stego_key must be exactly {STEGANOGRAPHY_KEY_SIZE} bytes")
