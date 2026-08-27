"""Round-trip tests for the SynthID-inspired empirical candidate-pool channel."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg import SteganographyConfig
from lsteg.steg.engine import CoverTokenError, InsufficientCoverCapacityError
from lsteg.steg.tournament_channel import (
    _empirical_pool_table,
    _xor_whiten,
    extract_bytes_tournament,
    hide_bytes_tournament,
)

_STEGO_KEY = bytes(range(32))


@dataclass
class _MockBackend:
    _vocab_size: int = 64

    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return self._vocab_size

    def tokenize(self, text: str) -> list[int]:
        return [ord(ch) % self._vocab_size for ch in text]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(0x3041 + (token_id % 80)) for token_id in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        seed = 1103 + len(token_ids) * 17 + (token_ids[-1] if token_ids else 0)
        rng = random.Random(seed)
        return Logits.from_values(tuple(rng.uniform(-1.0, 1.0) for _ in range(self._vocab_size)))


def test_tournament_round_trip_random_payloads() -> None:
    backend = _MockBackend()
    config = SteganographyConfig(top_k=32, temperature=0.9, top_p=0.95)
    rng = random.Random(20260815)
    for size in (1, 2, 3, 7, 16, 32):
        payload = bytes(rng.getrandbits(8) for _ in range(size))
        cover = hide_bytes_tournament(
            backend,
            "prompt",
            payload,
            stego_key=_STEGO_KEY,
            config=config,
            bits_per_token=4,
            max_tokens=4096,
        )
        recovered = extract_bytes_tournament(
            backend,
            "prompt",
            cover,
            size,
            stego_key=_STEGO_KEY,
            config=config,
            bits_per_token=4,
        )
        assert recovered == payload


def test_tournament_is_deterministic_for_same_key_and_payload() -> None:
    backend = _MockBackend()
    config = SteganographyConfig(top_k=32)
    payload = b"candidate-pool"
    first = hide_bytes_tournament(backend, "prompt", payload, stego_key=_STEGO_KEY, config=config)
    second = hide_bytes_tournament(backend, "prompt", payload, stego_key=_STEGO_KEY, config=config)
    assert first == second


def test_tournament_rejects_token_outside_keyed_pool() -> None:
    backend = _MockBackend()
    config = SteganographyConfig(top_k=8)
    payload = b"x"
    cover = hide_bytes_tournament(backend, "prompt", payload, stego_key=_STEGO_KEY, config=config)
    tampered = list(cover)
    tampered[0] = backend.vocabulary_size + 100
    with pytest.raises(CoverTokenError):
        extract_bytes_tournament(
            backend,
            "prompt",
            tampered,
            len(payload),
            stego_key=_STEGO_KEY,
            config=config,
        )


def test_tournament_one_symbol_channel_has_zero_capacity() -> None:
    backend = _MockBackend()
    config = SteganographyConfig(top_k=1)
    with pytest.raises(InsufficientCoverCapacityError):
        hide_bytes_tournament(
            backend,
            "prompt",
            b"x",
            stego_key=_STEGO_KEY,
            config=config,
            bits_per_token=4,
            max_tokens=8,
        )


def test_empirical_pool_keeps_duplicate_counts_as_frequency_mass() -> None:
    # A 15:1 base table and a deterministic key produce many repeated iid
    # draws.  The empirical table must retain all 16 slots as count mass rather
    # than turning duplicates into zero-capacity erasures.
    token_ids, table = _empirical_pool_table(
        [10, 20],
        [15, 1],
        16,
        stego_key=_STEGO_KEY,
        position=7,
        cover_prefix=[1, 2, 3, 4],
        pool_bits=4,
        context_window=4,
    )
    assert table.total == 16
    assert sum(table.frequencies) == 16
    assert set(token_ids) <= {10, 20}
    assert len(token_ids) == len(table.frequencies)


def test_payload_whitening_round_trip_and_changes_known_header() -> None:
    payload = b"LSEC\x01\x01\x00\x10" + bytes(range(32))
    whitened = _xor_whiten(payload, _STEGO_KEY)
    assert whitened != payload
    assert _xor_whiten(whitened, _STEGO_KEY) == payload


def test_parameter_validation() -> None:
    backend = _MockBackend()
    with pytest.raises(ValueError, match="bits_per_token"):
        hide_bytes_tournament(backend, "prompt", b"x", stego_key=_STEGO_KEY, bits_per_token=0)
    with pytest.raises(ValueError, match="context_window"):
        hide_bytes_tournament(backend, "prompt", b"x", stego_key=_STEGO_KEY, context_window=0)
