"""Round-trip tests for lsteg.steg.engine using a mock backend.

Design contract
---------------
A ``CodedBits`` object must be produced by ``RangeEncoder.finish()`` to be a
*canonical* range-coded stream.  Arbitrary byte strings are **not** guaranteed
to survive a decode→encode round-trip because the encoder always emits the
shortest canonical suffix that selects the current interval.

The correct invariant to test is therefore:

    symbols = [s0, s1, …, sn]
    coded   = RangeEncoder.encode(symbols, tables).finish()
    decoded = [RangeDecoder(coded).decode(table_i) for each i]
    assert decoded == symbols            # ← this is the contract

For the steganography engine the analogous invariant is:

    extract(hide(coded_bits)) recovers the same symbols
    (i.e. the payload bytes can be re-derived from the extracted CodedBits).

The tests below use a uniform-256 alphabet to encode raw bytes into a canonical
CodedBits, then pass that through hide/extract and verify the payload bytes are
recovered correctly.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from lsteg.coding.frequencies import FrequencyTable
from lsteg.coding.range_coder import CodedBits, RangeDecoder, RangeEncoder
from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg.engine import (
    CoverTokenError,
    InsufficientCoverCapacityError,
    SteganographyConfig,
    extract,
    extract_bytes,
    hide,
    hide_bytes,
)

# ---------------------------------------------------------------------------
# Minimal in-process mock backend
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 1000

_STEGO_KEY = bytes(range(32))


@dataclass
class _MockBackend:
    """A deterministic fake backend with a fixed vocabulary.

    ``next_logits`` is seeded from the last token ID so the probability
    distribution changes at every step but is perfectly reproducible.
    """

    _seed: int = 42
    _vocab_size: int = _VOCAB_SIZE

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

    def detokenize(self, token_ids: Sequence[int]) -> str:  # pragma: no cover
        return "".join(chr(t % 128) for t in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        rng = random.Random(self._seed + (token_ids[-1] if token_ids else 0))
        values = tuple(rng.gauss(0.0, 1.0) for _ in range(self._vocab_size))
        return Logits.from_values(values)


# ---------------------------------------------------------------------------
# Canonical stream helpers
# ---------------------------------------------------------------------------

_UNIFORM_256 = FrequencyTable([1] * 256)


def _encode_bytes(data: bytes) -> CodedBits:
    """Encode raw bytes as a canonical Range-coded stream (uniform 256-symbol table).

    This mirrors the real pipeline: the payload codec produces a canonical
    CodedBits object which is then handed to the steganography engine.
    """
    enc = RangeEncoder()
    for byte in data:
        enc.encode(_UNIFORM_256, byte)
    return enc.finish()


def _decode_bytes(coded: CodedBits, n_bytes: int) -> bytes:
    """Decode n_bytes from a canonical Range-coded stream."""
    dec = RangeDecoder(coded)
    return bytes(dec.decode(_UNIFORM_256) for _ in range(n_bytes))


def _make_payload(n_bytes: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(n_bytes))


# ---------------------------------------------------------------------------
# Core round-trip invariant
# ---------------------------------------------------------------------------


class TestHideExtractRoundTrip:
    """extract(hide(coded_bits)) must recover the original payload bytes."""

    def _round_trip(
        self,
        n_bytes: int,
        top_k: int = 64,
        seed: int = 0,
    ) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=top_k)
        payload = _make_payload(n_bytes, seed=seed)
        coded = _encode_bytes(payload)

        cover_tokens = hide(backend, "hello", coded, stego_key=_STEGO_KEY, config=cfg)
        assert len(cover_tokens) > 0, "hide must produce at least one token"

        recovered_coded = extract(backend, "hello", cover_tokens, stego_key=_STEGO_KEY, config=cfg)
        recovered_payload = _decode_bytes(recovered_coded, n_bytes)
        assert recovered_payload == payload, (
            f"payload mismatch for {n_bytes} bytes: "
            f"{recovered_payload.hex()!r} != {payload.hex()!r}"
        )

    # --- exact byte counts (both byte-boundary and sub-byte) -----------------

    def test_1_byte(self) -> None:
        self._round_trip(1)

    def test_2_bytes(self) -> None:
        self._round_trip(2)

    def test_4_bytes(self) -> None:
        self._round_trip(4)

    def test_8_bytes(self) -> None:
        self._round_trip(8)

    def test_16_bytes(self) -> None:
        self._round_trip(16)

    def test_32_bytes(self) -> None:
        self._round_trip(32)

    def test_44_bytes_typical_secret(self) -> None:
        """Typical secret after NFC normalization (~350 bits / 8 ≈ 44 bytes)."""
        self._round_trip(44)

    # --- termination / word-boundary regression tests -----------------------
    # These byte counts were chosen because they probe the 32-bit and 64-bit
    # word-boundary behaviour of the Range Coder (bits = bytes * 8 approx).

    def test_3_bytes_24_bits(self) -> None:
        self._round_trip(3)

    def test_4_bytes_32_bits(self) -> None:
        """Boundary: payload is exactly 32 bits when encoded."""
        self._round_trip(4)

    def test_5_bytes_40_bits(self) -> None:
        self._round_trip(5)

    def test_7_bytes_56_bits(self) -> None:
        self._round_trip(7)

    def test_8_bytes_64_bits(self) -> None:
        """Boundary: payload is exactly 64 bits when encoded."""
        self._round_trip(8)

    def test_9_bytes_72_bits(self) -> None:
        self._round_trip(9)

    # --- edge-case payloads --------------------------------------------------

    def test_all_zero_bytes(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        payload = b"\x00" * 4
        coded = _encode_bytes(payload)
        cover = hide(backend, "hello", coded, stego_key=_STEGO_KEY, config=cfg)
        recovered = extract(backend, "hello", cover, stego_key=_STEGO_KEY, config=cfg)
        assert _decode_bytes(recovered, 4) == payload

    def test_all_0xff_bytes(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        payload = b"\xff" * 4
        coded = _encode_bytes(payload)
        cover = hide(backend, "hello", coded, stego_key=_STEGO_KEY, config=cfg)
        recovered = extract(backend, "hello", cover, stego_key=_STEGO_KEY, config=cfg)
        assert _decode_bytes(recovered, 4) == payload

    def test_trailing_zeros(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        payload = b"\xab\xcd\x00\x00"
        coded = _encode_bytes(payload)
        cover = hide(backend, "hello", coded, stego_key=_STEGO_KEY, config=cfg)
        recovered = extract(backend, "hello", cover, stego_key=_STEGO_KEY, config=cfg)
        assert _decode_bytes(recovered, 4) == payload

    def test_trailing_ones(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        payload = b"\x00\x00\xff\xff"
        coded = _encode_bytes(payload)
        cover = hide(backend, "hello", coded, stego_key=_STEGO_KEY, config=cfg)
        recovered = extract(backend, "hello", cover, stego_key=_STEGO_KEY, config=cfg)
        assert _decode_bytes(recovered, 4) == payload

    # --- determinism and property tests -------------------------------------

    def test_deterministic_given_same_inputs(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        coded = _encode_bytes(_make_payload(8))
        tokens_1 = hide(backend, "deterministic", coded, stego_key=_STEGO_KEY, config=cfg)
        tokens_2 = hide(backend, "deterministic", coded, stego_key=_STEGO_KEY, config=cfg)
        assert tokens_1 == tokens_2

    def test_different_prompts_produce_different_cover(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        coded = _encode_bytes(_make_payload(8))
        tokens_a = hide(backend, "prompt_a", coded, stego_key=_STEGO_KEY, config=cfg)
        tokens_b = hide(backend, "prompt_b", coded, stego_key=_STEGO_KEY, config=cfg)
        assert tokens_a != tokens_b

    def test_repeated_seeds(self) -> None:
        for seed in range(5):
            self._round_trip(8, seed=seed)

    def test_property_random_payloads(self) -> None:
        """Property: round-trip holds for 20 random (size, seed) pairs."""
        rng = random.Random(9999)
        for _ in range(20):
            n = rng.randint(1, 32)
            seed = rng.randint(0, 10000)
            self._round_trip(n, top_k=64, seed=seed)

    # --- error cases --------------------------------------------------------

    def test_extract_raises_on_wrong_token(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=4)
        coded = _encode_bytes(b"\xab")
        cover_tokens = hide(backend, "test", coded, stego_key=_STEGO_KEY, config=cfg)
        tampered = [_VOCAB_SIZE - 1, *cover_tokens[1:]]
        with pytest.raises(CoverTokenError):
            extract(backend, "test", tampered, stego_key=_STEGO_KEY, config=cfg)

    def test_config_mismatch_raises(self) -> None:
        backend = _MockBackend()
        coded = _encode_bytes(b"\xab\xcd")
        cover_tokens = hide(
            backend,
            "test",
            coded,
            stego_key=_STEGO_KEY,
            config=SteganographyConfig(top_k=8),
        )
        with pytest.raises(CoverTokenError):
            extract(
                backend,
                "test",
                cover_tokens,
                stego_key=_STEGO_KEY,
                config=SteganographyConfig(top_k=4),
            )


# ---------------------------------------------------------------------------
# Arbitrary byte transport (Phase 7 channel primitive)
# ---------------------------------------------------------------------------


class TestByteChannelRoundTrip:
    def test_short_payload(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        payload = b"\x00\x42\xff\x10"
        cover = hide_bytes(
            backend,
            "hello",
            payload,
            stego_key=_STEGO_KEY,
            config=cfg,
        )
        recovered = extract_bytes(
            backend,
            "hello",
            cover,
            len(payload),
            stego_key=_STEGO_KEY,
            config=cfg,
        )
        assert recovered == payload

    def test_random_payloads(self) -> None:
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        rng = random.Random(20260815)
        for size in (1, 2, 3, 7, 8, 9, 16, 32, 50):
            payload = bytes(rng.getrandbits(8) for _ in range(size))
            cover = hide_bytes(
                backend,
                "transport",
                payload,
                stego_key=_STEGO_KEY,
                config=cfg,
            )
            assert (
                extract_bytes(
                    backend,
                    "transport",
                    cover,
                    size,
                    stego_key=_STEGO_KEY,
                    config=cfg,
                )
                == payload
            )

    def test_empty_payload(self) -> None:
        backend = _MockBackend()
        assert hide_bytes(backend, "hello", b"", stego_key=_STEGO_KEY) == []
        assert extract_bytes(backend, "hello", [], 0, stego_key=_STEGO_KEY) == b""

    def test_key_changes_cover_and_wrong_key_changes_payload(self) -> None:
        backend = _MockBackend()
        config = SteganographyConfig(top_k=64)
        payload = b"key separation regression payload"
        other_key = bytes(reversed(range(32)))
        cover = hide_bytes(
            backend,
            "hello",
            payload,
            stego_key=_STEGO_KEY,
            config=config,
        )
        other_cover = hide_bytes(
            backend,
            "hello",
            payload,
            stego_key=other_key,
            config=config,
        )
        assert other_cover != cover

        wrong = extract_bytes(
            backend,
            "hello",
            cover,
            len(payload),
            stego_key=other_key,
            config=config,
        )
        assert wrong != payload

    def test_too_small_token_budget_fails_closed(self) -> None:
        backend = _MockBackend()
        with pytest.raises(InsufficientCoverCapacityError):
            hide_bytes(
                backend,
                "hello",
                b"payload that cannot fit",
                stego_key=_STEGO_KEY,
                config=SteganographyConfig(top_k=2),
                max_tokens=1,
            )

    def test_excluded_tokens_never_appear(self) -> None:
        backend = _MockBackend()
        # Exclude a broad set so the property is exercised repeatedly.
        excluded = tuple(range(100))
        cfg = SteganographyConfig(top_k=64, excluded_token_ids=excluded)
        cover = hide_bytes(
            backend,
            "hello",
            b"abcdefgh",
            stego_key=_STEGO_KEY,
            config=cfg,
        )
        assert not set(cover).intersection(excluded)


# ---------------------------------------------------------------------------
# Canonical CodedBits contract: decode(encode(symbols)) == symbols
# ---------------------------------------------------------------------------


class TestCanonicalCodedBitsContract:
    """Verify the Range Coder round-trip invariant: decode(encode(x)) == x.

    This is the contract that CodedBits is built on.  Breaking it would
    silently corrupt all steganographic payloads.
    """

    def _rc_round_trip(self, data: bytes) -> bytes:
        coded = _encode_bytes(data)
        return _decode_bytes(coded, len(data))

    def test_single_byte(self) -> None:
        assert self._rc_round_trip(b"\x42") == b"\x42"

    def test_zero_byte(self) -> None:
        assert self._rc_round_trip(b"\x00") == b"\x00"

    def test_0xff(self) -> None:
        assert self._rc_round_trip(b"\xff") == b"\xff"

    def test_4_bytes(self) -> None:
        assert self._rc_round_trip(b"\xde\xad\xbe\xef") == b"\xde\xad\xbe\xef"

    def test_boundary_lengths(self) -> None:
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]:
            data = _make_payload(n, seed=n)
            assert self._rc_round_trip(data) == data, f"failed at n={n}"

    def test_all_zero_payload(self) -> None:
        assert self._rc_round_trip(b"\x00" * 8) == b"\x00" * 8

    def test_all_ones_payload(self) -> None:
        assert self._rc_round_trip(b"\xff" * 8) == b"\xff" * 8

    def test_property_random(self) -> None:
        rng = random.Random(12345)
        for _ in range(100):
            n = rng.randint(1, 64)
            data = bytes(rng.getrandbits(8) for _ in range(n))
            assert self._rc_round_trip(data) == data


# ---------------------------------------------------------------------------
# SteganographyConfig tests
# ---------------------------------------------------------------------------


class TestSteganographyConfig:
    def test_default_config(self) -> None:
        from lsteg.steg.frequencies import DEFAULT_TOP_K, FREQUENCY_TOTAL

        cfg = SteganographyConfig()
        assert cfg.top_k == DEFAULT_TOP_K
        assert cfg.frequency_total == FREQUENCY_TOTAL

    def test_invalid_top_k_zero(self) -> None:
        with pytest.raises(ValueError, match="top_k must satisfy"):
            SteganographyConfig(top_k=0)

    def test_invalid_frequency_total(self) -> None:
        from lsteg.coding.frequencies import MAX_FREQUENCY_TOTAL

        with pytest.raises(ValueError, match="frequency_total must be"):
            SteganographyConfig(frequency_total=MAX_FREQUENCY_TOTAL + 1)


# ---------------------------------------------------------------------------
# hide() / extract() property tests
# ---------------------------------------------------------------------------


class TestHideProperties:
    def test_cover_tokens_are_ints(self) -> None:
        backend = _MockBackend()
        coded = _encode_bytes(b"\xab")
        tokens = hide(backend, "hi", coded, stego_key=_STEGO_KEY)
        assert all(isinstance(t, int) for t in tokens)

    def test_cover_tokens_in_vocab_range(self) -> None:
        backend = _MockBackend()
        coded = _encode_bytes(b"\xab\xcd")
        tokens = hide(backend, "hi", coded, stego_key=_STEGO_KEY)
        assert all(0 <= t < _VOCAB_SIZE for t in tokens)


class TestExtractProperties:
    def test_empty_cover_produces_coded_bits(self) -> None:
        backend = _MockBackend()
        result = extract(backend, "hi", [], stego_key=_STEGO_KEY)
        assert isinstance(result, CodedBits)
