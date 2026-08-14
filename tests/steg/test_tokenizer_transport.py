"""Phase 6.5 — Tokenizer transport invariance tests.

Problem statement
-----------------
The steganography pipeline requires that when the *sender* generates a sequence
of token IDs with ``hide()``, and the *receiver* receives the cover text as a
**Unicode string**, the receiver can reconstruct the exact same token ID
sequence via ``backend.tokenize()``.

If this round-trip is broken, the Range Coder state on the receiver side will
diverge from the sender's state at the first mismatched token, making
decryption impossible even if the Range Coder itself is correct.

Failure modes to guard against
-------------------------------
1. ``tokenize(detokenize(ids)) != ids``
   - whitespace insertion/removal around special tokens
   - BOS/EOS token injection or stripping
   - subword boundary shifts for adjacent tokens
   - special-character (e.g. ``<|im_start|>``) handling
2. Token IDs that detokenize to the empty string (invisible tokens) will
   disappear from the cover text, making the receiver skip them.
3. Tokens that contain null bytes or control characters may be dropped by
   text-layer transmission.

These tests use the mock backend from test_engine to verify the *interface
contract* without requiring a real LLM.  A separate marked test class
(``TestTokenizerTransportReal``) is provided for the real Qwen3 backend.

The invariant
-------------
    detokenize(tokenize(text)) == text                  (lossless text encoding)
    tokenize(detokenize(ids)) == ids                    (transport invariance)
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from lsteg.coding.frequencies import FrequencyTable
from lsteg.coding.range_coder import RangeDecoder, RangeEncoder
from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg.engine import (
    SteganographyConfig,
    extract,
    extract_bytes,
    hide,
    hide_bytes,
)

# ---------------------------------------------------------------------------
# Mock backend (same as test_engine, kept local for clarity)
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 1000

_STEGO_KEY = bytes(range(32))
_QWEN_CONTROL_TOKENS = (151643, 151644, 151645)


@dataclass
class _MockBackend:
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
        # Bijective: each character is a single private-use Unicode codepoint
        # in the range U+E000 to U+E3E7 (1000 values), mapping to token 0-999.
        return [(ord(ch) - 0xE000) % self._vocab_size for ch in text]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        # Inverse: token t → chr(0xE000 + t) — guaranteed unique and round-trippable.
        return "".join(chr(0xE000 + t) for t in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        rng = random.Random(self._seed + (token_ids[-1] if token_ids else 0))
        values = tuple(rng.gauss(0.0, 1.0) for _ in range(self._vocab_size))
        return Logits.from_values(values)


# ---------------------------------------------------------------------------
# Helper: encode bytes as canonical CodedBits
# ---------------------------------------------------------------------------

_UNIFORM_256 = FrequencyTable([1] * 256)


def _encode_bytes(data: bytes) -> object:
    enc = RangeEncoder()
    for byte in data:
        enc.encode(_UNIFORM_256, byte)
    return enc.finish()


def _decode_bytes(coded: object, n: int) -> bytes:
    dec = RangeDecoder(coded)  # type: ignore[arg-type]
    return bytes(dec.decode(_UNIFORM_256) for _ in range(n))


# ---------------------------------------------------------------------------
# Mock backend transport invariance
# ---------------------------------------------------------------------------


class TestMockBackendTransportInvariance:
    """Verify the mock backend satisfies tokenize(detokenize(ids)) == ids."""

    def test_single_token_roundtrip(self) -> None:
        backend = _MockBackend()
        for token_id in range(0, _VOCAB_SIZE, 50):
            text = backend.detokenize([token_id])
            recovered = backend.tokenize(text)
            assert recovered == [token_id], (
                f"token {token_id}: detokenize→tokenize did not round-trip"
            )

    def test_multi_token_roundtrip(self) -> None:
        backend = _MockBackend()
        rng = random.Random(0)
        for _ in range(50):
            ids = [rng.randint(0, 127) for _ in range(rng.randint(1, 20))]
            text = backend.detokenize(ids)
            assert backend.tokenize(text) == ids

    def test_text_to_ids_and_back(self) -> None:
        backend = _MockBackend()
        # Use PUA chars that map back correctly: chr(0xE000 + i) → token i
        for ids in [[0], [1, 2, 3], list(range(10)), [999]]:
            text = backend.detokenize(ids)
            assert backend.tokenize(text) == ids

    def test_empty_sequence(self) -> None:
        backend = _MockBackend()
        assert backend.tokenize("") == []
        assert backend.detokenize([]) == ""

    def test_roundtrip_after_hide(self) -> None:
        """Token IDs produced by hide() must survive detokenize→tokenize."""
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        coded = _encode_bytes(b"\xde\xad\xbe\xef")
        cover_ids = hide(
            backend,
            "hello",
            coded,
            stego_key=_STEGO_KEY,
            config=cfg,
        )  # type: ignore[arg-type]

        # Simulate text transmission
        cover_text = backend.detokenize(cover_ids)
        recovered_ids = backend.tokenize(cover_text)

        assert recovered_ids == cover_ids, (
            "Token IDs changed after detokenize→tokenize. "
            "The receiver would compute different logit distributions."
        )

    def test_full_pipeline_via_text_transmission(self) -> None:
        """Full sender→text→receiver pipeline using the mock backend.

        This is the key Phase 7 invariant tested with a mock backend:
            sender:   hide(coded_bits) → cover_ids → detokenize → cover_text
            receiver: cover_text → tokenize → cover_ids → extract → coded_bits
        """
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=64)
        payload = b"\x42\x13\x37\xff"
        coded = _encode_bytes(payload)

        # Sender side
        cover_ids = hide(
            backend,
            "hello",
            coded,
            stego_key=_STEGO_KEY,
            config=cfg,
        )  # type: ignore[arg-type]
        prompt_text = "hello"
        cover_text = backend.detokenize(cover_ids)

        # Text transmission (just passing the string, as in real use)
        received_text = cover_text

        # Receiver side
        received_ids = backend.tokenize(received_text)
        assert received_ids == cover_ids, (
            "Transport invariance broken: receiver gets different token IDs"
        )

        recovered_coded = extract(
            backend,
            prompt_text,
            received_ids,
            stego_key=_STEGO_KEY,
            config=cfg,
        )
        recovered_payload = _decode_bytes(recovered_coded, len(payload))
        assert recovered_payload == payload


# ---------------------------------------------------------------------------
# Invariant documentation tests (always run)
# ---------------------------------------------------------------------------


class TestTransportInvarianceDocumentation:
    """Document what transport invariance means and why it matters.

    These tests encode the *requirement*, not just the mock implementation.
    Any real backend must pass analogous tests.
    """

    def test_invariant_description_in_docstring(self) -> None:
        """Ensure this module's docstring is present (documentation hygiene)."""
        import importlib
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "_tt",
            Path(__file__).resolve(),
        )
        assert spec is not None
        m = importlib.util.module_from_spec(spec)
        doc = m.__doc__ or ""
        # The module-level docstring is the one in this file itself.
        # Just check the current module's __doc__.
        import sys

        current = sys.modules.get(__name__) or sys.modules.get("test_tokenizer_transport")
        doc = (current.__doc__ if current else __doc__) or ""
        assert "transport" in doc.lower() and "invariant" in doc.lower()

    def test_cover_text_carries_full_information(self) -> None:
        """Every cover token must survive the text round-trip with the mock."""
        backend = _MockBackend()
        cfg = SteganographyConfig(top_k=32)
        coded = _encode_bytes(b"\xca\xfe\xba\xbe")
        cover_ids = hide(
            backend,
            "test",
            coded,
            stego_key=_STEGO_KEY,
            config=cfg,
        )  # type: ignore[arg-type]

        text = backend.detokenize(cover_ids)
        recovered = backend.tokenize(text)

        # Every token must be present and in the same order
        assert len(recovered) == len(cover_ids)
        assert recovered == cover_ids


# ---------------------------------------------------------------------------
# Real backend tests (skipped unless LSTEG_RUN_MODEL_TESTS=1)
# ---------------------------------------------------------------------------

_RUN_MODEL = os.getenv("LSTEG_RUN_MODEL_TESTS") == "1"
_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "config" / "models" / "qwen3-1.7b-debug.json"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _PROJECT_ROOT / "artifacts" / "model-cache"


@pytest.mark.skipif(
    not _RUN_MODEL,
    reason="set LSTEG_RUN_MODEL_TESTS=1 after uv sync --extra model",
)
class TestTokenizerTransportReal:
    """Tokenizer transport invariance using the real Qwen3-1.7B backend.

    These tests are the Phase 6.5 acceptance criteria.  They verify that:
      1. tokenize(detokenize(ids)) == ids for tokens drawn from top-k logits
      2. The full sender→text→receiver pipeline recovers the payload

    Run with: LSTEG_RUN_MODEL_TESTS=1 uv run pytest tests/steg/test_tokenizer_transport.py -v
    """

    @pytest.fixture(scope="class")
    def backend(self):  # type: ignore[no-untyped-def]
        from lsteg.model.manifest import ModelManifest
        from lsteg.model.transformers_backend import TransformersBackend

        manifest = ModelManifest.from_path(_DEFAULT_MANIFEST)
        return TransformersBackend.load(manifest, cache_dir=_DEFAULT_CACHE, local_files_only=True)

    def test_top_k_tokens_survive_detokenize_tokenize(
        self, backend
    ) -> None:  # type: ignore[no-untyped-def]
        """Every token in the top-k alphabet must round-trip through text."""
        from lsteg.steg.frequencies import logits_to_frequency_table

        prompt = "架空のニュース記事：\n本日午後、東京都内の"  # noqa: RUF001
        token_ids = backend.tokenize(prompt)
        logits = backend.next_logits(token_ids)
        top_ids, _ = logits_to_frequency_table(
            list(logits),
            top_k=256,
            excluded_token_ids=_QWEN_CONTROL_TOKENS,
        )

        failures: list[int] = []
        for tid in top_ids:
            text = backend.detokenize([tid])
            recovered = backend.tokenize(text) if text else []
            if recovered != [tid]:
                failures.append(tid)

        # Report all failures, not just the first
        assert not failures, (
            f"{len(failures)}/{len(top_ids)} top-k tokens failed transport invariance: "
            f"{failures[:10]!r}{'…' if len(failures) > 10 else ''}"
        )

    def test_full_pipeline_8bit(self, backend) -> None:  # type: ignore[no-untyped-def]
        """8-bit payload round-trip via text transmission with real LLM."""
        cfg = SteganographyConfig(
            top_k=64,
            excluded_token_ids=_QWEN_CONTROL_TOKENS,
        )
        payload = b"\x42"
        coded = _encode_bytes(payload)
        prompt = "テスト："  # noqa: RUF001

        cover_ids = hide(
            backend,
            prompt,
            coded,
            stego_key=_STEGO_KEY,
            config=cfg,
        )  # type: ignore[arg-type]
        cover_text = backend.detokenize(cover_ids)

        received_ids = backend.tokenize(cover_text)
        assert received_ids == cover_ids, "Transport invariance failed with real Qwen3 backend"

        recovered_coded = extract(
            backend,
            prompt,
            received_ids,
            stego_key=_STEGO_KEY,
            config=cfg,
        )
        recovered = _decode_bytes(recovered_coded, len(payload))
        assert recovered == payload


@pytest.mark.skipif(
    not _RUN_MODEL,
    reason="set LSTEG_RUN_MODEL_TESTS=1 after uv sync --extra model",
)
def test_phase7_byte_channel_via_text() -> None:
    """Phase 7 smoke test: arbitrary bytes survive real-Qwen text transport."""
    from lsteg.model.manifest import ModelManifest
    from lsteg.model.transformers_backend import TransformersBackend

    manifest = ModelManifest.from_path(_DEFAULT_MANIFEST)
    backend = TransformersBackend.load(
        manifest,
        cache_dir=_DEFAULT_CACHE,
        local_files_only=True,
    )
    config = SteganographyConfig(
        top_k=64,
        excluded_token_ids=_QWEN_CONTROL_TOKENS,
    )
    prompt = "架空のニュース記事：\n本日午後、東京都内の"
    payload = b"\x00\x42\xff\x10"

    cover_ids = hide_bytes(
        backend,
        prompt,
        payload,
        stego_key=_STEGO_KEY,
        config=config,
        max_tokens=256,
    )
    cover_text = backend.detokenize(cover_ids)
    received_ids = backend.tokenize(cover_text)
    assert received_ids == cover_ids, "real tokenizer transport changed generated token IDs"

    recovered = extract_bytes(
        backend,
        prompt,
        received_ids,
        len(payload),
        stego_key=_STEGO_KEY,
        config=config,
    )
    assert recovered == payload
