from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg import SteganographyConfig, append_sentence_tail, extract_bytes, hide_bytes


@dataclass
class _TailBackend:
    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return 4

    def tokenize(self, text: str) -> list[int]:
        mapping = {"a": 0, "b": 1, "。": 2, "x": 3}
        return [mapping[ch] for ch in text]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        mapping = ["a", "b", "。", "x"]
        return "".join(mapping[token_id] for token_id in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        # Tail sampling strongly prefers the Japanese sentence terminator.
        return Logits.from_values([0.0, 0.0, 8.0, -4.0])


def test_sentence_tail_ends_at_punctuation_and_payload_still_extracts() -> None:
    backend = _TailBackend()
    config = SteganographyConfig(top_k=4, frequency_total=64)
    key = bytes(range(32))
    payload = b"A"

    cover = hide_bytes(backend, "a", payload, stego_key=key, config=config, max_tokens=64)
    tailed = append_sentence_tail(
        backend,
        "a",
        cover,
        seed_key=key,
        config=config,
        max_tail_tokens=8,
    )

    assert backend.detokenize(tailed).endswith("。")
    assert (
        extract_bytes(backend, "a", tailed, len(payload), stego_key=key, config=config) == payload
    )


@dataclass
class _AsciiTailBackend:
    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return 3

    def tokenize(self, text: str) -> list[int]:
        mapping = {"a": 0, "x": 1, ".": 2}
        return [mapping[ch] for ch in text]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        mapping = ["a", "x", "."]
        return "".join(mapping[token_id] for token_id in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        # Deterministic-ish tail: period dominates and must terminate immediately.
        return Logits.from_values([-4.0, -4.0, 8.0])


def test_sentence_tail_accepts_ascii_period_as_emergency_termination() -> None:
    backend = _AsciiTailBackend()
    config = SteganographyConfig(top_k=3, frequency_total=64)
    key = bytes(range(32))
    tailed = append_sentence_tail(
        backend,
        "a",
        [1],
        seed_key=key,
        config=config,
        max_tail_tokens=8,
    )
    assert backend.detokenize(tailed).endswith(".")
    assert len(tailed) == 2


@dataclass
class _MergeTailBackend:
    """Tokenizer where ``a`` + ``b`` canonicalizes to the single token ``ab``."""

    pieces: ClassVar[dict[int, str]] = {0: "a", 1: "b", 2: "ab", 3: "c", 4: ""}

    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return len(self.pieces)

    def tokenize(self, text: str) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith("ab", index):
                result.append(2)
                index += 2
            elif text[index] == "a":
                result.append(0)
                index += 1
            elif text[index] == "b":
                result.append(1)
                index += 1
            elif text[index] == "c":
                result.append(3)
                index += 1
            else:
                raise ValueError(text[index:])
        return result

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        del token_ids
        return Logits.from_values([-10.0, 10.0, -10.0, 8.0, 9.0])


def test_tail_uses_same_transport_first_nucleus_fallback_as_payload_channel() -> None:
    backend = _MergeTailBackend()
    config = SteganographyConfig(
        top_k=3,
        top_p=0.80,
        excluded_token_ids=(0, 2),
        enforce_transport_invariance=True,
    )
    tailed = append_sentence_tail(
        backend,
        "a",
        [0],
        seed_key=bytes(range(32)),
        config=config,
        max_tail_tokens=1,
    )
    # The narrow nucleus contains only unsafe/empty continuations. The tail
    # must widen to the configured top-k and keep the sole canonical token c.
    assert tailed == [0, 3]
