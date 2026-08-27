"""Regression tests for hard transport filtering vs soft naturalness filters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from lsteg.model.incremental import start_incremental_logits
from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg.engine import SteganographyConfig, _position_table

_STEGO_KEY = bytes(range(32))


class _MergeBackend:
    """Tokenizer where ``a`` + ``b`` canonicalizes to the single token ``ab``."""

    pieces: ClassVar[dict[int, str]] = {0: "a", 1: "b", 2: "ab", 3: "c", 4: ""}

    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover - unused by this regression test
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover - unused by this regression test
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
        # token 1 ("b") is highest but is transport-unsafe after a trailing "a";
        # token 3 ("c") is the safe continuation used by the regressions below.
        return Logits.from_values([-10.0, 10.0, -10.0, 8.0, 9.0])


def test_transport_filter_precedes_no_repeat_fail_open() -> None:
    backend = _MergeBackend()
    # Prefix "aca" is canonical.  Appending token 3 recreates the seen bigram
    # (0, 3), while token 1 is not repetitive but is transport-unsafe because
    # the trailing "a" + "b" retokenizes as token 2.  The hard transport filter
    # must run first; then no-repeat may fail open to the sole safe token 3.
    prefix = [0, 3, 0]
    session = start_incremental_logits(backend, prefix)
    ids, _ = _position_table(
        backend,
        session,
        prefix,
        position=0,
        stego_key=_STEGO_KEY,
        config=SteganographyConfig(
            top_k=2,
            top_p=1.0,
            excluded_token_ids=(0, 2, 4),
            enforce_transport_invariance=True,
            no_repeat_ngram_size=2,
        ),
        cover_token_ids=prefix,
    )
    assert ids == [3]


def test_transport_filter_widens_nucleus_to_full_top_k_if_needed() -> None:
    backend = _MergeBackend()
    prefix = [0]
    session = start_incremental_logits(backend, prefix)
    ids, _ = _position_table(
        backend,
        session,
        prefix,
        position=0,
        stego_key=_STEGO_KEY,
        config=SteganographyConfig(
            top_k=3,
            # With logits 10, 9, 8 the narrow nucleus keeps tokens 1 and 4.
            # Both are transport-unsafe (merge / empty text).  Full top-k adds
            # token 3, which is canonical and must be used as deterministic
            # fallback rather than aborting the channel.
            top_p=0.80,
            excluded_token_ids=(0, 2),
            enforce_transport_invariance=True,
        ),
        cover_token_ids=prefix,
    )
    assert ids == [3]
