"""Transport-safe candidate filtering tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from lsteg.coding.frequencies import FrequencyTable
from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg.transport import (
    NoTransportSafeCandidatesError,
    filter_transport_safe_candidates,
)


class _MergeBackend:
    """Tiny tokenizer where token 0 + token 1 canonicalizes to token 2."""

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

    def next_logits(self, token_ids: Sequence[int]) -> Logits:  # pragma: no cover
        return Logits.from_values([0.0] * len(self.pieces))


def test_filter_rejects_cross_boundary_retokenization() -> None:
    backend = _MergeBackend()
    ids, table = filter_transport_safe_candidates(
        backend,
        [0],
        [1, 3],
        FrequencyTable([7, 5]),
    )
    assert ids == [3]
    assert table.frequencies == (5,)


def test_filter_keeps_canonical_extensions_and_relative_frequencies() -> None:
    backend = _MergeBackend()
    ids, table = filter_transport_safe_candidates(
        backend,
        [],
        [0, 2, 3],
        FrequencyTable([11, 7, 3]),
    )
    assert ids == [0, 2, 3]
    assert table.frequencies == (11, 7, 3)


def test_filter_rejects_empty_text_token() -> None:
    backend = _MergeBackend()
    ids, table = filter_transport_safe_candidates(
        backend,
        [],
        [4, 3],
        FrequencyTable([9, 2]),
    )
    assert ids == [3]
    assert table.frequencies == (2,)


def test_filter_fails_closed_if_no_candidate_is_safe() -> None:
    backend = _MergeBackend()
    with pytest.raises(NoTransportSafeCandidatesError):
        filter_transport_safe_candidates(
            backend,
            [0],
            [1],
            FrequencyTable([1]),
        )


class _BatchMergeBackend(_MergeBackend):
    def __init__(self) -> None:
        self.batch_calls = 0

    def transport_safe_candidate_ids(
        self,
        cover_prefix_token_ids: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> list[int]:
        self.batch_calls += 1
        safe: list[int] = []
        prefix = list(cover_prefix_token_ids)
        for token_id in candidate_token_ids:
            trial = [*prefix, token_id]
            text = self.detokenize(trial)
            if text and self.tokenize(text) == trial:
                safe.append(token_id)
        return safe


def test_filter_prefers_backend_batch_transport_path() -> None:
    backend = _BatchMergeBackend()
    ids, table = filter_transport_safe_candidates(
        backend,
        [0],
        [1, 3],
        FrequencyTable([7, 5]),
    )

    assert backend.batch_calls == 1
    assert ids == [3]
    assert table.frequencies == (5,)
