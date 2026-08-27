from __future__ import annotations

from collections.abc import Sequence

from lsteg.model.incremental import start_incremental_logits
from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest


class _FallbackBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

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
        return [0]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return ""

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        self.calls.append(tuple(token_ids))
        return Logits.from_values([0.0, 1.0, 2.0])


class _NativeSession:
    def __init__(self) -> None:
        self.appended: list[int] = []

    def next_logits(self) -> Logits:
        return Logits.from_values([2.0, 1.0, 0.0])

    def append(self, token_id: int) -> None:
        self.appended.append(token_id)


class _NativeBackend(_FallbackBackend):
    def __init__(self) -> None:
        super().__init__()
        self.session = _NativeSession()
        self.started_with: tuple[int, ...] | None = None

    def start_incremental_logits(self, token_ids: Sequence[int]) -> _NativeSession:
        self.started_with = tuple(token_ids)
        return self.session


def test_incremental_fallback_recomputes_current_prefix() -> None:
    backend = _FallbackBackend()
    session = start_incremental_logits(backend, [0, 1])

    assert session.next_logits().argmax_token_id == 2
    session.append(2)
    assert session.next_logits().argmax_token_id == 2
    assert backend.calls == [(0, 1), (0, 1, 2)]


def test_incremental_selector_prefers_native_backend_session() -> None:
    backend = _NativeBackend()
    session = start_incremental_logits(backend, [0, 1])

    assert session is backend.session
    assert backend.started_with == (0, 1)
    assert session.next_logits().argmax_token_id == 0
    session.append(2)
    assert backend.session.appended == [2]
    assert backend.calls == []
