"""Incremental next-token logits with an optional backend KV cache.

Steganography needs the next-token distribution at every generated position.
Backends that expose a native incremental session can reuse transformer KV
states; model-neutral/mock backends fall back to recomputing the full prefix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lsteg.model.interface import (
    IncrementalLanguageModelBackend,
    IncrementalLogitsSession,
    LanguageModelBackend,
    Logits,
)


def start_incremental_logits(
    backend: LanguageModelBackend,
    token_ids: Sequence[int],
) -> IncrementalLogitsSession:
    """Start a next-logit session, using native KV caching when available."""
    if isinstance(backend, IncrementalLanguageModelBackend):
        return backend.start_incremental_logits(token_ids)
    return _FallbackIncrementalLogitsSession(backend, list(token_ids))


@dataclass(slots=True)
class _FallbackIncrementalLogitsSession:
    backend: LanguageModelBackend
    token_ids: list[int]

    def next_logits(self) -> Logits:
        return self.backend.next_logits(self.token_ids)

    def append(self, token_id: int) -> None:
        self.token_ids.append(token_id)
