"""Cross-feature release-gate tests for patch candidates.

The purpose of this module is to catch combinations that used to pass in
isolation but fail when semantic re-anchoring, Unicode transport filtering,
anti-repeat filtering, incremental inference, and exact byte Range Coding are
used together.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from lsteg.model.errors import ModelInputError
from lsteg.model.interface import IncrementalLogitsSession, Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg import (
    SemanticAnchorPlan,
    SteganographyConfig,
    extract_bytes,
    hide_bytes,
)
from lsteg.steg.engine import InsufficientCoverCapacityError

_KEY = bytes(range(32))
_VOCAB = 96
_PUA_BASE = 0xE000


@dataclass
class _CombinedBackend:
    """Transport-invariant mock whose visible tokens always end a sentence."""

    seed: int = 1701

    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover - protocol only
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover - protocol only
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return _VOCAB

    def tokenize(self, text: str) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(text):
            codepoint = ord(text[index])
            if (
                _PUA_BASE <= codepoint < _PUA_BASE + _VOCAB
                and index + 1 < len(text)
                and text[index + 1] == "。"
            ):
                result.append(codepoint - _PUA_BASE)
                index += 2
                continue
            # Prompt-only characters only need deterministic sender/receiver
            # tokenization. Visible cover text uses the reversible branch above.
            result.append((codepoint % (_VOCAB - 1)) + 1)
            index += 1
        return result or [1]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(_PUA_BASE + token_id) + "。" for token_id in token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        context_seed = self.seed
        for token_id in token_ids[-4:]:
            context_seed = (context_seed * 131 + token_id) & 0xFFFFFFFF
        rng = random.Random(context_seed)
        return Logits.from_values([rng.gauss(0.0, 1.0) for _ in range(_VOCAB)])

    def render_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        assert enable_thinking is False
        return f"SYS:{system_prompt or ''}\nUSER:{user_prompt}\nASSISTANT:"

    def render_chat_continuation(
        self,
        user_prompt: str,
        assistant_prefix: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        assert enable_thinking is False
        return f"SYS:{system_prompt or ''}\nUSER:{user_prompt}\nASSISTANT:{assistant_prefix}"


def _sparse_plan() -> SemanticAnchorPlan:
    return SemanticAnchorPlan(
        system_prompt="設定を維持する。",
        scenario="大学の昼休み。",
        phases=("研究室で作業する。", "昼食へ行く。", "研究室で作業を続ける。"),
        min_tokens_per_phase=2,
        include_full_outline=False,
        repeat_final_phase=True,
    )


def test_combined_channel_survives_text_transport_and_hidden_reanchors() -> None:
    backend = _CombinedBackend()
    plan = _sparse_plan()
    prompt = plan.initial_prompt(backend)
    config = SteganographyConfig(
        top_k=32,
        frequency_total=1024,
        temperature=0.9,
        top_p=0.85,
        presence_penalty=0.4,
        enforce_transport_invariance=True,
        no_repeat_ngram_size=3,
    )
    payload = bytes(range(12))

    cover = hide_bytes(
        backend,
        prompt,
        payload,
        stego_key=_KEY,
        config=config,
        semantic_plan=plan,
        max_tokens=512,
    )
    received = backend.tokenize(backend.detokenize(cover))
    assert received == cover

    recovered = extract_bytes(
        backend,
        prompt,
        received,
        len(payload),
        stego_key=_KEY,
        config=config,
        semantic_plan=plan,
    )
    assert recovered == payload


class _LimitedSession:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.logits = Logits.from_values([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])

    def next_logits(self) -> Logits:
        return self.logits

    def append(self, token_id: int) -> None:
        del token_id
        self.count += 1
        if self.count >= self.limit:
            raise ModelInputError("context exceeds 8 tokens")


@dataclass
class _LimitedBackend:
    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover - protocol only
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover - protocol only
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return 8

    def tokenize(self, text: str) -> list[int]:
        del text
        return [1]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return "x" * len(token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:  # pragma: no cover
        del token_ids
        return Logits.from_values([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])

    def start_incremental_logits(
        self,
        token_ids: Sequence[int],
    ) -> IncrementalLogitsSession:
        del token_ids
        return _LimitedSession(limit=2)


def test_context_limit_is_reported_as_structured_capacity_diagnostic() -> None:
    backend = _LimitedBackend()
    with pytest.raises(InsufficientCoverCapacityError) as caught:
        hide_bytes(
            backend,
            "prompt",
            b"this payload cannot settle in two tokens",
            stego_key=_KEY,
            config=SteganographyConfig(top_k=8, frequency_total=256),
            max_tokens=100,
        )

    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert "model context limit reached" in diagnostics.reason
    assert diagnostics.target_bits > diagnostics.settled_bits
    assert diagnostics.generated_tokens == 2
    assert diagnostics.mean_table_entropy > 0.0
    assert diagnostics.mean_selected_surprisal >= 0.0
    assert "payload settled" in str(caught.value)
