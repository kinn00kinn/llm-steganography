"""Tests for deterministic semantic re-anchoring."""

from __future__ import annotations

import random
from collections.abc import Sequence

from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest
from lsteg.steg import (
    SemanticAnchorPlan,
    SteganographyConfig,
    extract_bytes,
    hide_bytes,
)

_KEY = bytes(range(32))


class _SemanticMockBackend:
    """Tiny backend whose visible tokens always form sentence boundaries."""

    @property
    def manifest(self) -> ModelManifest:  # pragma: no cover - protocol only
        raise NotImplementedError

    @property
    def runtime(self) -> RuntimeFingerprint:  # pragma: no cover - protocol only
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        return 64

    def tokenize(self, text: str) -> list[int]:
        return [sum(text.encode("utf-8")) % 63 + 1]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return "。" * len(token_ids)

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        rng = random.Random(991 + token_ids[-1])
        return Logits.from_values([rng.gauss(0.0, 1.0) for _ in range(64)])

    def render_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        del enable_thinking
        return f"<system>{system_prompt or ''}</system><user>{user_prompt}</user><assistant>"

    def render_chat_continuation(
        self,
        user_prompt: str,
        assistant_prefix: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        del enable_thinking
        return (
            f"<system>{system_prompt or ''}</system><user>{user_prompt}</user>"
            f"<assistant>{assistant_prefix}"
        )


def _plan(min_tokens: int = 1) -> SemanticAnchorPlan:
    return SemanticAnchorPlan(
        system_prompt="設定を維持する。",
        scenario="大学の昼休み。",
        phases=("研究室。", "学食へ行く。", "研究室へ戻る。"),
        min_tokens_per_phase=min_tokens,
    )


def test_semantic_state_advances_only_at_sentence_boundary() -> None:
    backend = _SemanticMockBackend()
    plan = _plan(min_tokens=2)
    state = plan.new_state()

    assert state.maybe_advance(backend, [], position=0) is None
    assert state.maybe_advance(backend, [1], position=1) is None
    replacement = state.maybe_advance(backend, [1, 2], position=2)
    assert replacement is not None
    assert state.phase_index == 1
    assert "学食へ行く" in replacement


def test_current_phase_replays_boundaries() -> None:
    backend = _SemanticMockBackend()
    plan = _plan(min_tokens=2)
    assert plan.current_phase_for_cover(backend, [1]) == 0
    assert plan.current_phase_for_cover(backend, [1, 2]) == 1
    assert plan.current_phase_for_cover(backend, [1, 2, 3, 4]) == 2


def test_continuation_prompt_prefills_visible_cover_as_assistant() -> None:
    backend = _SemanticMockBackend()
    plan = _plan(min_tokens=1)
    replacement = plan.continuation_prompt(backend, [1, 2], phase_index=1)

    assert "<assistant>。。" in replacement
    assert "ここまでの本文" not in replacement
    assert "学食へ行く" in replacement


def test_closure_prompt_prefills_cover_and_requests_short_japanese_ending() -> None:
    backend = _SemanticMockBackend()
    plan = _plan(min_tokens=1)
    replacement = plan.closure_prompt(backend, [1, 2], phase_index=1)

    assert "<assistant>。。" in replacement
    assert "句点「。」" in replacement
    assert "英語表現" in replacement


def test_hide_extract_bytes_roundtrip_with_hidden_reanchors() -> None:
    backend = _SemanticMockBackend()
    plan = _plan(min_tokens=1)
    prompt = plan.initial_prompt(backend)
    config = SteganographyConfig(top_k=16, frequency_total=256)
    payload = b"semantic-anchor-roundtrip"

    cover = hide_bytes(
        backend,
        prompt,
        payload,
        stego_key=_KEY,
        config=config,
        semantic_plan=plan,
        max_tokens=1024,
    )
    recovered = extract_bytes(
        backend,
        prompt,
        cover,
        len(payload),
        stego_key=_KEY,
        config=config,
        semantic_plan=plan,
    )
    assert recovered == payload


def test_current_phase_replay_does_not_render_continuation_prompts() -> None:
    class _CountingBackend(_SemanticMockBackend):
        continuation_calls = 0

        def render_chat_continuation(
            self,
            user_prompt: str,
            assistant_prefix: str,
            *,
            system_prompt: str | None = None,
            enable_thinking: bool = False,
        ) -> str:
            self.continuation_calls += 1
            return super().render_chat_continuation(
                user_prompt,
                assistant_prefix,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
            )

    backend = _CountingBackend()
    plan = _plan(min_tokens=1)
    assert plan.current_phase_for_cover(backend, [1, 2, 3]) == 2
    assert backend.continuation_calls == 0


def test_sparse_initial_prompt_hides_future_outline() -> None:
    backend = _SemanticMockBackend()
    plan = SemanticAnchorPlan(
        system_prompt="設定を維持する。",
        scenario="大学の昼休み。",
        phases=("最初の出来事。", "次の出来事。", "最後の出来事。"),
        min_tokens_per_phase=1,
        include_full_outline=False,
    )

    prompt = plan.initial_prompt(backend)
    assert "最初の出来事" in prompt
    assert "次の出来事。" not in prompt
    assert "最後の出来事" not in prompt
    assert "具体的な言い回しや細部" in prompt


def test_repeat_final_phase_reanchors_without_advancing_past_plan() -> None:
    backend = _SemanticMockBackend()
    plan = SemanticAnchorPlan(
        system_prompt="設定を維持する。",
        scenario="大学の昼休み。",
        phases=("昼食へ行く。", "研究室で作業を続ける。"),
        min_tokens_per_phase=1,
        include_full_outline=False,
        repeat_final_phase=True,
    )
    state = plan.new_state()

    first = state.maybe_advance(backend, [1], position=1)
    assert first is not None
    assert state.phase_index == 1
    second = state.maybe_advance(backend, [1, 2], position=2)
    assert second is not None
    assert state.phase_index == 1
    assert "研究室で作業を続ける" in second


def test_guarded_sparse_plan_keeps_hard_invariants_without_full_outline() -> None:
    from lsteg.steg import university_lunch_guarded_sparse_cover_plan

    backend = _SemanticMockBackend()
    plan = university_lunch_guarded_sparse_cover_plan(min_tokens_per_phase=2)
    prompt = plan.initial_prompt(backend)

    assert plan.include_full_outline is False
    assert plan.repeat_final_phase is True
    assert "研究室では食べません" in prompt
    assert "身体接触" in prompt
    assert "高校" in prompt
    assert "現在の目標" in prompt
    assert "それぞれ昼食を選んで" not in prompt  # later phase wording is hidden
