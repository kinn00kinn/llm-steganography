from __future__ import annotations

import pytest

from lsteg.training.tokenization import build_completion_training_ids


class _FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return "PROMPT|"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        mapping = {"PROMPT|": [1, 2, 3], "本文": [10, 11]}
        return mapping[text]


def test_completion_training_ids_preserve_inference_boundary() -> None:
    prompt_ids, completion_ids = build_completion_training_ids(
        _FakeTokenizer(), "ignored", "本文", max_tokens=16
    )
    assert prompt_ids == [1, 2, 3]
    assert completion_ids == [10, 11]


def test_completion_training_ids_reject_context_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        build_completion_training_ids(_FakeTokenizer(), "ignored", "本文", max_tokens=4)
