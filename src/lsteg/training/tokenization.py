"""Model-neutral helpers for completion-only chat training."""

from __future__ import annotations

from typing import Protocol

from lsteg.training.prompts import JAPANESE_PROSE_SYSTEM_PROMPT


class ChatTokenizer(Protocol):
    """Minimal tokenizer surface needed by the LoRA scripts."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def build_completion_training_ids(
    tokenizer: ChatTokenizer,
    prompt: str,
    completion: str,
    *,
    max_tokens: int,
) -> tuple[list[int], list[int]]:
    """Return inference-identical prompt IDs plus separately tokenized completion IDs.

    The completion is tokenized separately on purpose.  Inference first consumes the
    rendered chat prompt and then generates a new token; allowing a tokenizer to
    retokenize the prompt/completion boundary would train a trajectory the decoder
    can never reproduce.
    """
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": JAPANESE_PROSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    if not prompt_ids or not completion_ids:
        raise ValueError("training example tokenized to an empty prompt/completion")
    if len(prompt_ids) + len(completion_ids) > max_tokens:
        raise ValueError(
            f"training example exceeds max_tokens={max_tokens}: "
            f"prompt={len(prompt_ids)}, completion={len(completion_ids)}"
        )
    return prompt_ids, completion_ids
