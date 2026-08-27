"""Strict recipe configuration for Japanese-prose QLoRA experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TRAINING_RECIPE_SCHEMA_VERSION = 1
_RECIPE_FIELDS = {
    "schema_version",
    "base_model_id",
    "teacher_model",
    "teacher_scenarios",
    "teacher_completions_per_scenario",
    "min_chars",
    "max_chars",
    "min_japanese_ratio",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "learning_rate",
    "epochs",
    "gradient_accumulation_steps",
    "max_sequence_tokens",
    "label_smoothing",
    "kl_weight",
    "kl_positions_per_example",
    "seed",
}


@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    """Versioned QLoRA recipe selected for the fixed Qwen3-1.7B experiment."""

    schema_version: int
    base_model_id: str
    teacher_model: str
    teacher_scenarios: int
    teacher_completions_per_scenario: int
    min_chars: int
    max_chars: int
    min_japanese_ratio: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    learning_rate: float
    epochs: int
    gradient_accumulation_steps: int
    max_sequence_tokens: int
    label_smoothing: float
    kl_weight: float
    kl_positions_per_example: int
    seed: int

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_RECIPE_SCHEMA_VERSION:
            raise ValueError(f"unsupported training recipe schema: {self.schema_version}")
        if self.base_model_id != "Qwen/Qwen3-1.7B":
            raise ValueError("training recipe must keep the development model at Qwen/Qwen3-1.7B")
        if not self.teacher_model:
            raise ValueError("teacher_model must not be empty")
        if not 1 <= self.teacher_scenarios <= 10_000:
            raise ValueError("teacher_scenarios must be between 1 and 10000")
        if not 1 <= self.teacher_completions_per_scenario <= 16:
            raise ValueError("teacher_completions_per_scenario must be between 1 and 16")
        if not 100 <= self.min_chars < self.max_chars <= 2_000:
            raise ValueError("teacher character bounds are invalid")
        if not 0.5 <= self.min_japanese_ratio <= 1.0:
            raise ValueError("min_japanese_ratio must be between 0.5 and 1.0")
        if self.lora_rank not in {4, 8, 16, 32}:
            raise ValueError("lora_rank must be one of 4, 8, 16, 32")
        if self.lora_alpha < self.lora_rank:
            raise ValueError("lora_alpha must be at least lora_rank")
        if not 0.0 <= self.lora_dropout < 0.5:
            raise ValueError("lora_dropout must be in [0, 0.5)")
        if self.target_modules != "all-linear":
            raise ValueError("v1 recipe requires target_modules='all-linear'")
        if not 0.0 < self.learning_rate <= 1e-2:
            raise ValueError("learning_rate is out of range")
        if not 1 <= self.epochs <= 20:
            raise ValueError("epochs must be between 1 and 20")
        if not 1 <= self.gradient_accumulation_steps <= 128:
            raise ValueError("gradient_accumulation_steps must be between 1 and 128")
        if not 128 <= self.max_sequence_tokens <= 2048:
            raise ValueError("max_sequence_tokens must be between 128 and 2048")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if not 0.0 <= self.kl_weight <= 10.0:
            raise ValueError("kl_weight must be in [0, 10]")
        if not 0 <= self.kl_positions_per_example <= 128:
            raise ValueError("kl_positions_per_example must be between 0 and 128")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed must fit a signed 32-bit integer")

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "schema_version": self.schema_version,
            "base_model_id": self.base_model_id,
            "teacher_model": self.teacher_model,
            "teacher_scenarios": self.teacher_scenarios,
            "teacher_completions_per_scenario": self.teacher_completions_per_scenario,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "min_japanese_ratio": self.min_japanese_ratio,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_sequence_tokens": self.max_sequence_tokens,
            "label_smoothing": self.label_smoothing,
            "kl_weight": self.kl_weight,
            "kl_positions_per_example": self.kl_positions_per_example,
            "seed": self.seed,
        }

    @classmethod
    def from_path(cls, path: Path) -> TrainingRecipe:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read training recipe: {path}") from error
        if not isinstance(raw, dict):
            raise ValueError("training recipe root must be an object")
        fields = set(raw)
        if fields != _RECIPE_FIELDS:
            missing = sorted(_RECIPE_FIELDS - fields)
            unknown = sorted(fields - _RECIPE_FIELDS)
            raise ValueError(f"training recipe fields differ; missing={missing}, unknown={unknown}")
        try:
            return cls(**raw)
        except TypeError as error:
            raise ValueError("training recipe contains invalid field types") from error
