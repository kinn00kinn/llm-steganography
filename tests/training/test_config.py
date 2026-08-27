from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsteg.training.config import TrainingRecipe

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "filename",
    [
        "qwen3-1.7b-japanese-prose-lora-v1.json",
        "qwen3-1.7b-japanese-prose-lora-v2.json",
        "qwen3-1.7b-japanese-prose-lora-v3-kl050.json",
        "qwen3-1.7b-japanese-prose-lora-v3-kl100.json",
        "qwen3-1.7b-japanese-prose-lora-r8-sft.json",
        "qwen3-1.7b-japanese-prose-lora-r16-kl.json",
    ],
)
def test_repository_training_recipes_are_valid(filename: str) -> None:
    recipe = TrainingRecipe.from_path(ROOT / "config/training" / filename)
    assert recipe.base_model_id == "Qwen/Qwen3-1.7B"
    assert recipe.target_modules == "all-linear"


def test_preferred_recipe_uses_rank8_without_uniform_label_smoothing() -> None:
    recipe = TrainingRecipe.from_path(
        ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json"
    )
    assert recipe.lora_rank == 8
    assert recipe.label_smoothing == pytest.approx(0.0)
    assert recipe.kl_weight == pytest.approx(0.2)
    assert recipe.kl_positions_per_example == 8


def test_v1_recipe_is_retained_for_reproducibility() -> None:
    recipe = TrainingRecipe.from_path(
        ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v1.json"
    )
    assert recipe.label_smoothing == pytest.approx(0.05)


def test_training_recipe_rejects_unknown_fields(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json").read_text(
            encoding="utf-8"
        )
    )
    source["surprise"] = True
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        TrainingRecipe.from_path(path)


def test_training_recipe_rejects_model_substitution(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "config/training/qwen3-1.7b-japanese-prose-lora-v1.json").read_text(
            encoding="utf-8"
        )
    )
    source["base_model_id"] = "Qwen/Qwen3-4B"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Qwen3-1[.]7B"):
        TrainingRecipe.from_path(path)


def test_v3_kl_sweep_recipes_change_only_kl_weight() -> None:
    root = Path(__file__).resolve().parents[2]
    recipe_paths = [
        root / "config/training/qwen3-1.7b-japanese-prose-lora-v2.json",
        root / "config/training/qwen3-1.7b-japanese-prose-lora-v3-kl050.json",
        root / "config/training/qwen3-1.7b-japanese-prose-lora-v3-kl100.json",
    ]
    recipes = [TrainingRecipe.from_path(path).as_dict() for path in recipe_paths]
    weights = [float(recipe.pop("kl_weight")) for recipe in recipes]
    assert weights == [0.2, 0.5, 1.0]
    assert recipes[0] == recipes[1] == recipes[2]
