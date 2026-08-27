from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsteg.training.data import (
    TEACHER_DATA_SCHEMA_VERSION,
    TeacherExample,
    TeacherValidationPolicy,
    audit_teacher_examples,
    load_teacher_examples,
    validate_teacher_completion,
)


def _natural_text() -> str:
    return (
        "正午前、私は研究室で午前中の作業を続けていた。時計を見ると昼休みが近かったので、"
        "保存していなかった資料を確認してから席を立った。廊下で友人と会い、二人で学食へ向かった。"
        "学食では混雑していない列を選び、私は定食、友人は麺類を注文した。食事中は午前中の課題について"
        "少し話しただけで、特別な出来事はなかった。食べ終えると同じ道を研究室へ戻り、机に座って作業を再開した。"
    )


def test_teacher_filter_accepts_plain_japanese_prose() -> None:
    normalized, reasons = validate_teacher_completion(
        _natural_text(),
        TeacherValidationPolicy(min_chars=100, max_chars=650, min_japanese_ratio=0.9),
    )
    assert normalized == _natural_text()
    assert reasons == ()


@pytest.mark.parametrize(
    ("suffix", "reason"),
    [
        (" Tulsa Oklahoma", "long-latin-run"),
        (" 학식에 간다", "foreign-script"),
        (" <think>内部推論</think>", "meta"),
        ("\n- 箇条書き", "markdown"),
    ],
)
def test_teacher_filter_rejects_observed_failure_modes(suffix: str, reason: str) -> None:
    _, reasons = validate_teacher_completion(
        _natural_text() + suffix,
        TeacherValidationPolicy(min_chars=100, max_chars=800, min_japanese_ratio=0.75),
    )
    assert reason in reasons


def test_teacher_filter_rejects_exact_sentence_loop() -> None:
    sentence = "研究室へ戻ったあと、私は机に座って資料の整理を続けた。"
    text = _natural_text() + sentence + sentence
    _, reasons = validate_teacher_completion(
        text, TeacherValidationPolicy(min_chars=100, max_chars=800, min_japanese_ratio=0.9)
    )
    assert "repeated-sentence" in reasons


def test_teacher_jsonl_round_trip_and_audit(tmp_path: Path) -> None:
    examples = [
        TeacherExample(
            schema_version=TEACHER_DATA_SCHEMA_VERSION,
            scenario_id="scenario-a",
            category="大学での作業",
            split="train",
            prompt="自然な文章を書いてください。",
            completion=_natural_text(),
            teacher_model="qwen2.5:7b",
            generation_seed=1,
        ),
        TeacherExample(
            schema_version=TEACHER_DATA_SCHEMA_VERSION,
            scenario_id="scenario-b",
            category="買い物",
            split="eval",
            prompt="自然な文章を書いてください。",
            completion=_natural_text(),
            teacher_model="qwen2.5:7b",
            generation_seed=2,
        ),
    ]
    path = tmp_path / "teacher.jsonl"
    path.write_text(
        "".join(json.dumps(item.as_dict(), ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )
    loaded = load_teacher_examples(path)
    assert loaded == examples
    report = audit_teacher_examples(
        loaded, TeacherValidationPolicy(min_chars=100, max_chars=650, min_japanese_ratio=0.9)
    )
    assert report["examples"] == 2
    assert report["train_examples"] == 1
    assert report["eval_examples"] == 1
    assert report["invalid_examples"] == 0


def test_teacher_audit_counts_duplicate_completions() -> None:
    repeated = TeacherExample(
        schema_version=TEACHER_DATA_SCHEMA_VERSION,
        scenario_id="same-scenario",
        category="大学での作業",
        split="train",
        prompt="自然な文章を書いてください。",
        completion=_natural_text(),
        teacher_model="qwen2.5:7b",
        generation_seed=1,
    )
    duplicate = TeacherExample(
        schema_version=TEACHER_DATA_SCHEMA_VERSION,
        scenario_id="same-scenario",
        category="大学での作業",
        split="train",
        prompt="自然な文章を書いてください。",
        completion=_natural_text(),
        teacher_model="qwen2.5:7b",
        generation_seed=2,
    )
    report = audit_teacher_examples(
        [repeated, duplicate],
        TeacherValidationPolicy(min_chars=100, max_chars=650, min_japanese_ratio=0.9),
    )
    assert report["duplicate_completions"] == 1
