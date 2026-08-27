"""Teacher-data schema and mechanical quality filters."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TEACHER_DATA_SCHEMA_VERSION = 1

_DISALLOWED_SCRIPT = re.compile(r"[\u0400-\u052f\u0600-\u06ff\u1100-\u11ff\uac00-\ud7af]")
_LONG_LATIN = re.compile(r"[A-Za-z]{6,}")
_MARKDOWN_LINE = re.compile(r"(?m)^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)")
_META_TERMS = (
    "<think>",
    "</think>",
    "以下の文章",
    "ご指定",
    "ご依頼",
    "本文を書き",
    "文章を作成",
    "語り手",
)
_SENTENCE_SPLIT = re.compile(r"[。！？!?]+")  # noqa: RUF001


@dataclass(frozen=True, slots=True)
class TeacherValidationPolicy:
    min_chars: int = 220
    max_chars: int = 650
    min_japanese_ratio: float = 0.90


@dataclass(frozen=True, slots=True)
class TeacherExample:
    schema_version: int
    scenario_id: str
    category: str
    split: str
    prompt: str
    completion: str
    teacher_model: str
    generation_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != TEACHER_DATA_SCHEMA_VERSION:
            raise ValueError(f"unsupported teacher-data schema: {self.schema_version}")
        if not self.scenario_id or not self.category or not self.prompt or not self.completion:
            raise ValueError("teacher example text fields must not be empty")
        if self.split not in {"train", "eval"}:
            raise ValueError("teacher example split must be train or eval")
        if not self.teacher_model:
            raise ValueError("teacher_model must not be empty")
        if not 0 <= self.generation_seed <= 2**31 - 1:
            raise ValueError("generation_seed is out of range")

    @classmethod
    def from_mapping(cls, raw: object) -> TeacherExample:
        if not isinstance(raw, dict):
            raise ValueError("teacher JSONL row must be an object")
        expected = {
            "schema_version",
            "scenario_id",
            "category",
            "split",
            "prompt",
            "completion",
            "teacher_model",
            "generation_seed",
        }
        if set(raw) != expected:
            raise ValueError("teacher JSONL row has missing or unknown fields")
        try:
            return cls(**raw)
        except TypeError as error:
            raise ValueError("teacher JSONL row has invalid field types") from error

    def as_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "category": self.category,
            "split": self.split,
            "prompt": self.prompt,
            "completion": self.completion,
            "teacher_model": self.teacher_model,
            "generation_seed": self.generation_seed,
        }


def validate_teacher_completion(
    text: str,
    policy: TeacherValidationPolicy,
) -> tuple[str, tuple[str, ...]]:
    """Normalize a candidate and return deterministic rejection reasons."""
    normalized = unicodedata.normalize("NFC", text).strip()
    reasons: list[str] = []
    if not policy.min_chars <= len(normalized) <= policy.max_chars:
        reasons.append("length")
    if any(term in normalized for term in _META_TERMS):
        reasons.append("meta")
    if _DISALLOWED_SCRIPT.search(normalized):
        reasons.append("foreign-script")
    if _LONG_LATIN.search(normalized):
        reasons.append("long-latin-run")
    if _MARKDOWN_LINE.search(normalized):
        reasons.append("markdown")
    if _japanese_ratio(normalized) < policy.min_japanese_ratio:
        reasons.append("japanese-ratio")
    if _has_repeated_sentence(normalized):
        reasons.append("repeated-sentence")
    return normalized, tuple(sorted(set(reasons)))


def load_teacher_examples(path: Path) -> list[TeacherExample]:
    examples: list[TeacherExample] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read teacher dataset: {path}") from error
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            examples.append(TeacherExample.from_mapping(raw))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid teacher dataset row {line_no}") from error
    if not examples:
        raise ValueError("teacher dataset is empty")
    return examples


def audit_teacher_examples(
    examples: Iterable[TeacherExample],
    policy: TeacherValidationPolicy,
) -> dict[str, int | float]:
    items = list(examples)
    if not items:
        raise ValueError("cannot audit an empty dataset")
    rejection_counts: Counter[str] = Counter()
    char_counts: list[int] = []
    scenario_completions: Counter[str] = Counter()
    seen_completions: dict[str, set[str]] = {}
    duplicate_completions = 0
    invalid = 0
    for item in items:
        normalized, reasons = validate_teacher_completion(item.completion, policy)
        char_counts.append(len(normalized))
        scenario_completions[item.scenario_id] += 1
        rejection_counts.update(reasons)
        invalid += bool(reasons)
        scenario_seen = seen_completions.setdefault(item.scenario_id, set())
        if normalized in scenario_seen:
            duplicate_completions += 1
        scenario_seen.add(normalized)
    return {
        "examples": len(items),
        "train_examples": sum(item.split == "train" for item in items),
        "eval_examples": sum(item.split == "eval" for item in items),
        "scenarios": len(scenario_completions),
        "invalid_examples": invalid,
        "duplicate_completions": duplicate_completions,
        "mean_chars": sum(char_counts) / len(char_counts),
        "min_completions_per_scenario": min(scenario_completions.values()),
        "max_completions_per_scenario": max(scenario_completions.values()),
        **{f"reject_{key}": value for key, value in sorted(rejection_counts.items())},
    }


def _japanese_ratio(text: str) -> float:
    japanese = 0
    alphabetic = 0
    for char in text:
        codepoint = ord(char)
        is_japanese = (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
        if is_japanese:
            japanese += 1
            alphabetic += 1
        elif char.isalpha():
            alphabetic += 1
    return 1.0 if alphabetic == 0 else japanese / alphabetic


def _has_repeated_sentence(text: str) -> bool:
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        normalized = "".join(sentence.split())
        if len(normalized) < 12:
            continue
        if normalized in seen:
            return True
        seen.add(normalized)
    return False
