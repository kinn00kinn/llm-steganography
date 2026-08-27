from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lsteg.training import TEACHER_DATA_SCHEMA_VERSION, TeacherExample

ROOT = Path(__file__).resolve().parents[2]


def _long_completion(label: str) -> str:
    # One long sentence avoids the exact-sentence-repeat filter while remaining
    # mechanically valid Japanese for CLI dry-run tests.
    body = (
        f"{label}では研究室で資料を確認し、昼前になったので作業を保存してから友人と学食へ向かい、"
        "混雑していない席で昼食を取りながら午前中の作業について短く話し、食後は同じ道を戻って"
        "机の上の資料を整理し直し、次に確認する項目をノートへ書いてから落ち着いて作業を再開し、"
        "途中で気づいた小さな修正点を一つ直して保存し、友人にも結果だけを簡単に伝えてから、"
        "残っていた確認作業を静かに進め、特別な出来事を加えずに予定していた範囲を終えた"
    )
    return body + "という平凡な出来事を、場所や人物を変えずに順番どおり振り返った。"


def _write_dataset(path: Path) -> None:
    examples = [
        TeacherExample(
            schema_version=TEACHER_DATA_SCHEMA_VERSION,
            scenario_id="train-scenario",
            category="大学での作業",
            split="train",
            prompt="正午前の研究室で起きた日常を書く。",
            completion=_long_completion("今日は"),
            teacher_model="qwen2.5:7b",
            generation_seed=1,
        ),
        TeacherExample(
            schema_version=TEACHER_DATA_SCHEMA_VERSION,
            scenario_id="eval-scenario",
            category="大学での作業",
            split="eval",
            prompt="昼食後に研究室へ戻る日常を書く。",
            completion=_long_completion("昨日は"),
            teacher_model="qwen2.5:7b",
            generation_seed=2,
        ),
    ]
    path.write_text(
        "".join(json.dumps(item.as_dict(), ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )


def test_training_cli_dry_run_does_not_import_gpu_stack(tmp_path: Path) -> None:
    dataset = tmp_path / "teacher.jsonl"
    _write_dataset(dataset)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_japanese_lora.py"),
            str(dataset),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["train_examples"] == 1
    assert report["recipe"]["lora_rank"] == 8
    assert report["recipe"]["label_smoothing"] == 0.0
    assert report["recipe"]["kl_weight"] == 0.2


def test_merge_cli_refuses_missing_evaluation_before_optional_imports(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/merge_japanese_lora.py"), str(adapter)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "evaluation report" in completed.stderr


def test_merge_cli_binds_passing_report_to_exact_adapter_bytes(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-a")
    evaluation = {
        "gate_pass": True,
        "adapter_sha256": "not-the-adapter-hash",
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/merge_japanese_lora.py"), str(adapter)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "does not match this LoRA adapter" in completed.stderr


def test_kl_sweep_cli_dry_run_is_model_free(tmp_path: Path) -> None:
    output_root = tmp_path / "sweep"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sweep_japanese_lora_kl.py"),
            str(tmp_path / "unused.jsonl"),
            "--profiles",
            "kl020",
            "kl100",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert [item["name"] for item in report["profiles"]] == ["kl020", "kl100"]
    assert not output_root.exists()
