from __future__ import annotations

from pathlib import Path

import pytest

from lsteg.training.artifacts import hash_file, hash_tree


def test_artifact_hashes_are_deterministic_and_path_sensitive(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    first = hash_tree(root)
    assert first == hash_tree(root)
    assert hash_file(root / "a.txt") != hash_file(root / "b.txt")
    (root / "b.txt").write_text("changed", encoding="utf-8")
    assert first != hash_tree(root)


def test_hash_tree_can_exclude_generated_metadata(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"weights")
    (root / "REPORT.json").write_text("one", encoding="utf-8")
    first = hash_tree(root, exclude_names={"REPORT.json"})
    (root / "REPORT.json").write_text("two", encoding="utf-8")
    assert first == hash_tree(root, exclude_names={"REPORT.json"})


def test_hash_tree_rejects_empty_or_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        hash_tree(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        hash_tree(empty)
