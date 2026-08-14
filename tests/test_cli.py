from __future__ import annotations

from pathlib import Path

import pytest

from lsteg import __version__
from lsteg.cli import main
from lsteg.payload import MASTER_KEY_SIZE, read_master_key


def test_root_help_lists_planned_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    for command in ("keygen", "encode", "decode", "benchmark"):
        assert command in output


@pytest.mark.parametrize("command", ["encode", "decode", "benchmark"])
def test_future_commands_are_explicit_stubs(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([command]) == 2
    assert f"steg {command}: not implemented yet" in capsys.readouterr().out


def test_keygen_creates_default_key_without_disclosing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["keygen"]) == 0

    key_path = tmp_path / "lsteg.key"
    key = read_master_key(key_path)
    output = capsys.readouterr().out
    assert len(key) == MASTER_KEY_SIZE
    assert "lsteg.key" in output
    assert key.hex() not in output


def test_keygen_refuses_to_overwrite_existing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_path = tmp_path / "shared.key"

    assert main(["keygen", "--output", str(key_path)]) == 0
    original = key_path.read_bytes()
    capsys.readouterr()

    assert main(["keygen", "--output", str(key_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "refusing to overwrite" in captured.err
    assert key_path.read_bytes() == original


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])

    assert capsys.readouterr().out.strip() == f"steg {__version__}"
