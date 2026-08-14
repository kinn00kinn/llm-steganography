from __future__ import annotations

import pytest

from lsteg import __version__
from lsteg.cli import main


def test_root_help_lists_planned_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    for command in ("keygen", "encode", "decode", "benchmark"):
        assert command in output


@pytest.mark.parametrize("command", ["keygen", "encode", "decode", "benchmark"])
def test_phase_zero_commands_are_explicit_stubs(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([command]) == 2
    assert f"steg {command}: not implemented in Phase 0" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])

    assert capsys.readouterr().out.strip() == f"steg {__version__}"
