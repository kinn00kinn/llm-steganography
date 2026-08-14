"""Command-line entry point for the staged implementation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lsteg import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI without executing a command."""
    parser = argparse.ArgumentParser(
        prog="steg",
        description="Shared-key linguistic steganography research toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_stub(commands, "keygen", "Generate a shared master key (Phase 2).")
    _add_stub(commands, "encode", "Encode a secret into cover text (Phase 6+).")
    _add_stub(commands, "decode", "Decode a secret from cover text (Phase 6+).")
    _add_stub(commands, "benchmark", "Run capacity and quality benchmarks (Phase 5+).")
    return parser


def _add_stub(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> None:
    command = commands.add_parser(name, help=help_text, description=help_text)
    command.set_defaults(handler=_not_implemented)


def _not_implemented(args: argparse.Namespace) -> int:
    command = str(args.command)
    print(f"steg {command}: not implemented in Phase 0")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
