"""Command-line entry point for the staged implementation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lsteg import __version__
from lsteg.payload import KeyManagementError, create_master_key_file


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI without executing a command."""
    parser = argparse.ArgumentParser(
        prog="steg",
        description="Shared-key linguistic steganography research toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    keygen = commands.add_parser(
        "keygen",
        help="Generate a shared master key.",
        description="Generate a versioned 256-bit master-key file without overwriting it.",
    )
    keygen.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("lsteg.key"),
        metavar="PATH",
        help="Key file to create (default: lsteg.key).",
    )
    keygen.set_defaults(handler=_keygen)
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
    print(f"steg {command}: not implemented yet")
    return 2


def _keygen(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        created = create_master_key_file(output)
    except KeyManagementError as error:
        print(f"steg keygen: {error}", file=sys.stderr)
        return 1
    print(f"Created master key file: {created}")
    return 0


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
