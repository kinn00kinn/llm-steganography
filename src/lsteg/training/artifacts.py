"""Deterministic hashes for local training artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, *, exclude_names: Iterable[str] = ()) -> str:
    """Hash relative paths and bytes of all files below ``root`` deterministically."""
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    excluded = set(exclude_names)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded)
    if not files:
        raise ValueError(f"artifact directory contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
