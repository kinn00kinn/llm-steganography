from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from lsteg.payload import (
    MASTER_KEY_FILE_SIZE,
    MASTER_KEY_FILE_VERSION,
    MASTER_KEY_SIZE,
    InvalidMasterKeyError,
    KeyFileExistsError,
    KeyFileWriteError,
    create_master_key_file,
    derive_keys,
    generate_master_key,
    parse_master_key_file,
    read_master_key,
    serialize_master_key,
    validate_master_key,
)


def test_master_keys_are_256_bit_and_unique() -> None:
    keys = {generate_master_key() for _ in range(100)}

    assert len(keys) == 100
    assert all(len(key) == MASTER_KEY_SIZE for key in keys)


@pytest.mark.parametrize("size", [0, MASTER_KEY_SIZE - 1, MASTER_KEY_SIZE + 1])
def test_master_key_requires_exact_size(size: int) -> None:
    with pytest.raises(InvalidMasterKeyError, match=str(size)):
        validate_master_key(b"x" * size)


def test_master_key_requires_bytes() -> None:
    with pytest.raises(TypeError, match="bytes"):
        validate_master_key(bytearray(MASTER_KEY_SIZE))  # type: ignore[arg-type]


def test_key_derivation_is_stable_and_purpose_separated() -> None:
    derived = derive_keys(bytes(range(MASTER_KEY_SIZE)))

    assert derived.encryption.hex() == (
        "d95b16e06595312e151a0e70a2485d99b0dff85820f4d9cfcaf6a64cac172b5f"
    )
    assert derived.steganography.hex() == (
        "69494c90e0a09a580056a1fd73a892eab3b104b413df071ba12cbddcda0a9143"
    )
    assert derived.encryption != derived.steganography
    assert repr(derived) == "DerivedKeys()"


def test_key_file_round_trip() -> None:
    master_key = bytes(range(MASTER_KEY_SIZE))
    serialized = serialize_master_key(master_key)

    assert len(serialized) == MASTER_KEY_FILE_SIZE
    assert serialized[:4] == b"LSTK"
    assert serialized[4] == MASTER_KEY_FILE_VERSION
    assert serialized[5:8] == b"\x00\x00\x00"
    assert parse_master_key_file(serialized) == master_key


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data[:-1], "exactly"),
        (lambda data: data + b"x", "exactly"),
        (lambda data: b"BAD!" + data[4:], "magic"),
        (lambda data: data[:4] + b"\x02" + data[5:], "version"),
        (lambda data: data[:5] + b"\x01\x00\x00" + data[8:], "reserved"),
    ],
)
def test_invalid_key_file_is_rejected(
    mutate: Callable[[bytes], bytes],
    message: str,
) -> None:
    serialized = serialize_master_key(bytes(range(MASTER_KEY_SIZE)))

    with pytest.raises(InvalidMasterKeyError, match=message):
        parse_master_key_file(mutate(serialized))


def test_key_file_is_created_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "shared.key"

    assert create_master_key_file(target) == target
    first_contents = target.read_bytes()
    assert read_master_key(target) == parse_master_key_file(first_contents)

    with pytest.raises(KeyFileExistsError, match="refusing"):
        create_master_key_file(target)
    assert target.read_bytes() == first_contents
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="Windows chmod exposes only the read-only flag")
def test_key_file_permissions_are_owner_only(tmp_path: Path) -> None:
    target = create_master_key_file(tmp_path / "shared.key")

    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR


def test_key_file_requires_existing_parent(tmp_path: Path) -> None:
    with pytest.raises(KeyFileWriteError, match="directory"):
        create_master_key_file(tmp_path / "missing" / "shared.key")


def test_failed_atomic_install_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_link(source: str, destination: Path) -> None:
        del source, destination
        raise OSError("simulated unsupported hard link")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(KeyFileWriteError, match="safely"):
        create_master_key_file(tmp_path / "shared.key")
    assert list(tmp_path.iterdir()) == []
