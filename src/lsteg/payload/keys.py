"""Master-key generation, storage, validation, and purpose separation."""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from struct import Struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from nacl.utils import random as random_bytes

from lsteg.payload.errors import (
    InvalidMasterKeyError,
    KeyFileExistsError,
    KeyFileWriteError,
)

MASTER_KEY_SIZE = 32
MASTER_KEY_FILE_MAGIC = b"LSTK"
MASTER_KEY_FILE_VERSION = 1

_KEY_FILE_HEADER = Struct(">4sB3x")
MASTER_KEY_FILE_SIZE = _KEY_FILE_HEADER.size + MASTER_KEY_SIZE
_KEY_FILE_RESERVED = b"\x00\x00\x00"
_ENCRYPTION_CONTEXT = b"llm-steganography/v1/encryption"
_STEGANOGRAPHY_CONTEXT = b"llm-steganography/v1/steganography"


@dataclass(frozen=True, slots=True)
class DerivedKeys:
    """Purpose-specific subkeys whose repr never exposes their bytes."""

    encryption: bytes = field(repr=False)
    steganography: bytes = field(repr=False)


def generate_master_key() -> bytes:
    """Generate a uniformly random 256-bit master key with libsodium's CSPRNG."""
    return random_bytes(MASTER_KEY_SIZE)


def derive_keys(master_key: bytes) -> DerivedKeys:
    """Derive versioned encryption and steganography keys from a master key."""
    validated = validate_master_key(master_key)
    return DerivedKeys(
        encryption=_expand_key(validated, _ENCRYPTION_CONTEXT),
        steganography=_expand_key(validated, _STEGANOGRAPHY_CONTEXT),
    )


def validate_master_key(master_key: bytes) -> bytes:
    """Return a valid master key or raise without including key bytes in the error."""
    if not isinstance(master_key, bytes):
        msg = "master key must be bytes"
        raise TypeError(msg)
    if len(master_key) != MASTER_KEY_SIZE:
        raise InvalidMasterKeyError(
            f"master key must be exactly {MASTER_KEY_SIZE} bytes: got {len(master_key)}"
        )
    return master_key


def serialize_master_key(master_key: bytes) -> bytes:
    """Serialize a master key using the version-1 binary key-file format."""
    validated = validate_master_key(master_key)
    return _KEY_FILE_HEADER.pack(MASTER_KEY_FILE_MAGIC, MASTER_KEY_FILE_VERSION) + validated


def parse_master_key_file(data: bytes) -> bytes:
    """Parse a complete version-1 key file and return its raw master key."""
    if not isinstance(data, bytes):
        msg = "key file data must be bytes"
        raise TypeError(msg)
    if len(data) != MASTER_KEY_FILE_SIZE:
        raise InvalidMasterKeyError(
            f"key file must be exactly {MASTER_KEY_FILE_SIZE} bytes: got {len(data)}"
        )
    if data[:4] != MASTER_KEY_FILE_MAGIC:
        raise InvalidMasterKeyError("key file has an invalid magic value")
    if data[4] != MASTER_KEY_FILE_VERSION:
        raise InvalidMasterKeyError(f"unsupported key file version: {data[4]}")
    if data[5:8] != _KEY_FILE_RESERVED:
        raise InvalidMasterKeyError("key file reserved bytes must be zero")
    return validate_master_key(data[_KEY_FILE_HEADER.size :])


def read_master_key(path: str | os.PathLike[str]) -> bytes:
    """Read and validate a bounded master-key file."""
    key_path = Path(path)
    try:
        with key_path.open("rb") as key_file:
            data = key_file.read(MASTER_KEY_FILE_SIZE + 1)
    except OSError as error:
        raise InvalidMasterKeyError(f"cannot read key file: {key_path}") from error
    return parse_master_key_file(data)


def create_master_key_file(path: str | os.PathLike[str]) -> Path:
    """Atomically create a mode-0600 key file without replacing an existing path."""
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise KeyFileWriteError(f"key file directory does not exist: {parent}")

    serialized = serialize_master_key(generate_master_key())
    descriptor = -1
    temporary_path: str | None = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "wb") as key_file:
            descriptor = -1
            key_file.write(serialized)
            key_file.flush()
            os.fsync(key_file.fileno())

        try:
            os.link(temporary_path, target)
        except FileExistsError as error:
            message = f"refusing to overwrite existing key file: {target}"
            raise KeyFileExistsError(message) from error
        except OSError as error:
            raise KeyFileWriteError(f"cannot install key file safely: {target}") from error

        return target
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def _expand_key(master_key: bytes, context: bytes) -> bytes:
    return HKDFExpand(
        algorithm=hashes.SHA256(),
        length=MASTER_KEY_SIZE,
        info=context,
    ).derive(master_key)
