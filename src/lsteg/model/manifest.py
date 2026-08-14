"""Strict, versioned manifests for reproducible model inference."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lsteg.model.errors import InvalidModelManifestError

MODEL_MANIFEST_SCHEMA_VERSION = 1
NUMERIC_POLICY = "cuda-float16-same-runtime-device-v1"

_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:\+[A-Za-z0-9.]+)?")
_FIELDS = {
    "schema_version",
    "role",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "license",
    "transformers_version",
    "torch_version",
    "dtype",
    "device",
    "numeric_policy",
    "max_context_tokens",
    "trust_remote_code",
}


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """All artifact and runtime inputs that can affect next-token logits."""

    schema_version: int
    role: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    license: str
    transformers_version: str
    torch_version: str
    dtype: str
    device: str
    numeric_policy: str
    max_context_tokens: int
    trust_remote_code: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != MODEL_MANIFEST_SCHEMA_VERSION
        ):
            raise InvalidModelManifestError(
                f"unsupported model manifest schema: {self.schema_version}"
            )
        if not isinstance(self.role, str) or self.role not in {"debug", "quality"}:
            raise InvalidModelManifestError("model role must be debug or quality")
        _validate_artifact("model_id", self.model_id)
        _validate_artifact("tokenizer_id", self.tokenizer_id)
        _validate_revision("model_revision", self.model_revision)
        _validate_revision("tokenizer_revision", self.tokenizer_revision)
        if not isinstance(self.license, str) or not self.license:
            raise InvalidModelManifestError("model license must not be empty")
        _validate_version("transformers_version", self.transformers_version)
        _validate_version("torch_version", self.torch_version)
        if self.dtype != "float16":
            raise InvalidModelManifestError("model dtype must be float16")
        if self.device != "cuda:0":
            raise InvalidModelManifestError("model device must be cuda:0")
        if self.numeric_policy != NUMERIC_POLICY:
            raise InvalidModelManifestError(f"unsupported numeric policy: {self.numeric_policy}")
        if isinstance(self.max_context_tokens, bool) or not isinstance(
            self.max_context_tokens, int
        ):
            raise InvalidModelManifestError("max_context_tokens must be int")
        if not 1 <= self.max_context_tokens <= 32_768:
            raise InvalidModelManifestError("max_context_tokens must be between 1 and 32768")
        if self.trust_remote_code is not False:
            raise InvalidModelManifestError("trust_remote_code must remain false")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ModelManifest:
        """Parse a mapping while rejecting missing and unknown fields."""
        fields = set(raw)
        if fields != _FIELDS:
            missing = sorted(_FIELDS - fields)
            unknown = sorted(fields - _FIELDS)
            raise InvalidModelManifestError(
                f"model manifest fields differ; missing={missing}, unknown={unknown}"
            )
        try:
            return cls(
                schema_version=raw["schema_version"],  # type: ignore[arg-type]
                role=raw["role"],  # type: ignore[arg-type]
                model_id=raw["model_id"],  # type: ignore[arg-type]
                model_revision=raw["model_revision"],  # type: ignore[arg-type]
                tokenizer_id=raw["tokenizer_id"],  # type: ignore[arg-type]
                tokenizer_revision=raw["tokenizer_revision"],  # type: ignore[arg-type]
                license=raw["license"],  # type: ignore[arg-type]
                transformers_version=raw["transformers_version"],  # type: ignore[arg-type]
                torch_version=raw["torch_version"],  # type: ignore[arg-type]
                dtype=raw["dtype"],  # type: ignore[arg-type]
                device=raw["device"],  # type: ignore[arg-type]
                numeric_policy=raw["numeric_policy"],  # type: ignore[arg-type]
                max_context_tokens=raw["max_context_tokens"],  # type: ignore[arg-type]
                trust_remote_code=raw["trust_remote_code"],  # type: ignore[arg-type]
            )
        except TypeError as error:
            raise InvalidModelManifestError("model manifest field has an invalid type") from error

    @classmethod
    def from_path(cls, path: Path) -> ModelManifest:
        """Load a UTF-8 JSON manifest from disk."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InvalidModelManifestError(f"cannot read model manifest: {path}") from error
        if not isinstance(raw, dict):
            raise InvalidModelManifestError("model manifest root must be an object")
        return cls.from_mapping(raw)

    def as_dict(self) -> dict[str, bool | int | str]:
        """Return a deterministic JSON-compatible representation."""
        return {field: getattr(self, field) for field in sorted(_FIELDS)}


def _validate_artifact(field: str, value: object) -> None:
    if not isinstance(value, str) or _ARTIFACT_ID.fullmatch(value) is None:
        raise InvalidModelManifestError(f"{field} must be an owner/artifact ID")


def _validate_revision(field: str, value: object) -> None:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise InvalidModelManifestError(f"{field} must be a full lowercase commit SHA")


def _validate_version(field: str, value: object) -> None:
    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise InvalidModelManifestError(f"{field} must be an exact version")
