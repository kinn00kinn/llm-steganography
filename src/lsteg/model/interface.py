"""Model-neutral tokenization and next-logit types."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from struct import Struct
from typing import Protocol, overload, runtime_checkable

from lsteg.model.manifest import ModelManifest

_FLOAT32 = Struct(">f")


@dataclass(frozen=True, slots=True)
class Logits(Sequence[float]):
    """An immutable, model-neutral float32 next-token logit vector."""

    _values: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not self._values:
            raise ValueError("logits must contain at least one value")
        canonical: list[float] = []
        for value in self._values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("logit values must be numbers")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("logit values must be finite")
            try:
                numeric = _FLOAT32.unpack(_FLOAT32.pack(numeric))[0]
            except OverflowError as error:
                raise ValueError("logit value is outside float32 range") from error
            canonical.append(numeric)
        object.__setattr__(self, "_values", tuple(canonical))

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Logits:
        return cls(tuple(values))

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return self._values[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    @property
    def argmax_token_id(self) -> int:
        """Return the greatest logit, resolving exact ties by lowest token ID."""
        return max(range(len(self)), key=lambda token_id: (self[token_id], -token_id))

    @property
    def sha256(self) -> str:
        """Hash canonical big-endian float32 values for regression evidence."""
        digest = sha256()
        for value in self:
            digest.update(_FLOAT32.pack(value))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    """Observed inference environment needed to interpret reproducibility claims."""

    python_version: str
    platform: str
    torch_version: str
    transformers_version: str
    cuda_version: str
    device_name: str
    compute_capability: str

    def as_dict(self) -> dict[str, str]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
        }


@runtime_checkable
class LanguageModelBackend(Protocol):
    """The only model behavior exposed to steganography layers."""

    @property
    def manifest(self) -> ModelManifest: ...

    @property
    def runtime(self) -> RuntimeFingerprint: ...

    @property
    def vocabulary_size(self) -> int: ...

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, token_ids: Sequence[int]) -> str: ...

    def next_logits(self, token_ids: Sequence[int]) -> Logits: ...
