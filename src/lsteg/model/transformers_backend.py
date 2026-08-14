"""Optional Hugging Face Transformers implementation of the model boundary."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any

from lsteg.model.errors import (
    ModelArtifactError,
    ModelDependencyError,
    ModelDeviceError,
    ModelInputError,
)
from lsteg.model.interface import Logits, RuntimeFingerprint
from lsteg.model.manifest import ModelManifest


class TransformersBackend:
    """Pinned causal-LM inference without leaking framework tensors."""

    __slots__ = ("_manifest", "_model", "_runtime", "_tokenizer", "_torch", "_vocab_size")

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        torch_module: Any,
        tokenizer: Any,
        model: Any,
        runtime: RuntimeFingerprint,
        vocabulary_size: int,
    ) -> None:
        self._manifest = manifest
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._model = model
        self._runtime = runtime
        self._vocab_size = vocabulary_size

    @classmethod
    def load(
        cls,
        manifest: ModelManifest,
        *,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ) -> TransformersBackend:
        """Load exactly the runtime and artifacts declared by a manifest."""
        torch = _load_dependency("torch", manifest.torch_version)
        transformers = _load_dependency("transformers", manifest.transformers_version)
        _configure_determinism(torch)
        runtime = _runtime_fingerprint(torch, manifest)

        common_options: dict[str, object] = {
            "revision": manifest.model_revision,
            "trust_remote_code": manifest.trust_remote_code,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            common_options["cache_dir"] = str(cache_dir)

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                manifest.tokenizer_id,
                revision=manifest.tokenizer_revision,
                trust_remote_code=manifest.trust_remote_code,
                local_files_only=local_files_only,
                **({"cache_dir": str(cache_dir)} if cache_dir is not None else {}),
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                manifest.model_id,
                dtype=torch.float16,
                use_safetensors=True,
                **common_options,
            )
            model.eval()
            model.to(manifest.device)
        except Exception as error:
            raise ModelArtifactError(
                f"cannot load pinned model artifacts at {manifest.model_revision}"
            ) from error

        _verify_resolved_revision(model, manifest.model_revision, "model")
        _verify_resolved_revision(tokenizer, manifest.tokenizer_revision, "tokenizer")
        vocabulary_size = int(model.config.vocab_size)
        if vocabulary_size <= 0:
            raise ModelArtifactError("model reports an invalid vocabulary size")
        return cls(
            manifest,
            torch_module=torch,
            tokenizer=tokenizer,
            model=model,
            runtime=runtime,
            vocabulary_size=vocabulary_size,
        )

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    @property
    def runtime(self) -> RuntimeFingerprint:
        return self._runtime

    @property
    def vocabulary_size(self) -> int:
        return self._vocab_size

    def tokenize(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        raw_ids = self._tokenizer.encode(text, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in raw_ids]
        self._validate_token_ids(token_ids, allow_empty=True)
        return token_ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        validated = self._validate_token_ids(token_ids, allow_empty=True)
        decoded = self._tokenizer.decode(
            validated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise ModelArtifactError("tokenizer returned a non-string value")
        return decoded

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        validated = self._validate_token_ids(token_ids, allow_empty=False)
        with self._torch.inference_mode():
            input_ids = self._torch.tensor(
                [validated],
                dtype=self._torch.long,
                device=self._manifest.device,
            )
            output = self._model(input_ids=input_ids, use_cache=False)
            raw_values = output.logits[0, -1].detach().to("cpu", dtype=self._torch.float32).tolist()
        logits = Logits.from_values(raw_values)
        if len(logits) != self._vocab_size:
            raise ModelArtifactError(
                f"logit width mismatch: expected {self._vocab_size}, got {len(logits)}"
            )
        return logits

    def _validate_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        allow_empty: bool,
    ) -> list[int]:
        if len(token_ids) == 0 and not allow_empty:
            raise ModelInputError("next_logits requires at least one context token")
        if len(token_ids) > self._manifest.max_context_tokens:
            raise ModelInputError(f"context exceeds {self._manifest.max_context_tokens} tokens")
        validated: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if not 0 <= token_id < self._vocab_size:
                raise ModelInputError(f"token ID is outside the vocabulary: {token_id}")
            validated.append(token_id)
        return validated


def _load_dependency(name: str, expected_version: str) -> ModuleType:
    try:
        installed_version = version(name)
    except PackageNotFoundError as error:
        raise ModelDependencyError(
            f"optional dependency {name} is missing; run uv sync --extra model"
        ) from error
    if installed_version != expected_version:
        raise ModelDependencyError(
            f"{name} version mismatch: expected {expected_version}, got {installed_version}"
        )
    return import_module(name)


def _configure_determinism(torch: Any) -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _runtime_fingerprint(torch: Any, manifest: ModelManifest) -> RuntimeFingerprint:
    if not torch.cuda.is_available():
        raise ModelDeviceError(f"required device is unavailable: {manifest.device}")
    device_name = str(torch.cuda.get_device_name(manifest.device))
    major, minor = torch.cuda.get_device_capability(manifest.device)
    cuda_version = str(torch.version.cuda)
    if cuda_version != "13.0":
        raise ModelDeviceError(f"CUDA runtime mismatch: expected 13.0, got {cuda_version}")
    return RuntimeFingerprint(
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.release()}-{platform.machine()}",
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        cuda_version=cuda_version,
        device_name=device_name,
        compute_capability=f"{major}.{minor}",
    )


def _verify_resolved_revision(
    artifact: Any,
    expected_revision: str,
    artifact_name: str,
) -> None:
    config = getattr(artifact, "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    init_kwargs = getattr(artifact, "init_kwargs", None)
    if resolved_revision is None and isinstance(init_kwargs, dict):
        resolved_revision = init_kwargs.get("_commit_hash")
    if resolved_revision is not None and resolved_revision != expected_revision:
        raise ModelArtifactError(
            f"{artifact_name} revision mismatch: expected {expected_revision}, "
            f"got {resolved_revision}"
        )
