"""Optional Hugging Face Transformers implementation of the model boundary."""

from __future__ import annotations

import math
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
from lsteg.model.interface import (
    IncrementalLogitsSession,
    Logits,
    RankedLogits,
    RuntimeFingerprint,
)
from lsteg.model.manifest import GPTQ_NUMERIC_POLICY, ModelManifest

GPTQMODEL_VERSION = "7.3.1"


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

        is_gptq = manifest.numeric_policy == GPTQ_NUMERIC_POLICY
        if is_gptq:
            _load_dependency(
                "gptqmodel",
                GPTQMODEL_VERSION,
                install_hint=(
                    "uv pip install --python .venv\\Scripts\\python.exe -r requirements-gptq.txt"
                ),
            )

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                manifest.tokenizer_id,
                revision=manifest.tokenizer_revision,
                trust_remote_code=manifest.trust_remote_code,
                local_files_only=local_files_only,
                **({"cache_dir": str(cache_dir)} if cache_dir is not None else {}),
            )
            if is_gptq:
                # Pre-quantized GPTQ checkpoints must be dispatched while they
                # load. Moving the model afterwards with ``model.to`` can break
                # quantized modules and can silently introduce CPU offload.
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    manifest.model_id,
                    use_safetensors=True,
                    device_map={"": manifest.device},
                    **common_options,
                )
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    manifest.model_id,
                    dtype=torch.float16,
                    use_safetensors=True,
                    **common_options,
                )
                model.to(manifest.device)
            model.eval()
        except Exception as error:
            raise ModelArtifactError(
                f"cannot load pinned model artifacts at {manifest.model_revision}"
            ) from error

        if is_gptq:
            _verify_gptq_model(model, manifest.device)

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

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        """Return tokenizer-declared special token IDs in canonical order."""
        raw_ids = getattr(self._tokenizer, "all_special_ids", ())
        validated = self._validate_token_ids(
            [int(token_id) for token_id in raw_ids],
            allow_empty=True,
        )
        return tuple(sorted(set(validated)))

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

    def transport_safe_candidate_ids(
        self,
        cover_prefix_token_ids: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> list[int]:
        """Batch-check Unicode transport invariance with the fast tokenizer.

        This is equivalent to repeatedly evaluating
        ``tokenize(detokenize(prefix + [candidate]))`` but uses one batch decode
        and one batch encode.  It removes hundreds of Python/Rust boundary
        crossings per generated token.
        """
        prefix = self._validate_token_ids(cover_prefix_token_ids, allow_empty=True)
        candidates = self._validate_token_ids(candidate_token_ids, allow_empty=True)
        if not candidates:
            return []

        trials = [[*prefix, token_id] for token_id in candidates]
        decoded = self._tokenizer.batch_decode(
            trials,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != len(trials):
            raise ModelArtifactError("tokenizer batch_decode returned an invalid batch size")
        encoded = self._tokenizer(
            list(decoded),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        raw_input_ids = encoded["input_ids"]
        if len(raw_input_ids) != len(trials):
            raise ModelArtifactError("tokenizer batch encode returned an invalid batch size")

        safe: list[int] = []
        for token_id, text, expected, actual in zip(
            candidates, decoded, trials, raw_input_ids, strict=True
        ):
            if not text:
                continue
            if [int(value) for value in actual] == expected:
                safe.append(token_id)
        return safe

    def render_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Render a pinned Qwen chat-template prompt as plain text.

        The returned string can be passed through :meth:`tokenize` again; the
        tokenizer's special tokens remain part of the exact sender/receiver
        context.  ``enable_thinking=False`` is the cover-generation default so
        hidden payload capacity is spent on the visible Japanese text rather
        than an internal thinking block.
        """
        if not isinstance(user_prompt, str):
            raise TypeError("user_prompt must be str")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be str or None")
        if not isinstance(enable_thinking, bool):
            raise TypeError("enable_thinking must be bool")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not isinstance(rendered, str):
            raise ModelArtifactError("chat template returned a non-string value")
        return rendered

    def render_chat_continuation(
        self,
        user_prompt: str,
        assistant_prefix: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Render a chat context that continues an existing assistant message.

        ``assistant_prefix`` is treated as an assistant prefill rather than being
        quoted inside a new user message.  Hugging Face chat templates implement
        this with ``continue_final_message=True`` so the model predicts the next
        token as a literal continuation of the already-visible cover text.
        """
        if not isinstance(user_prompt, str):
            raise TypeError("user_prompt must be str")
        if not isinstance(assistant_prefix, str):
            raise TypeError("assistant_prefix must be str")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be str or None")
        if not isinstance(enable_thinking, bool):
            raise TypeError("enable_thinking must be bool")

        if enable_thinking:
            raise ValueError("assistant continuation currently requires enable_thinking=False")

        # Qwen3's chat template rewrites final assistant messages: it extracts
        # anything before the last ``</think>`` as reasoning content and may
        # inject an empty thinking block even when thinking is disabled.  That
        # transformation is incompatible with Hugging Face's
        # ``continue_final_message=True`` validation because the rendered final
        # message is no longer byte-for-byte identical to ``assistant_prefix``.
        # Render only system/user + the non-thinking generation prompt, then
        # append the already-visible cover literally as an assistant prefill.
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise ModelArtifactError("chat template returned a non-string value")
        return rendered + assistant_prefix

    def next_logits(self, token_ids: Sequence[int]) -> Logits:
        validated = self._validate_token_ids(token_ids, allow_empty=False)
        with self._torch.inference_mode():
            input_ids = self._torch.tensor(
                [validated],
                dtype=self._torch.long,
                device=self._manifest.device,
            )
            output = self._model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=1,
            )
        return self._logits_from_output(output)

    def start_incremental_logits(self, token_ids: Sequence[int]) -> IncrementalLogitsSession:
        """Prefill ``token_ids`` once and reuse the model KV cache thereafter."""
        validated = self._validate_token_ids(token_ids, allow_empty=False)
        return _TransformersIncrementalLogitsSession(self, validated)

    def _logits_from_output(self, output: Any) -> Logits:
        return self._logits_from_tensor(output.logits[0, -1].detach())

    def _logits_from_tensor(self, tensor: Any) -> Logits:
        raw_values = tensor.to("cpu", dtype=self._torch.float32).tolist()
        logits = Logits.from_values(raw_values)
        if len(logits) != self._vocab_size:
            raise ModelArtifactError(
                f"logit width mismatch: expected {self._vocab_size}, got {len(logits)}"
            )
        return logits

    def _ranked_logits_from_tensor(
        self,
        tensor: Any,
        *,
        top_k: int,
        excluded_token_ids: Sequence[int],
        penalized_token_ids: Sequence[int] = (),
        presence_penalty: float = 0.0,
    ) -> RankedLogits:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be int")
        excluded = self._validate_token_ids(excluded_token_ids, allow_empty=True)
        penalized = self._validate_token_ids(penalized_token_ids, allow_empty=True)
        if isinstance(presence_penalty, bool) or not isinstance(presence_penalty, (int, float)):
            raise TypeError("presence_penalty must be a number")
        penalty = float(presence_penalty)
        if not math.isfinite(penalty) or not 0.0 <= penalty <= 2.0:
            raise ValueError("presence_penalty must satisfy 0 <= presence_penalty <= 2")
        available = self._vocab_size - len(set(excluded))
        if not 1 <= top_k <= available:
            raise ValueError(f"top_k must satisfy 1 <= top_k <= available vocabulary ({available})")

        # ``topk`` avoids a full-vocabulary sort.  Its tie ordering is not part
        # of the API, so recover the k-th cutoff and deterministically choose
        # the lowest token IDs among exact float16 ties at that boundary.
        working = tensor
        if excluded or (penalty > 0.0 and penalized):
            working = tensor.clone()
        if penalty > 0.0 and penalized:
            working[sorted(set(penalized))] -= penalty
        if excluded:
            working[excluded] = -self._torch.inf
        top_values, _ = self._torch.topk(
            working,
            top_k,
            largest=True,
            sorted=False,
        )
        cutoff = top_values.min()
        strict_ids = self._torch.nonzero(working > cutoff, as_tuple=False).flatten()
        tie_ids = self._torch.nonzero(working == cutoff, as_tuple=False).flatten()
        needed = top_k - int(strict_ids.numel())
        selected_ids_tensor = self._torch.cat((strict_ids, tie_ids[:needed]))
        selected_values_tensor = working[selected_ids_tensor]

        selected_ids = selected_ids_tensor.to("cpu").tolist()
        selected_values = selected_values_tensor.to("cpu", dtype=self._torch.float32).tolist()
        pairs = sorted(
            (
                (int(token_id), float(value))
                for token_id, value in zip(selected_ids, selected_values, strict=True)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return RankedLogits(
            tuple(token_id for token_id, _ in pairs),
            tuple(value for _, value in pairs),
        )

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


class _TransformersIncrementalLogitsSession:
    """One batch-1 autoregressive stream backed by a Hugging Face KV cache."""

    __slots__ = ("_backend", "_context_length", "_logits_tensor", "_past_key_values")

    def __init__(self, backend: TransformersBackend, token_ids: Sequence[int]) -> None:
        self._backend = backend
        self._context_length = len(token_ids)
        with backend._torch.inference_mode():
            input_ids = backend._torch.tensor(
                [list(token_ids)],
                dtype=backend._torch.long,
                device=backend._manifest.device,
            )
            output = backend._model(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        past_key_values = getattr(output, "past_key_values", None)
        if past_key_values is None:
            raise ModelArtifactError("model did not return past_key_values with use_cache=True")
        self._past_key_values = past_key_values
        self._logits_tensor = output.logits[0, -1].detach()

    def next_logits(self) -> Logits:
        return self._backend._logits_from_tensor(self._logits_tensor)

    def top_logits(
        self,
        top_k: int,
        *,
        excluded_token_ids: Sequence[int] = (),
        penalized_token_ids: Sequence[int] = (),
        presence_penalty: float = 0.0,
    ) -> RankedLogits:
        return self._backend._ranked_logits_from_tensor(
            self._logits_tensor,
            top_k=top_k,
            excluded_token_ids=excluded_token_ids,
            penalized_token_ids=penalized_token_ids,
            presence_penalty=presence_penalty,
        )

    def entropy(self, *, temperature: float = 1.0) -> float:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")
        value = float(temperature)
        if value <= 0.0:
            raise ValueError("temperature must be greater than zero")
        torch = self._backend._torch
        logits = self._logits_tensor.to(dtype=torch.float32) / value
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        entropy_nats = -(probabilities * log_probabilities).sum()
        return float((entropy_nats / math.log(2.0)).item())

    def append(self, token_id: int) -> None:
        validated = self._backend._validate_token_ids([token_id], allow_empty=False)
        if self._context_length >= self._backend._manifest.max_context_tokens:
            raise ModelInputError(
                f"context exceeds {self._backend._manifest.max_context_tokens} tokens"
            )
        with self._backend._torch.inference_mode():
            input_ids = self._backend._torch.tensor(
                [validated],
                dtype=self._backend._torch.long,
                device=self._backend._manifest.device,
            )
            output = self._backend._model(
                input_ids=input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        past_key_values = getattr(output, "past_key_values", None)
        if past_key_values is None:
            raise ModelArtifactError("model stopped returning past_key_values during decoding")
        self._past_key_values = past_key_values
        self._logits_tensor = output.logits[0, -1].detach()
        self._context_length += 1


def _load_dependency(
    name: str,
    expected_version: str,
    *,
    install_hint: str = "uv sync --extra model",
) -> ModuleType:
    try:
        installed_version = version(name)
    except PackageNotFoundError as error:
        raise ModelDependencyError(
            f"optional dependency {name} is missing; run {install_hint}"
        ) from error
    if installed_version != expected_version:
        raise ModelDependencyError(
            f"{name} version mismatch: expected {expected_version}, got {installed_version}"
        )
    return import_module(name)


def _verify_gptq_model(model: Any, required_device: str) -> None:
    """Reject non-GPTQ, non-4-bit, or CPU-offloaded artifacts early."""
    config = getattr(model, "config", None)
    raw_quantization = getattr(config, "quantization_config", None)
    to_dict = getattr(raw_quantization, "to_dict", None)
    if callable(to_dict):
        raw_quantization = to_dict()
    if not isinstance(raw_quantization, dict):
        raise ModelArtifactError("GPTQ manifest resolved to a non-quantized model")

    quant_method = raw_quantization.get("quant_method")
    quant_method = getattr(quant_method, "value", quant_method)
    if str(quant_method).lower() != "gptq" or raw_quantization.get("bits") != 4:
        raise ModelArtifactError("model is not the required 4-bit GPTQ artifact")

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        normalized = {str(device) for device in device_map.values()}
        accepted = {required_device, "0" if required_device == "cuda:0" else required_device}
        if not normalized.issubset(accepted):
            raise ModelDeviceError(
                f"GPTQ model is not fully resident on {required_device}: {sorted(normalized)}"
            )
        return

    observed_device = str(getattr(model, "device", ""))
    if observed_device != required_device:
        raise ModelDeviceError(
            f"GPTQ model is not resident on {required_device}: {observed_device or 'unknown'}"
        )


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
