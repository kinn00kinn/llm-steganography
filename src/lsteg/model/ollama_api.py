"""Minimal persistent Ollama client for deterministic local logprob queries.

This module deliberately uses only the Python standard library.  The direct
steganography channel makes one localhost request per cover token, so keeping a
single HTTP/1.1 connection alive avoids reconnect overhead while Ollama keeps
the model and its prompt/KV cache resident.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from lsteg.model.errors import ModelArtifactError, ModelInputError

OLLAMA_TOP_LOGPROBS_MAX = 20


@dataclass(frozen=True, slots=True)
class OllamaModelIdentity:
    model: str
    digest: str
    ollama_version: str
    parameter_size: str
    quantization_level: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path) -> OllamaModelIdentity:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Ollama identity file must contain a JSON object")
        return cls(
            model=_required_string(raw, "model"),
            digest=_required_string(raw, "digest"),
            ollama_version=_required_string(raw, "ollama_version"),
            parameter_size=_required_string(raw, "parameter_size"),
            quantization_level=_required_string(raw, "quantization_level"),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def assert_matches(self, actual: OllamaModelIdentity) -> None:
        if self != actual:
            raise ModelArtifactError(
                "Ollama runtime/model identity mismatch; expected "
                f"{self.as_dict()}, got {actual.as_dict()}"
            )


@dataclass(frozen=True, slots=True)
class OllamaTokenCandidate:
    raw_bytes: bytes
    token: str
    logprob: float

    def __post_init__(self) -> None:
        if not self.raw_bytes:
            raise ValueError("candidate bytes must not be empty")
        if not isinstance(self.token, str):
            raise TypeError("candidate token must be str")
        if not math.isfinite(self.logprob):
            raise ValueError("candidate logprob must be finite")


@dataclass(frozen=True, slots=True)
class OllamaRequestMetrics:
    wall_seconds: float
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    eval_count: int
    eval_duration_ns: int
    request_count: int = 1


@dataclass(frozen=True, slots=True)
class OllamaLogprobResponse:
    candidates: tuple[OllamaTokenCandidate, ...]
    metrics: OllamaRequestMetrics


class OllamaAPIClient:
    """Persistent localhost Ollama client specialized for one-token probes."""

    __slots__ = (
        "_base_path",
        "_connection",
        "_host",
        "_model",
        "_port",
        "_timeout",
    )

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        parsed = urlsplit(base_url)
        if parsed.scheme != "http":
            raise ValueError("only http:// Ollama endpoints are supported")
        if not parsed.hostname:
            raise ValueError("Ollama base_url must include a hostname")
        self._model = model
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._base_path = parsed.path.rstrip("/")
        self._timeout = float(timeout)
        self._connection: http.client.HTTPConnection | None = None

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> OllamaAPIClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def identity(self) -> OllamaModelIdentity:
        version_payload = self._request_json("GET", "/api/version")
        version = _required_string(version_payload, "version")
        tags = self._request_json("GET", "/api/tags")
        models = tags.get("models")
        if not isinstance(models, list):
            raise ModelArtifactError("Ollama /api/tags returned no models list")
        selected: dict[str, Any] | None = None
        for raw in models:
            if not isinstance(raw, dict):
                continue
            if raw.get("name") == self._model or raw.get("model") == self._model:
                selected = raw
                break
        if selected is None:
            raise ModelArtifactError(f"Ollama model {self._model!r} is not installed")
        details = selected.get("details")
        if not isinstance(details, dict):
            details = {}
        digest = _required_string(selected, "digest")
        parameter_size = str(details.get("parameter_size") or "unknown")
        quantization = str(details.get("quantization_level") or "unknown")
        return OllamaModelIdentity(
            model=self._model,
            digest=digest,
            ollama_version=version,
            parameter_size=parameter_size,
            quantization_level=quantization,
        )

    def next_token_candidates(
        self,
        raw_prompt: str,
        *,
        top_logprobs: int = OLLAMA_TOP_LOGPROBS_MAX,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
    ) -> OllamaLogprobResponse:
        """Return one-step top logprobs with neutral sampling parameters.

        ``raw_prompt`` is sent with ``raw=true``.  The caller should make each
        successive prompt an exact textual extension of the previous prompt so
        Ollama's longest-common-prefix prompt cache can reuse almost all KV
        state.  Sampling itself is intentionally neutral; channel temperature
        and nucleus truncation are applied locally and deterministically.
        """
        if not isinstance(raw_prompt, str) or not raw_prompt:
            raise TypeError("raw_prompt must be a non-empty str")
        if not 1 <= top_logprobs <= OLLAMA_TOP_LOGPROBS_MAX:
            raise ValueError(f"top_logprobs must be in [1, {OLLAMA_TOP_LOGPROBS_MAX}]")
        if num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        body = {
            "model": self._model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "keep_alive": keep_alive,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "options": {
                "num_predict": 1,
                "num_ctx": num_ctx,
                # Neutralize Ollama-side sampling transforms.  The channel
                # applies its own temperature/top-p to the returned top scores.
                "temperature": 1.0,
                "top_k": OLLAMA_TOP_LOGPROBS_MAX,
                "top_p": 1.0,
                "min_p": 0.0,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "repeat_last_n": 0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "seed": 0,
            },
        }
        started = time.perf_counter()
        payload = self._request_json("POST", "/api/generate", body)
        wall_seconds = time.perf_counter() - started
        raw_logprobs = payload.get("logprobs")
        if not isinstance(raw_logprobs, list) or not raw_logprobs:
            raise ModelArtifactError(
                "Ollama returned no logprobs; update Ollama to a version whose "
                "native /api/generate endpoint supports logprobs/top_logprobs"
            )
        first = raw_logprobs[0]
        if not isinstance(first, dict):
            raise ModelArtifactError("Ollama returned malformed logprobs")
        top = first.get("top_logprobs")
        if not isinstance(top, list) or not top:
            raise ModelArtifactError("Ollama returned no top_logprobs alternatives")

        candidates: list[OllamaTokenCandidate] = []
        for entry in top:
            if not isinstance(entry, dict):
                continue
            token = entry.get("token")
            logprob = entry.get("logprob")
            if (
                not isinstance(token, str)
                or isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
            ):
                continue
            raw_bytes = _candidate_bytes(entry, token)
            if not raw_bytes:
                continue
            candidates.append(OllamaTokenCandidate(raw_bytes, token, float(logprob)))
        if not candidates:
            raise ModelArtifactError("Ollama top_logprobs contained no usable candidates")

        metrics = OllamaRequestMetrics(
            wall_seconds=wall_seconds,
            total_duration_ns=_int_metric(payload, "total_duration"),
            load_duration_ns=_int_metric(payload, "load_duration"),
            prompt_eval_count=_int_metric(payload, "prompt_eval_count"),
            prompt_eval_duration_ns=_int_metric(payload, "prompt_eval_duration"),
            eval_count=_int_metric(payload, "eval_count"),
            eval_duration_ns=_int_metric(payload, "eval_duration"),
        )
        return OllamaLogprobResponse(tuple(candidates), metrics)

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None
        headers = {"Accept": "application/json", "Connection": "keep-alive"}
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        request_path = f"{self._base_path}{path}" or "/"
        for attempt in range(2):
            connection = self._get_connection()
            try:
                connection.request(method, request_path, body=payload, headers=headers)
                response = connection.getresponse()
                raw = response.read()
            except (OSError, http.client.HTTPException) as error:
                self.close()
                if attempt == 0:
                    continue
                raise ModelInputError(
                    f"cannot reach Ollama at {self._host}:{self._port}: {error}"
                ) from error
            if response.status >= 400:
                message = raw.decode("utf-8", errors="replace")
                raise ModelInputError(
                    f"Ollama {method} {path} failed with HTTP {response.status}: {message}"
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ModelArtifactError("Ollama returned invalid JSON") from error
            if not isinstance(parsed, dict):
                raise ModelArtifactError("Ollama response must be a JSON object")
            if "error" in parsed:
                raise ModelInputError(f"Ollama error: {parsed['error']}")
            return parsed
        raise AssertionError("unreachable")

    def _get_connection(self) -> http.client.HTTPConnection:
        if self._connection is None:
            self._connection = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=self._timeout,
            )
        return self._connection


def candidate_fingerprint(
    candidates: tuple[OllamaTokenCandidate, ...], *, quantum: float = 1e-4
) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        quantized = quantize_logprob(candidate.logprob, quantum=quantum)
        digest.update(len(candidate.raw_bytes).to_bytes(4, "big"))
        digest.update(candidate.raw_bytes)
        digest.update(quantized.to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def quantize_logprob(value: float, *, quantum: float) -> int:
    if not math.isfinite(value):
        raise ValueError("logprob must be finite")
    if not math.isfinite(quantum) or quantum <= 0.0:
        raise ValueError("quantum must be finite and positive")
    scaled = value / quantum
    if scaled >= 0:
        return math.floor(scaled + 0.5)
    return math.ceil(scaled - 0.5)


def _candidate_bytes(entry: dict[str, Any], token: str) -> bytes:
    raw = entry.get("bytes")
    if isinstance(raw, list) and raw:
        values: list[int] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                return b""
            values.append(value)
        return bytes(values)
    if "\ufffd" in token:
        return b""
    return token.encode("utf-8")


def _int_metric(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if value > 0 else 0


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing non-empty string field: {name}")
    return value
