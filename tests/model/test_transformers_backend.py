from __future__ import annotations

import json
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsteg.model import (
    LanguageModelBackend,
    ModelArtifactError,
    ModelInputError,
    ModelManifest,
    RuntimeFingerprint,
    TransformersBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
BASELINE_PATH = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug-baseline.json"


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 3 for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(str(token_id) for token_id in token_ids)


class _FakeVector:
    def detach(self) -> _FakeVector:
        return self

    def to(self, device: str, *, dtype: object) -> _FakeVector:
        assert device == "cpu"
        del dtype
        return self

    def tolist(self) -> list[float]:
        return [0.25, 1.5, -1.0]


class _FakeLogitTensor:
    def __getitem__(self, key: tuple[int, int]) -> _FakeVector:
        assert key == (0, -1)
        return _FakeVector()


class _FakeModel:
    def __call__(self, *, input_ids: object, use_cache: bool) -> object:
        assert input_ids == [[0, 1]]
        assert use_cache is False
        return SimpleNamespace(logits=_FakeLogitTensor())


class _FakeTorch:
    long = object()
    float32 = object()

    def inference_mode(self) -> nullcontext[None]:
        return nullcontext()

    def tensor(self, values: list[list[int]], *, dtype: object, device: str) -> object:
        del dtype
        assert device == "cuda:0"
        return values


def _backend(*, vocabulary_size: int = 3) -> TransformersBackend:
    manifest = ModelManifest.from_path(MANIFEST_PATH)
    runtime = RuntimeFingerprint(
        "3.12.10",
        "Windows-11-AMD64",
        "2.13.0+cu130",
        "5.15.0",
        "13.0",
        "fake GPU",
        "8.9",
    )
    return TransformersBackend(
        manifest,
        torch_module=_FakeTorch(),
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        runtime=runtime,
        vocabulary_size=vocabulary_size,
    )


def test_backend_exposes_only_model_neutral_values() -> None:
    backend = _backend()

    assert isinstance(backend, LanguageModelBackend)
    assert backend.tokenize("ab") == [1, 2]
    assert backend.detokenize([0, 1, 2]) == "012"
    logits = backend.next_logits([0, 1])
    assert tuple(logits) == (0.25, 1.5, -1.0)
    assert logits.argmax_token_id == 1


@pytest.mark.parametrize("token_ids", [[], [3], [-1]])
def test_backend_rejects_invalid_next_logit_context(token_ids: list[int]) -> None:
    with pytest.raises(ModelInputError):
        _backend().next_logits(token_ids)


def test_backend_rejects_wrong_logit_width() -> None:
    with pytest.raises(ModelArtifactError, match="width mismatch"):
        _backend(vocabulary_size=4).next_logits([0, 1])


def test_token_ids_must_be_plain_integers() -> None:
    with pytest.raises(TypeError, match="integers"):
        _backend().detokenize([True])


def test_model_baseline_is_bounded_and_matches_manifest() -> None:
    manifest = ModelManifest.from_path(MANIFEST_PATH)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["model_revision"] == manifest.model_revision
    assert baseline["tokenization_round_trip"] is True
    assert baseline["token_count"] == 13
    assert baseline["vocabulary_size"] == 151_936
    assert len(baseline["input_sha256"]) == 64
    assert len(baseline["token_ids_sha256"]) == 64
    assert len(baseline["logits_sha256"]) == 64
    assert "text" not in baseline
    assert "logits" not in baseline


@pytest.mark.model
@pytest.mark.skipif(
    __import__("os").environ.get("LSTEG_RUN_MODEL_TESTS") != "1",
    reason="set LSTEG_RUN_MODEL_TESTS=1 after uv sync --extra model",
)
def test_pinned_qwen_backend_repeats_logits() -> None:
    backend = TransformersBackend.load(
        ModelManifest.from_path(MANIFEST_PATH),
        cache_dir=PROJECT_ROOT / "artifacts" / "model-cache",
        local_files_only=True,
    )
    text = "今日は研究室で再現可能な推論を確認する。"
    token_ids = backend.tokenize(text)
    token_bytes = b"".join(token_id.to_bytes(4, "big") for token_id in token_ids)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    first = backend.next_logits(token_ids)
    second = backend.next_logits(token_ids)

    assert backend.detokenize(token_ids) == text
    assert sha256(text.encode("utf-8")).hexdigest() == baseline["input_sha256"]
    assert sha256(token_bytes).hexdigest() == baseline["token_ids_sha256"]
    assert len(token_ids) == baseline["token_count"]
    assert len(first) == backend.vocabulary_size == baseline["vocabulary_size"]
    assert first.sha256 == second.sha256
    assert first.sha256 == baseline["logits_sha256"]
    assert first.argmax_token_id == second.argmax_token_id == baseline["argmax_token_id"]
    assert backend.runtime.as_dict() == baseline["runtime"]
