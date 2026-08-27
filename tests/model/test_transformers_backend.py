from __future__ import annotations

import json
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest

from lsteg.model import (
    LanguageModelBackend,
    ModelArtifactError,
    ModelDeviceError,
    ModelInputError,
    ModelManifest,
    RankedIncrementalLogitsSession,
    RuntimeFingerprint,
    TransformersBackend,
)
from lsteg.model.transformers_backend import _verify_gptq_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"
BASELINE_PATH = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug-baseline.json"


class _FakeTokenizer:
    all_special_ids: ClassVar[list[int]] = [2, 1, 2]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool | None = None,
        continue_final_message: bool | None = None,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert enable_thinking is False
        joined = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        assert continue_final_message in (None, False)
        assert add_generation_prompt is True
        return f"CHAT:{joined}|assistant:"

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
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool, object | None]] = []

    def __call__(
        self,
        *,
        input_ids: object,
        use_cache: bool,
        logits_to_keep: int,
        past_key_values: object | None = None,
    ) -> object:
        assert logits_to_keep == 1
        self.calls.append((input_ids, use_cache, past_key_values))
        if past_key_values is None:
            assert input_ids == [[0, 1]]
        else:
            assert input_ids == [[2]]
            assert past_key_values == "fake-cache-1"
        return SimpleNamespace(
            logits=_FakeLogitTensor(),
            past_key_values=("fake-cache-1" if past_key_values is None else "fake-cache-2"),
        )


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


def test_backend_reports_canonical_special_token_ids() -> None:
    assert _backend().special_token_ids == (1, 2)


def test_backend_exposes_only_model_neutral_values() -> None:
    backend = _backend()

    assert isinstance(backend, LanguageModelBackend)
    assert backend.tokenize("ab") == [1, 2]
    assert backend.detokenize([0, 1, 2]) == "012"
    logits = backend.next_logits([0, 1])
    assert tuple(logits) == (0.25, 1.5, -1.0)
    assert logits.argmax_token_id == 1


def test_backend_incremental_session_reuses_kv_cache() -> None:
    backend = _backend()

    session = backend.start_incremental_logits([0, 1])
    assert tuple(session.next_logits()) == (0.25, 1.5, -1.0)
    session.append(2)
    assert tuple(session.next_logits()) == (0.25, 1.5, -1.0)

    assert backend._model.calls == [
        ([[0, 1]], True, None),
        ([[2]], True, "fake-cache-1"),
    ]


def test_backend_renders_non_thinking_chat_prompt() -> None:
    backend = _backend()

    rendered = backend.render_chat_prompt(
        "日記を書いて",
        system_prompt="自然な日本語で",
        enable_thinking=False,
    )

    assert rendered == "CHAT:system:自然な日本語で|user:日記を書いて|assistant:"


def test_backend_renders_assistant_continuation() -> None:
    backend = _backend()

    rendered = backend.render_chat_continuation(
        "設定を維持して続きを書いて",
        "今日は研究室で作業していた。",
        system_prompt="自然な日本語で",
        enable_thinking=False,
    )

    assert rendered == (
        "CHAT:system:自然な日本語で|user:設定を維持して続きを書いて|"
        "assistant:今日は研究室で作業していた。"
    )


def test_backend_continuation_does_not_reparse_think_delimiter_in_cover() -> None:
    backend = _backend()

    rendered = backend.render_chat_continuation(
        "短く続きを書いて",
        "本文の途中。</think>この文字列も本文として保持する。",
        system_prompt="自然な日本語で",
        enable_thinking=False,
    )

    assert rendered.endswith("本文の途中。</think>この文字列も本文として保持する。")


def test_backend_rejects_thinking_mode_for_literal_assistant_continuation() -> None:
    backend = _backend()

    with pytest.raises(ValueError, match="requires enable_thinking=False"):
        backend.render_chat_continuation(
            "続きを書いて",
            "本文",
            enable_thinking=True,
        )


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


def test_gptq_artifact_verifier_accepts_four_bit_cuda_model() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "gptq", "bits": 4}),
        hf_device_map={"": "cuda:0"},
    )

    _verify_gptq_model(model, "cuda:0")


def test_gptq_artifact_verifier_rejects_cpu_offload() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "gptq", "bits": 4}),
        hf_device_map={"model": "cuda:0", "lm_head": "cpu"},
    )

    with pytest.raises(ModelDeviceError, match="not fully resident"):
        _verify_gptq_model(model, "cuda:0")


def test_gptq_manifest_is_accepted() -> None:
    manifest = ModelManifest.from_path(
        PROJECT_ROOT / "config" / "models" / "qwen2.5-7b-gptq-int4-quality.json"
    )

    assert manifest.model_id == "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    assert manifest.numeric_policy == "cuda-gptq-int4-same-runtime-device-v1"


@pytest.mark.model
@pytest.mark.skipif(
    __import__("os").environ.get("LSTEG_RUN_GPTQ_MODEL_TESTS") != "1",
    reason=("set LSTEG_RUN_GPTQ_MODEL_TESTS=1 after installing requirements-gptq.txt"),
)
def test_pinned_qwen25_gptq_repeats_incremental_topk() -> None:
    manifest = ModelManifest.from_path(
        PROJECT_ROOT / "config" / "models" / "qwen2.5-7b-gptq-int4-quality.json"
    )
    backend = TransformersBackend.load(
        manifest,
        cache_dir=PROJECT_ROOT / "artifacts" / "model-cache",
        local_files_only=True,
    )
    prompt = backend.render_chat_prompt(
        "大学の研究室での平凡な昼休みについて短く書いてください。",
        enable_thinking=False,
    )
    prefix = backend.tokenize(prompt)
    first = cast(RankedIncrementalLogitsSession, backend.start_incremental_logits(prefix))
    second = cast(RankedIncrementalLogitsSession, backend.start_incremental_logits(prefix))

    first_top = first.top_logits(128, excluded_token_ids=backend.special_token_ids)
    second_top = second.top_logits(128, excluded_token_ids=backend.special_token_ids)
    assert first_top == second_top

    token_id = first_top.token_ids[0]
    first.append(token_id)
    second.append(token_id)
    assert first.top_logits(128, excluded_token_ids=backend.special_token_ids) == second.top_logits(
        128, excluded_token_ids=backend.special_token_ids
    )


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

    continuation = backend.render_chat_continuation(
        "末尾から直接続きを書いてください。",
        "研究室で作業していた。</think>文字列があってもそのまま保持する。",
        system_prompt="自然な日本語で。",
        enable_thinking=False,
    )
    assert continuation.endswith("研究室で作業していた。</think>文字列があってもそのまま保持する。")
    assert "CONTINUE_FINAL_MESSAGE_TAG" not in continuation


@pytest.mark.model
@pytest.mark.skipif(
    __import__("os").environ.get("LSTEG_RUN_MODEL_TESTS") != "1",
    reason="set LSTEG_RUN_MODEL_TESTS=1 after uv sync --extra model",
)
def test_pinned_qwen_kv_cache_repeats_incremental_logits() -> None:
    backend = TransformersBackend.load(
        ModelManifest.from_path(MANIFEST_PATH),
        cache_dir=PROJECT_ROOT / "artifacts" / "model-cache",
        local_files_only=True,
    )
    prefix = backend.tokenize("今日は研究室で再現可能な推論を確認する。")

    first = cast(RankedIncrementalLogitsSession, backend.start_incremental_logits(prefix))
    second = cast(RankedIncrementalLogitsSession, backend.start_incremental_logits(prefix))
    assert first.next_logits().sha256 == second.next_logits().sha256

    token_id = first.next_logits().argmax_token_id
    first.append(token_id)
    second.append(token_id)
    assert first.next_logits().sha256 == second.next_logits().sha256
    assert first.next_logits().argmax_token_id == second.next_logits().argmax_token_id


@pytest.mark.model
@pytest.mark.skipif(
    __import__("os").environ.get("LSTEG_RUN_MODEL_TESTS") != "1",
    reason="set LSTEG_RUN_MODEL_TESTS=1 after uv sync --extra model",
)
def test_pinned_qwen_presence_penalty_ranking_is_repeatable() -> None:
    backend = TransformersBackend.load(
        ModelManifest.from_path(MANIFEST_PATH),
        cache_dir=PROJECT_ROOT / "artifacts" / "model-cache",
        local_files_only=True,
    )
    prefix = backend.tokenize("研究室で昼食前の作業を続けている。")
    first = cast(RankedIncrementalLogitsSession, backend.start_incremental_logits(prefix))
    second = cast(RankedIncrementalLogitsSession, backend.start_incremental_logits(prefix))

    raw = first.top_logits(64)
    repeated_token = raw.token_ids[0]
    penalized_a = first.top_logits(
        64,
        penalized_token_ids=(repeated_token,),
        presence_penalty=0.5,
    )
    penalized_b = second.top_logits(
        64,
        penalized_token_ids=(repeated_token,),
        presence_penalty=0.5,
    )

    assert penalized_a == penalized_b
    if repeated_token in penalized_a.token_ids:
        raw_value = raw.values[raw.token_ids.index(repeated_token)]
        penalized_value = penalized_a.values[penalized_a.token_ids.index(repeated_token)]
        assert penalized_value == pytest.approx(raw_value - 0.5, abs=1e-5)
