from __future__ import annotations

from pathlib import Path

import pytest

from lsteg.model.ollama_api import (
    OllamaModelIdentity,
    OllamaRequestMetrics,
    OllamaTokenCandidate,
    candidate_fingerprint,
    quantize_logprob,
)


def test_ollama_identity_round_trip(tmp_path: Path) -> None:
    identity = OllamaModelIdentity(
        model="qwen2.5:7b",
        digest="sha256:abc",
        ollama_version="0.0.1",
        parameter_size="7.6B",
        quantization_level="Q4_K_M",
    )
    path = tmp_path / "identity.json"
    identity.write(path)
    assert OllamaModelIdentity.from_path(path) == identity


def test_ollama_identity_mismatch_fails() -> None:
    expected = OllamaModelIdentity("qwen2.5:7b", "a", "1", "7B", "Q4")
    actual = OllamaModelIdentity("qwen2.5:7b", "b", "1", "7B", "Q4")
    with pytest.raises(Exception, match="identity mismatch"):
        expected.assert_matches(actual)


def test_quantize_logprob_is_symmetric_around_zero() -> None:
    assert quantize_logprob(0.00006, quantum=1e-4) == 1
    assert quantize_logprob(-0.00006, quantum=1e-4) == -1


def test_candidate_fingerprint_tracks_quantized_distribution() -> None:
    first = (
        OllamaTokenCandidate(b"a", "a", -0.1),
        OllamaTokenCandidate(b"b", "b", -0.2),
    )
    second = (
        OllamaTokenCandidate(b"a", "a", -0.10001),
        OllamaTokenCandidate(b"b", "b", -0.20001),
    )
    assert candidate_fingerprint(first) == candidate_fingerprint(second)


def test_request_metrics_default_to_one_http_request() -> None:
    metrics = OllamaRequestMetrics(0.1, 1, 2, 3, 4, 5, 6)
    assert metrics.request_count == 1
