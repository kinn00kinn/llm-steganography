from __future__ import annotations

import pytest

from lsteg.training.ollama import parse_model_identity


def test_parse_model_identity_records_digest_and_quantization() -> None:
    raw = {
        "models": [
            {
                "name": "qwen2.5:7b",
                "model": "qwen2.5:7b",
                "digest": "abc123",
                "size": 4_700_000_000,
                "modified_at": "2026-08-01T00:00:00Z",
                "details": {
                    "format": "gguf",
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }
    identity = parse_model_identity(raw, "qwen2.5:7b")
    assert identity["digest"] == "abc123"
    assert identity["quantization_level"] == "Q4_K_M"


def test_parse_model_identity_rejects_unpinned_or_missing_model() -> None:
    with pytest.raises(ValueError, match="not installed"):
        parse_model_identity({"models": []}, "qwen2.5:7b")
    with pytest.raises(ValueError, match="no digest"):
        parse_model_identity(
            {"models": [{"name": "qwen2.5:7b", "model": "qwen2.5:7b"}]},
            "qwen2.5:7b",
        )
