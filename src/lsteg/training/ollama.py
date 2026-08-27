"""Small, dependency-free helpers for recording local Ollama teacher identity."""

from __future__ import annotations


def parse_model_identity(raw: object, model: str) -> dict[str, object]:
    """Extract a reproducibility record for ``model`` from ``GET /api/tags``."""
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError("Ollama /api/tags returned an invalid response")
    for item in raw["models"]:
        if not isinstance(item, dict):
            continue
        if item.get("name") != model and item.get("model") != model:
            continue
        digest = item.get("digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"Ollama model {model!r} has no digest")
        raw_details = item.get("details")
        details = raw_details if isinstance(raw_details, dict) else {}
        return {
            "name": item.get("name"),
            "model": item.get("model"),
            "digest": digest,
            "size": item.get("size"),
            "modified_at": item.get("modified_at"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "format": details.get("format"),
        }
    raise ValueError(f"Ollama model {model!r} is not installed")
