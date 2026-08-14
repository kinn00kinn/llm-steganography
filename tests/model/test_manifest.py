from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsteg.model import NUMERIC_POLICY, InvalidModelManifestError, ModelManifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "config" / "models" / "qwen3-1.7b-debug.json"


def test_debug_manifest_pins_artifacts_runtime_and_license() -> None:
    manifest = ModelManifest.from_path(MANIFEST_PATH)

    assert manifest.model_id == "Qwen/Qwen3-1.7B"
    assert manifest.model_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert manifest.tokenizer_revision == manifest.model_revision
    assert manifest.license == "Apache-2.0"
    assert manifest.transformers_version == "5.15.0"
    assert manifest.torch_version == "2.13.0+cu130"
    assert manifest.numeric_policy == NUMERIC_POLICY
    assert manifest.trust_remote_code is False


def test_manifest_has_a_stable_json_compatible_representation() -> None:
    manifest = ModelManifest.from_path(MANIFEST_PATH)
    parsed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest.as_dict() == parsed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "unsupported"),
        ("role", 1, "debug or quality"),
        ("model_revision", "main", "commit SHA"),
        ("tokenizer_revision", "v1", "commit SHA"),
        ("transformers_version", ">=5", "exact version"),
        ("torch_version", "latest", "exact version"),
        ("dtype", "auto", "float16"),
        ("device", "auto", "cuda:0"),
        ("numeric_policy", "best-effort", "unsupported"),
        ("trust_remote_code", True, "remain false"),
        ("license", 1, "must not be empty"),
        ("max_context_tokens", 0, "between"),
    ],
)
def test_manifest_rejects_unpinned_or_unsafe_values(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw[field] = value

    with pytest.raises(InvalidModelManifestError, match=message):
        ModelManifest.from_mapping(raw)


def test_manifest_rejects_missing_and_unknown_fields() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    del raw["license"]
    raw["revision_alias"] = "main"

    with pytest.raises(InvalidModelManifestError, match=r"missing=.*license.*unknown"):
        ModelManifest.from_mapping(raw)
