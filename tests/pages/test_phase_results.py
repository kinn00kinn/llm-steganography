from __future__ import annotations

import json
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import cast

from lsteg.payload import decode_text_payload
from lsteg.reporting.phase_results import (
    DEFAULT_OUTPUT,
    PHASE_FOUR_COMMIT,
    PHASE_ONE_COMMIT,
    PHASE_THREE_COMMIT,
    PHASE_TWO_COMMIT,
    PHASE_ZERO_COMMIT,
    JsonObject,
    JsonValue,
    build_document,
    render_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _objects(value: JsonValue) -> Iterator[JsonObject]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def test_committed_phase_results_match_generator() -> None:
    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == render_document()


def test_phase_statuses_are_contiguous_and_honest() -> None:
    document = build_document()
    phases = cast(list[JsonObject], document["phases"])
    expected_ids = [0, 1, 2, 3, 4, "5A", "5B", "5C", "5D", 6, 7, 8, 9, 10, 11, 12]
    assert [phase["id"] for phase in phases] == expected_ids
    assert [phase["status"] for phase in phases[:7]] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert all(phase["status"] == "planned" for phase in phases[7:])
    assert phases[0]["commit"] == PHASE_ZERO_COMMIT
    assert phases[1]["commit"] == PHASE_ONE_COMMIT
    assert phases[2]["commit"] == PHASE_TWO_COMMIT
    assert phases[3]["commit"] == PHASE_THREE_COMMIT
    assert phases[4]["commit"] == PHASE_FOUR_COMMIT


def test_completed_phase_artifacts_exist_and_link_to_fixed_commits() -> None:
    document = build_document()
    phases = cast(list[JsonObject], document["phases"])

    for phase in phases[:5]:
        commit = cast(str, phase["commit"])
        artifacts = cast(list[JsonObject], phase["artifacts"])
        assert artifacts
        for artifact in artifacts:
            path = cast(str, artifact["path"])
            url = cast(str, artifact["url"])
            assert (PROJECT_ROOT / path).is_file()
            assert f"/blob/{commit}/{path}" in url


def test_every_public_sample_is_an_exact_decodable_round_trip() -> None:
    document = build_document()
    samples = cast(list[JsonObject], document["samples"])

    assert len(samples) == 5
    for sample in samples:
        frame = bytes.fromhex(cast(str, sample["frame_hex"]))
        restored = decode_text_payload(frame)
        assert restored == sample["normalized_text"]
        assert sample["restored_text"] == sample["normalized_text"]
        assert sample["exact_match"] is True
        assert len(frame) == cast(JsonObject, sample["metrics"])["frame_bytes"]
        secure_metrics = cast(JsonObject, sample["secure_metrics"])
        assert secure_metrics["algorithm"] == "XChaCha20-Poly1305"
        assert secure_metrics["authenticated"] is True
        assert secure_metrics["overhead_bytes"] == 50


def test_publication_mode_has_no_runtime_or_real_secrets() -> None:
    document = build_document()
    publication = cast(JsonObject, document["publication"])

    assert publication == {
        "mode": "static_pre_generated_samples",
        "runtime_api": False,
        "accepts_user_input": False,
        "contains_real_secrets": False,
    }

    forbidden_keys = {
        "key",
        "key_hex",
        "master_key",
        "nonce",
        "private_endpoint",
        "environment",
    }
    for item in _objects(document):
        assert forbidden_keys.isdisjoint(item)

    forbidden_fragments = ("C:\\Users\\", "/home/", "BEGIN PRIVATE KEY", "LSTEG_MASTER_KEY")
    for value in _strings(document):
        assert not any(fragment in value for fragment in forbidden_fragments)


def test_static_site_references_only_committed_same_origin_assets() -> None:
    html = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "pages" / "site.js").read_text(encoding="utf-8")
    css_revision = sha256((PROJECT_ROOT / "pages" / "site.css").read_bytes()).hexdigest()[:12]
    js_revision = sha256((PROJECT_ROOT / "pages" / "site.js").read_bytes()).hexdigest()[:12]

    assert f"./pages/site.css?v={css_revision}" in html
    assert f"./pages/site.js?v={js_revision}" in html
    assert "./pages/data/phase-results.json" in html
    assert "script-src 'self' https://static.cloudflareinsights.com" in html
    assert "connect-src 'self'" in html
    assert "./pages/data/phase-results.json" in javascript
    assert "fetch(RESULTS_URL" in javascript
    assert "https://" not in javascript


def test_static_site_prioritizes_current_results_and_progressive_disclosure() -> None:
    html = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "pages" / "site.js").read_text(encoding="utf-8")

    assert 'id="results"' in html
    assert 'id="samples"' in html
    assert 'id="phases"' in html
    assert 'class="planned-phases"' in html
    assert "文章への埋め込みはまだ未実装" in html
    assert "inner frameを表示" in javascript
    assert 'setAttribute("aria-pressed"' in javascript

    assert "status-console" not in html
    assert "bit-line" not in html
    assert "<form" not in html
    assert "<input" not in html


def test_json_artifact_is_valid_utf8_json() -> None:
    parsed = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert parsed == build_document()
