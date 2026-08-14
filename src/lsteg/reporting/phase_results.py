"""Build the static, redacted phase-results document used by GitHub Pages."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lsteg.payload import (
    decode_secure_text_payload,
    decode_text_payload,
    encode_secure_text_payload,
    encode_text_payload,
    generate_master_key,
)

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]

REPOSITORY = "kinn00kinn/llm-steganography"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "pages" / "data" / "phase-results.json"

PHASE_ZERO_COMMIT = "95a1e4e5019241123e1c7483c9f1a6d2614a42f8"
PHASE_ONE_COMMIT = "60178e9879c36b80e4ecdb525e4d6ce8a1bba437"
PHASE_TWO_COMMIT = "da6846a37341bd768a09436dc1a94bcb2756fa40"
PHASE_THREE_COMMIT = "61035a3347f109743c6a2e5e418942c98ebe3e6f"
PHASE_FOUR_COMMIT = "88189a3a18b55be51193ba19ffc79c54ffd8bcf1"

PUBLIC_SAMPLES: tuple[tuple[str, str, str], ...] = (
    ("raw-short", "短い秘密文", "秘密"),
    ("nfc-normalization", "NFC正規化", "e\u0301を含む公開用サンプル"),
    (
        "japanese-note",
        "日本語メモ",
        "今日は研究室に早く着いたので、窓を開けて静かな時間に実験ノートを整理した。",
    ),
    ("max-repeat", "100文字の境界", "あ" * 100),
    ("max-utf8", "400 UTF-8 bytesの境界", "😀" * 100),
)


def _source_url(path: str, commit: str) -> str:
    return f"{REPOSITORY_URL}/blob/{commit}/{path}"


def _artifact(label: str, path: str, commit: str) -> JsonObject:
    return {
        "label": label,
        "path": path,
        "url": _source_url(path, commit),
    }


def _phase(
    phase_id: int,
    name: str,
    status: str,
    summary: str,
    exit_criterion: str,
) -> JsonObject:
    return {
        "id": phase_id,
        "name": name,
        "status": status,
        "summary": summary,
        "exit_criterion": exit_criterion,
        "evidence": [],
        "artifacts": [],
    }


def _build_phases() -> list[JsonValue]:
    phase_zero = _phase(
        0,
        "開発基盤",
        "completed",
        "Python 3.12、uv、CLI、test/lint/type check、PR運用を固定。",
        "CLI skeletonと全品質チェックが成功する。",
    )
    phase_zero["commit"] = PHASE_ZERO_COMMIT
    phase_zero["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_ZERO_COMMIT}"
    phase_zero["pull_request_url"] = f"{REPOSITORY_URL}/pull/1"
    phase_zero["evidence"] = [
        {"label": "Tests", "value": "6 passed"},
        {"label": "Python", "value": "3.12.10"},
        {"label": "CLI", "value": "4 commands"},
        {"label": "CI", "value": "quality required"},
    ]
    phase_zero["artifacts"] = [
        _artifact("Project configuration", "pyproject.toml", PHASE_ZERO_COMMIT),
        _artifact("CLI skeleton", "src/lsteg/cli.py", PHASE_ZERO_COMMIT),
        _artifact("CI quality gate", ".github/workflows/ci.yml", PHASE_ZERO_COMMIT),
        _artifact("Repository workflow", "CONTRIBUTING.md", PHASE_ZERO_COMMIT),
    ]

    phase_one = _phase(
        1,
        "Text payload codec",
        "completed",
        "NFC、UTF-8、RAW/zlib、versioned frameを完全round-trip。",
        "境界値とrandomized textを100%復元し、bit数を計測できる。",
    )
    phase_one["commit"] = PHASE_ONE_COMMIT
    phase_one["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_ONE_COMMIT}"
    phase_one["pull_request_url"] = f"{REPOSITORY_URL}/pull/2"
    phase_one["evidence"] = [
        {"label": "Tests", "value": "48 passed"},
        {"label": "Random round-trips", "value": "1,000"},
        {"label": "Secret limit", "value": "100 code points"},
        {"label": "Frame header", "value": "10 bytes"},
    ]
    phase_one["artifacts"] = [
        _artifact("Payload codec", "src/lsteg/payload/codec.py", PHASE_ONE_COMMIT),
        _artifact("Binary framing", "src/lsteg/payload/framing.py", PHASE_ONE_COMMIT),
        _artifact("Codec tests", "tests/payload/test_codec.py", PHASE_ONE_COMMIT),
        _artifact("Framing tests", "tests/payload/test_framing.py", PHASE_ONE_COMMIT),
        _artifact(
            "Wire format ADR",
            "docs/adr/001-text-payload-frame-v1.md",
            PHASE_ONE_COMMIT,
        ),
    ]

    phase_two = _phase(
        2,
        "共有鍵・AEAD",
        "completed",
        "用途分離した共有鍵とXChaCha20-Poly1305でpayloadを暗号化・認証。",
        "正しい鍵で復元し、wrong key・改ざん・truncationを拒否する。",
    )
    phase_two["commit"] = PHASE_TWO_COMMIT
    phase_two["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_TWO_COMMIT}"
    phase_two["pull_request_url"] = f"{REPOSITORY_URL}/pull/5"
    phase_two["evidence"] = [
        {"label": "Core test suite", "value": "93 passed"},
        {"label": "Secure round-trips", "value": "500"},
        {"label": "Unique nonce trials", "value": "256 / 256"},
        {"label": "Secure overhead", "value": "50 bytes"},
    ]
    phase_two["artifacts"] = [
        _artifact("Authenticated payload", "src/lsteg/payload/crypto.py", PHASE_TWO_COMMIT),
        _artifact("Key management", "src/lsteg/payload/keys.py", PHASE_TWO_COMMIT),
        _artifact("Secure framing", "src/lsteg/payload/secure_framing.py", PHASE_TWO_COMMIT),
        _artifact("Crypto tests", "tests/payload/test_crypto.py", PHASE_TWO_COMMIT),
        _artifact(
            "Security decision",
            "docs/adr/002-shared-key-aead-v1.md",
            PHASE_TWO_COMMIT,
        ),
    ]

    phase_three = _phase(
        3,
        "Integer Range Coder",
        "completed",
        "整数演算だけでpayloadと固定・動的frequency symbol列を完全往復。",
        "1 byteから10 KiBまで数千ケースを完全復元する。",
    )
    phase_three["commit"] = PHASE_THREE_COMMIT
    phase_three["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_THREE_COMMIT}"
    phase_three["pull_request_url"] = f"{REPOSITORY_URL}/pull/7"
    phase_three["evidence"] = [
        {"label": "Core test suite", "value": "154 passed"},
        {"label": "Seeded round-trips", "value": "2,000"},
        {"label": "Largest payload", "value": "10 KiB"},
        {"label": "Coder state", "value": "32-bit integer"},
    ]
    phase_three["artifacts"] = [
        _artifact("Payload/symbol mapping", "src/lsteg/coding/codec.py", PHASE_THREE_COMMIT),
        _artifact("Integer Range Coder", "src/lsteg/coding/range_coder.py", PHASE_THREE_COMMIT),
        _artifact("Frequency tables", "src/lsteg/coding/frequencies.py", PHASE_THREE_COMMIT),
        _artifact("Coding tests", "tests/coding/test_codec.py", PHASE_THREE_COMMIT),
        _artifact(
            "Protocol decision",
            "docs/adr/003-integer-range-coder-v1.md",
            PHASE_THREE_COMMIT,
        ),
    ]

    phase_four = _phase(
        4,
        "Model backend",
        "completed",
        "固定Qwen artifactとGPU runtimeからmodel-neutralなnext-token logitsを取得。",
        "artifact revisionを固定してlogits interfaceを再現する。",
    )
    phase_four["commit"] = PHASE_FOUR_COMMIT
    phase_four["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_FOUR_COMMIT}"
    phase_four["pull_request_url"] = f"{REPOSITORY_URL}/pull/8"
    phase_four["evidence"] = [
        {"label": "Core test suite", "value": "183 passed"},
        {"label": "Model integration", "value": "30 passed"},
        {"label": "Vocabulary", "value": "151,936 logits"},
        {"label": "Repeated logits", "value": "SHA-256 exact"},
    ]
    phase_four["artifacts"] = [
        _artifact("Model backend", "src/lsteg/model/transformers_backend.py", PHASE_FOUR_COMMIT),
        _artifact("Pinned manifest", "config/models/qwen3-1.7b-debug.json", PHASE_FOUR_COMMIT),
        _artifact(
            "Reproducibility baseline",
            "config/models/qwen3-1.7b-debug-baseline.json",
            PHASE_FOUR_COMMIT,
        ),
        _artifact("Model tests", "tests/model/test_transformers_backend.py", PHASE_FOUR_COMMIT),
        _artifact(
            "Backend decision",
            "docs/adr/004-pinned-model-backend-v1.md",
            PHASE_FOUR_COMMIT,
        ),
    ]

    later_phases = [
        _phase(
            5,
            "日本語entropy probe",
            "next",
            "日本語coverの実測capacityを測定。",
            "500文字目標のGO/NO-GOを実測値で判断する。",
        ),
        _phase(
            6,
            "1-bit stego spike",
            "planned",
            "LLMを介した最小のencode/decode同期を確認。",
            "別processでも短いpayloadを完全復元する。",
        ),
        _phase(
            7,
            "LLM + Range Coding",
            "planned",
            "deterministic integer frequenciesとRange Codingを接続。",
            "小さいpayloadをend-to-endで完全復元する。",
        ),
        _phase(
            8,
            "100文字試験",
            "planned",
            "cover上限を1000文字から段階的に短縮。",
            "100秘密文字のreliable round-tripを達成する。",
        ),
        _phase(
            9,
            "自然さ・capacity",
            "planned",
            "control/stegoを同じmanifestで比較。",
            "reliabilityを保ったまま品質・容量指標を改善する。",
        ),
        _phase(
            10,
            "再現性hardening",
            "planned",
            "process、再起動、device差を段階的に検証。",
            "再起動後の互換とartifact mismatch検出を保証する。",
        ),
        _phase(
            11,
            "Sample export / viewer",
            "planned",
            "sanitized sample schemaと比較viewerを安定化。",
            "公開sampleがschema/redaction gateを通過する。",
        ),
        _phase(
            12,
            "Pages / deploy",
            "planned",
            "生成済みsampleをPagesで継続公開。",
            "固定manifestのsample siteとlocal runtimeを個別検証する。",
        ),
    ]
    return [phase_zero, phase_one, phase_two, phase_three, phase_four, *later_phases]


def _build_sample(sample_id: str, label: str, secret_text: str) -> JsonObject:
    encoded = encode_text_payload(secret_text)
    inner_restored = decode_text_payload(encoded.frame)
    if inner_restored != encoded.normalized_text:  # pragma: no cover - codec invariant
        raise RuntimeError("public sample inner payload did not round-trip")
    sample_master_key = generate_master_key()
    secure = encode_secure_text_payload(secret_text, sample_master_key)
    restored = decode_secure_text_payload(secure.frame, sample_master_key)
    metrics = encoded.metrics
    return {
        "id": sample_id,
        "label": label,
        "synthetic_secret": secret_text,
        "normalized_text": encoded.normalized_text,
        "restored_text": restored,
        "exact_match": restored == encoded.normalized_text,
        "normalization_changed": secret_text != encoded.normalized_text,
        "compression": metrics.compression.name,
        "metrics": {
            "input_code_points": len(secret_text),
            "normalized_code_points": metrics.code_points,
            "raw_bytes": metrics.raw_bytes,
            "raw_bits": metrics.raw_bits,
            "stored_bytes": metrics.stored_bytes,
            "stored_bits": metrics.stored_bits,
            "frame_bytes": metrics.frame_bytes,
            "frame_bits": metrics.frame_bits,
            "bytes_saved": metrics.bytes_saved,
            "compression_ratio": metrics.compression_ratio,
        },
        "secure_metrics": {
            "algorithm": "XChaCha20-Poly1305",
            "frame_bytes": secure.secure_metrics.secure_frame_bytes,
            "frame_bits": secure.secure_metrics.secure_frame_bits,
            "overhead_bytes": secure.secure_metrics.overhead_bytes,
            "authenticated": True,
        },
        "frame_hex": encoded.frame.hex(),
    }


def build_document() -> JsonObject:
    """Build a deterministic public document from allowlisted synthetic samples."""
    samples: list[JsonValue] = [
        _build_sample(sample_id, label, secret) for sample_id, label, secret in PUBLIC_SAMPLES
    ]
    return {
        "schema_version": 1,
        "project": {
            "name": "llm-steganography",
            "repository": REPOSITORY,
            "repository_url": REPOSITORY_URL,
            "last_completed_phase": 4,
            "next_phase": 5,
        },
        "publication": {
            "mode": "static_pre_generated_samples",
            "runtime_api": False,
            "accepts_user_input": False,
            "contains_real_secrets": False,
        },
        "summary": {
            "completed_phases": 5,
            "phase_one_tests": 48,
            "seeded_round_trips": 1_000,
            "secure_round_trips": 500,
            "range_round_trips": 2_000,
            "model_integration_tests": 30,
            "public_samples": len(samples),
        },
        "phases": _build_phases(),
        "samples": samples,
        "comparison": {
            "status": "not_available",
            "available_from_phase": 6,
            "reason": (
                "The language-model steganography path does not exist yet. "
                "Control/stego prose will be published only after a real round-trip works."
            ),
        },
    }


def render_document() -> str:
    """Render the public document in a stable, reviewable form."""
    return json.dumps(build_document(), ensure_ascii=False, indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the artifact is stale.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output: Path = args.output
    rendered = render_document()

    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"stale phase-results artifact: {output}")
            return 1
        print(f"phase-results artifact is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote phase-results artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
