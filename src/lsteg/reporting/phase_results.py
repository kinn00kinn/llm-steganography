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
PHASE_FIVE_COMMIT = "d2d680668e4840c6c7405790c53e742180ed142d"
PHASE_RESEARCH_COMMIT = "9faf27e3b22210a32381e73243d713fe79ab6680"

PUBLIC_SAMPLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "raw-short",
        "最小の日本語",
        "2文字の短い秘密文で、暗号化と復元の基本動作を確認します。",
        "秘密",
    ),
    (
        "nfc-normalization",
        "文字表現の統一",
        "見た目が同じでも内部表現が異なる文字を、NFC形式に統一して復元します。",
        "e\u0301を含む公開用サンプル",
    ),
    (
        "japanese-note",
        "一般的な日本語文",
        "短いメモ程度の自然な日本語文を、一文字も変えずに復元します。",
        "今日は研究室に早く着いたので、窓を開けて静かな時間に実験ノートを整理した。",
    ),
    (
        "max-repeat",
        "100文字の上限",
        "仕様上の最大文字数である100文字ちょうどを扱えることを確認します。",
        "あ" * 100,
    ),
    (
        "max-utf8",
        "400バイトの上限",
        "1文字が4バイトの絵文字を100個使い、UTF-8の最大400バイトを確認します。",
        "😀" * 100,
    ),
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
    phase_id: int | str,
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

    phase_five_a = _phase(
        "5A",
        "カバー文の容量計測",
        "completed",
        "15サンプルの初期計測で、下位10%点の容量は173.3 bitsと判明。",
        "容量分布を計測し、要求水準を判定できる基準を作る。",
    )
    phase_five_a["commit"] = PHASE_FIVE_COMMIT
    phase_five_a["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_FIVE_COMMIT}"
    phase_five_a["pull_request_url"] = f"{REPOSITORY_URL}/pull/9"
    phase_five_a["evidence"] = [
        {"label": "計測サンプル", "value": "15"},
        {"label": "下位10%点の容量", "value": "173.3 bits"},
        {"label": "計測した値", "value": "LLMの生のエントロピー"},
        {"label": "注意", "value": "実際の埋め込み容量ではない"},
    ]
    phase_five_a["artifacts"] = [
        _artifact("容量計測スクリプト", "scripts/probe_entropy.py", PHASE_FIVE_COMMIT),
        _artifact("固定モデル設定", "config/models/qwen3-1.7b-debug.json", PHASE_FIVE_COMMIT),
    ]

    phase_five_b = _phase(
        "5B",
        "プロンプト・生成設定の探索",
        "completed",
        (
            "生のLLMエントロピーでは、ニュース形式の下位10%点が"
            "435.3 bitsへ改善。実際の埋め込み容量は再計測が必要。"
        ),
        "比較した設定の中から、次の実チャネル計測に使う候補を選ぶ。",
    )
    phase_five_b["commit"] = PHASE_FIVE_COMMIT
    phase_five_b["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_FIVE_COMMIT}"
    phase_five_b["pull_request_url"] = f"{REPOSITORY_URL}/pull/9"
    phase_five_b["evidence"] = [
        {"label": "初期の下位10%点", "value": "173.3 bits"},
        {"label": "最良候補の下位10%点", "value": "435.3 bits"},
        {"label": "計測した値", "value": "LLMの生のエントロピー"},
        {"label": "次の検証", "value": "整数化後の実チャネル容量"},
    ]
    phase_five_b["artifacts"] = [
        _artifact("容量計測スクリプト", "scripts/probe_entropy.py", PHASE_FIVE_COMMIT),
        _artifact("比較デモ", "scripts/demo_e2e_steg.py", PHASE_FIVE_COMMIT),
        _artifact(
            "計測ベースライン", "config/models/qwen3-1.7b-debug-baseline.json", PHASE_FIVE_COMMIT
        ),
    ]

    phase_six = _phase(
        6,
        "埋め込みチャネルの最小試作",
        "research",
        (
            "mockモデルではビット列の埋め込み・抽出と文字列化を完全往復。"
            "実LLMでの同じ検証が残っている。"
        ),
        "別processと実LLMで、短いpayloadを文字列経由で100%復元する。",
    )
    phase_six["commit"] = PHASE_RESEARCH_COMMIT
    phase_six["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_RESEARCH_COMMIT}"
    phase_six["evidence"] = [
        {"label": "mock埋め込み", "value": "完全往復"},
        {"label": "文字列搬送", "value": "mockで検証済み"},
        {"label": "候補の鍵付き並べ替え", "value": "HMAC-SHA256 v1"},
        {"label": "実LLM/GPU", "value": "未検証 / exit criteria未達"},
    ]
    phase_six["artifacts"] = [
        _artifact("埋め込みエンジン", "src/lsteg/steg/engine.py", PHASE_RESEARCH_COMMIT),
        _artifact("鍵付き候補対応", "src/lsteg/steg/mapping.py", PHASE_RESEARCH_COMMIT),
        _artifact("エンジン試験", "tests/steg/test_engine.py", PHASE_RESEARCH_COMMIT),
        _artifact(
            "搬送不変性試験", "tests/steg/test_tokenizer_transport.py", PHASE_RESEARCH_COMMIT
        ),
    ]

    phase_seven = _phase(
        7,
        "LLMと整数Range Codingの接続",
        "research",
        (
            "任意bytesチャネル、搬送安全フィルタ、コンパクトAEADを先行実装。"
            "実モデル試験と長さの自己復元は未完了。"
        ),
        "実LLMの決定的整数頻度で、暗号化payloadを長さの別途共有なしに完全復元する。",
    )
    phase_seven["commit"] = PHASE_RESEARCH_COMMIT
    phase_seven["commit_url"] = f"{REPOSITORY_URL}/commit/{PHASE_RESEARCH_COMMIT}"
    phase_seven["evidence"] = [
        {"label": "モデル非依存試験", "value": "385 passed"},
        {"label": "任意bytes", "value": "mockで完全往復"},
        {"label": "GPU/実モデル試験", "value": "8 skipped"},
        {"label": "payload長の自己復元", "value": "未実装"},
    ]
    phase_seven["artifacts"] = [
        _artifact("任意bytesチャネル", "src/lsteg/steg/engine.py", PHASE_RESEARCH_COMMIT),
        _artifact("搬送安全フィルタ", "src/lsteg/steg/transport.py", PHASE_RESEARCH_COMMIT),
        _artifact(
            "コンパクト認証付き暗号", "src/lsteg/payload/compact_crypto.py", PHASE_RESEARCH_COMMIT
        ),
        _artifact("統合リリースゲート", "tests/steg/test_release_gate.py", PHASE_RESEARCH_COMMIT),
    ]

    later_phases = [
        phase_five_a,
        phase_five_b,
        _phase(
            "5C",
            "秘密文のLLM圧縮",
            "next",
            "100文字の秘密文を、復元可能なままどこまで小さくできるか計測する。",
            "代表的な100文字の合成秘密文100件の符号長分布を確定する。",
        ),
        _phase(
            "5D",
            "500文字に収まるかの容量判定",
            "planned",
            "秘密文・暗号化・カバー文の実測値をまとめて比較する。",
            "500文字目標を維持できるかGO/NO-GOを決定する。",
        ),
        phase_six,
        phase_seven,
        _phase(
            8,
            "100文字の完全復元試験",
            "planned",
            "cover上限を1000文字から段階的に短縮。",
            "100秘密文字のreliable round-tripを達成する。",
        ),
        _phase(
            9,
            "文章の自然さと容量の改善",
            "planned",
            "control/stegoを同じmanifestで比較。",
            "reliabilityを保ったまま品質・容量指標を改善する。",
        ),
        _phase(
            10,
            "再起動後も復元できる再現性",
            "planned",
            "process、再起動、device差を段階的に検証。",
            "再起動後の互換とartifact mismatch検出を保証する。",
        ),
        _phase(
            11,
            "公開サンプルと比較画面",
            "planned",
            "sanitized sample schemaと比較viewerを安定化。",
            "公開sampleがschema/redaction gateを通過する。",
        ),
        _phase(
            12,
            "GitHub Pagesとローカル配布",
            "planned",
            "生成済みsampleをPagesで継続公開。",
            "固定manifestのsample siteとlocal runtimeを個別検証する。",
        ),
    ]
    return [phase_zero, phase_one, phase_two, phase_three, phase_four, *later_phases]


def _build_sample(sample_id: str, label: str, description: str, secret_text: str) -> JsonObject:
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
        "description": description,
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
        _build_sample(sample_id, label, description, secret)
        for sample_id, label, description, secret in PUBLIC_SAMPLES
    ]
    return {
        "schema_version": 1,
        "project": {
            "name": "llm-steganography",
            "repository": REPOSITORY,
            "repository_url": REPOSITORY_URL,
            "last_completed_phase": "5B",
            "next_phase": "5C",
        },
        "publication": {
            "mode": "static_pre_generated_samples",
            "runtime_api": False,
            "accepts_user_input": False,
            "contains_real_secrets": False,
        },
        "summary": {
            "completed_phases": 7,
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
            "status": "not_validated",
            "title": "文章への埋め込み品質はまだ公開結果にできません",
            "reason": (
                "先行試作はありますが、実LLMでの完全復元、生成条件の固定、"
                "容量・自然さの比較がそろっていないためです。"
            ),
            "required_evidence": [
                "同じ公開プロトコルでの秘密文100%復元",
                "モデル・プロンプト・実行環境の固定",
                "通常生成と埋め込み生成の容量・品質比較",
            ],
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
