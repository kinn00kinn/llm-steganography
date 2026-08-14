# llm-steganography

[![CI](https://github.com/kinn00kinn/llm-steganography/actions/workflows/ci.yml/badge.svg)](https://github.com/kinn00kinn/llm-steganography/actions/workflows/ci.yml)

[フェーズ別の成果と公開sampleを見る](http://s.kinn-kinn.com/llm-steganography/)

共有鍵とローカル LLM を用いて、短い秘密文を自然な日本語の文章へ埋め込み、
同じ鍵で完全に復元するための研究開発プロジェクトです。

**Phase 1: text payload codec** まで完了しています。LLM や暗号処理を急いで結合せず、
payload、暗号、整数 Range Coding、モデル推論を独立に実装・検証します。次は Phase 2 の
共有鍵、AEAD、secure framing です。

## 目標

- NFC 正規化後 100 Unicode code points 以下の秘密文を扱う
- 共有するのは 256-bit master key のみとする
- 約 400～500 文字の日本語 cover text を最終目標とする
- 正しい鍵とプロトコル条件では秘密文を完全復元する
- RTX 4060 Laptop 8 GB / RAM 16 GB 程度でローカル実行可能にする
- 将来の API 化・コンテナ配備を妨げない構造にする

自然さや 500 文字という長さは研究上の最適化目標です。復号の正確性を先に
成立させ、容量測定の結果を見て実現可能性を判断します。

## クイックスタート

Python 3.12.10 と [uv](https://docs.astral.sh/uv/) を使用します。
`uv` は必要なら指定バージョンの Python も管理できます。

```powershell
uv python install 3.12.10
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run steg --help
```

`keygen`、`encode`、`decode`、`benchmark` は Phase 0 では意図的に
stub です。インターフェースだけを先に固定しています。

pyenv / pyenv-win を使う場合も、リポジトリ直下の `.python-version` が
同じ Python バージョンを選択します。詳しくは
[開発ガイド](docs/development.md)を参照してください。

## ドキュメント

- [要件と成功条件](docs/requirements.md)
- [アーキテクチャ](docs/architecture.md)
- [実装ロードマップ](docs/roadmap.md)
- [開発環境と日常コマンド](docs/development.md)
- [Web UI・比較可視化](docs/web-ui.md)
- [意思決定ログ](docs/decisions.md)
- [ADR-001: text payload frame v1](docs/adr/001-text-payload-frame-v1.md)
- [コントリビューション規約](CONTRIBUTING.md)
- [初期メモ](first.md)

## Repository workflow

`main` は保護対象です。作業は `feat/...`、`fix/...`、`docs/...`、
`chore/...`、`research/...` などの短命 branch で行い、CI が通った Pull Request を
squash merge します。詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## Web UI の位置づけ

ドキュメントとprogress viewerをGitHub Pagesで公開します。ローカルで生成・検証・
redaction済みのsynthetic sampleだけを静的配信し、Pages上での秘密文入力、鍵入力、
Python/LLM推論、API接続は行いません。

Phase 0/1のprogress viewerは公開済みです。各数値は`pages/data/phase-results.json`として
commitし、`scripts/export_phase_results.py --check`で現在のcodec出力と一致することをCIで
検証します。最終的なcontrol/stego比較は実装が成立するPhase 6以降に追加します。

## パッケージ構成

```text
src/lsteg/
  payload/    # 正規化、圧縮、framing（Phase 1 実装済み）、暗号
  coding/     # integer frequencies と Range Coding
  model/      # tokenizer / LLM backend
  stego/      # 各層を結合する encoder / decoder
  metrics/    # capacity、entropy、benchmark
```

各ディレクトリは、対応する Phase に入るときに追加します。先行して空の構造を
量産せず、テストと一緒に実装します。

## Phase 1 API

```python
from lsteg.payload import decode_text_payload, encode_text_payload

encoded = encode_text_payload("e\u0301 を含む秘密")
assert decode_text_payload(encoded.frame) == encoded.normalized_text

print(encoded.metrics.raw_bits)
print(encoded.metrics.frame_bits)
print(encoded.metrics.compression.name)
```

このframeはまだ暗号化・認証されていないinner payloadです。機密用途には使用せず、
Phase 2 でAEAD envelopeに格納します。
