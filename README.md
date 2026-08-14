# llm-steganography

[![CI](https://github.com/kinn00kinn/llm-steganography/actions/workflows/ci.yml/badge.svg)](https://github.com/kinn00kinn/llm-steganography/actions/workflows/ci.yml)

共有鍵とローカル LLM を用いて、短い秘密文を自然な日本語の文章へ埋め込み、
同じ鍵で完全に復元するための研究開発プロジェクトです。

現時点は **Phase 0: 開発基盤** です。LLM や暗号処理を急いで結合せず、
payload、暗号、整数 Range Coding、モデル推論を独立に実装・検証します。

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
- [コントリビューション規約](CONTRIBUTING.md)
- [初期メモ](first.md)

## Repository workflow

`main` は保護対象です。作業は `feat/...`、`fix/...`、`docs/...`、
`chore/...`、`research/...` などの短命 branch で行い、CI が通った Pull Request を
squash merge します。詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## Web UI の位置づけ

将来、ドキュメントと比較 viewer を GitHub Pages で公開する予定です。ローカルで生成・
検証・redaction 済みの synthetic sample だけを静的配信し、Pages 上での秘密文入力、
鍵入力、Python/LLM 推論、API 接続は行いません。

## パッケージ構成

```text
src/lsteg/
  payload/    # 正規化、圧縮、framing、暗号
  coding/     # integer frequencies と Range Coding
  model/      # tokenizer / LLM backend
  stego/      # 各層を結合する encoder / decoder
  metrics/    # capacity、entropy、benchmark
```

各ディレクトリは、対応する Phase に入るときに追加します。先行して空の構造を
量産せず、テストと一緒に実装します。
