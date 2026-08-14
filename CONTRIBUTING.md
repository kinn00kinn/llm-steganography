# Contributing

## 基本フロー

`main` は常に test 済みで、直接作業しない。

```powershell
git switch main
git pull --ff-only
git switch -c feat/short-description
```

1 branch は 1 つの issue、研究仮説、または変更目的だけを扱う。実装と対応する test、
必要な docs を同じ Pull Request に含める。

## branch naming

形式は `<type>/<lowercase-kebab-case>` とする。

| Prefix | 用途 | 例 |
|---|---|---|
| `feat/` | 利用者向け機能 | `feat/payload-normalization` |
| `fix/` | bug fix | `fix/frame-length-check` |
| `docs/` | 文書のみ | `docs/threat-model` |
| `chore/` | repository/tooling | `chore/repository-governance` |
| `research/` | 仮説検証・benchmark | `research/qwen-entropy-probe` |
| `refactor/` | 動作を変えない再構成 | `refactor/coder-state` |
| `test/` | test 基盤 | `test/randomized-payloads` |
| `ci/` | CI/CD | `ci/pages-preview` |
| `hotfix/` | 緊急修正 | `hotfix/key-log-redaction` |
| `release/` | release 準備 | `release/0.1.0` |

GitHub が作る `dependabot/` branch も許可する。人が作る branch では氏名や連番だけの
名前を避け、目的が一覧で分かる名前にする。

## commits and Pull Requests

commit subject は Conventional Commits を使う。

```text
feat: add NFC payload normalization
fix: reject truncated encrypted frames
docs: define prompt reconstruction policy
research: record entropy probe schema
```

Draft の間は途中 commit が複数あってよい。最終的には GitHub の squash merge だけを
使い、PR title を最終 commit subject にする。

PR の条件:

- purpose、scope、security/reproducibility impact を記載する
- encoder の変更には decoder round-trip test を付ける
- protocol behavior の変更には versioning/migration 判断を記載する
- `quality` check が成功している
- review conversation が解決済み
- key、secret、model weight、巨大な benchmark artifact を含まない

solo 開発中は approval 0 件で merge 可能だが、PR と CI は省略しない。collaborator が
増えた時点で required approval を 1 件へ引き上げる。

GitHub 上の `main` ruleset は `.github/rulesets/` の JSON と一致させる。UI だけで設定を
変更せず、JSON を Pull Request で更新してから適用する。branch 名は required `quality`
workflow で検査するため、不正な名前の同一 repository branch は merge できない。

## local quality gate

```powershell
uv sync --frozen --dev
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src tests
```

LLM/GPU test は通常 CI と分離し、model artifact、revision、hardware、runtime を結果に
記録する。

## security reports

鍵、実在する秘密文、未公開 vulnerability を public issue に投稿しない。
[SECURITY.md](SECURITY.md) の方針に従う。
