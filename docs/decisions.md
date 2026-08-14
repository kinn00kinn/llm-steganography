# 意思決定ログ

大きな判断は ADR として `docs/adr/` に残す。ここでは決定済み事項と、実装前に決める
必要がある項目を一覧管理する。

## 決定済み

| ID | 決定 | 理由 |
|---|---|---|
| D-001 | default branch は `main`、PR + CI + squash merge | main を常に再現可能な状態に保つ |
| D-002 | Python 3.12.10 + uv + lockfile | ローカル/CI の環境を揃える |
| D-003 | payload/coding/model/stego/metrics を分離 | 各層の round-trip と原因切り分けを可能にする |
| D-004 | Pages は生成済み sample の静的 viewer のみ | 公開環境で key/input/backend を扱わない |
| D-005 | control 比較は固定 manifest/seed と token-aware diff | 比較条件を再現可能にする |
| D-006 | UI は sensitive state を永続化しない | public frontend からの漏えいを避ける |
| D-007 | text payload frame v1 | NFC、100 code points、RAW/zlib、10-byte header |
| D-008 | XChaCha20-Poly1305 + HKDF-Expand-SHA256 | 24-byte random nonceと用途別subkey |
| D-009 | 32-bit integer Range Coder + length frame | floatを排除し有限messageを一意に復元 |
| D-010 | Qwen3-1.7B commit pin + same-runtime/device numeric policy | model境界と再現性claimを限定 |

## 実装前に決める事項

| 優先 | 項目 | 推奨初期値 | 決定期限 |
|---:|---|---|---|
| P0 | source code license | Apache-2.0 を候補に owner が選択 | 外部 contribution 前 |
| P0 | security contact/private reporting | GitHub private vulnerability reporting | public demo 前 |
| P0 | prompt/topic の復元 | 固定 versioned prompt で開始 | Phase 6 前 |
| P0 | cover text の canonical transport | exact Unicode string/UTF-8、編集時は復号保証外 | Phase 6 前 |
| P1 | experiment JSON schema | versioned、sensitive fields は既定 redaction | Phase 1～5 |
| P5 | GO/NO-GO capacity margin | 実測 entropy に安全余裕を設定 | Phase 5 |
| P9 | naturalness evaluation corpus | 合成/再配布可能な日本語 corpus と blind 評価 | Phase 9 |
| P9 | public sample export/redaction | allowlist 済み synthetic secret と schema CI | Pages 前 |
| P11 | local API authentication/retention | Pages とは接続せず、必要時に別 ADR | local API 前 |

## License について

public repository でも LICENSE がない限り、第三者へ一般的な利用・改変・再配布権は付与
されない。code license、model license、benchmark dataset license は別々に確認する。

Apache-2.0 は patent grant を明示できるため候補だが、これは repository owner が選ぶ。
選択されるまで LICENSE file を推測で追加しない。

## ADR template

```text
# ADR-NNN: title

- Status: proposed | accepted | superseded
- Date: YYYY-MM-DD
- Owners:

## Context
## Decision
## Alternatives considered
## Consequences
## Compatibility/security impact
```
