# 実装ロードマップ

各 Phase は独立したテスト可能な増分とする。exit criteria を満たすまで次へ進まない。

| Phase | 成果 | 主な exit criteria | LLM/GPU |
|---:|---|---|:---:|
| 0 | repo、CLI、品質基盤 | 全品質チェック成功、CLI 4 command が見える | 不要 |
| 1 | normalization/圧縮 | randomized text round-trip、bit 数計測 | 不要 |
| 2 | 共有鍵/AEAD/framing | 正常復号、wrong key/改ざん拒否、nonce test | 不要 |
| 3 | integer Range Coder | 多様な長さを数千ケース完全復元 | 不要 |
| 4 | model backend | 固定 model から tokenize/logits を再現 | 必要 |
| 5 | 日本語 entropy probe | capacity report と GO/NO-GO 判断 | 必要 |
| 6 | 1-bit/token spike | process をまたぐ短い payload の完全復元 | 必要 |
| 7 | LLM + Range Coding | integer frequencies で end-to-end 復元 | 必要 |
| 8 | 100 文字試験 | cover 上限を 1000 から段階的に短縮 | 必要 |
| 9 | 自然さ・capacity | reliability を保った benchmark 比較 | 必要 |
| 10 | 再現性 hardening | PC 再起動後まで互換、artifact mismatch 検出 | 必要 |
| 11 | API | encode/decode endpoint と key 配置設計 | 必要 |
| 12 | Docker/deploy | 固定 manifest で別環境の互換性を検証 | 必要 |

## Phase 0 — 開発基盤（現在）

成果物:

- Python 3.12.10 pin と `uv.lock`
- `src` layout の installable package
- `pytest`、`ruff`、`mypy`
- `steg keygen|encode|decode|benchmark` の CLI skeleton
- 要件、設計、開発ガイド、repository instructions

実コマンドは stub でよい。ここでは暗号や偽の stego 動作を先回りして実装しない。

## Phase 1 — text payload codec

実装対象:

- `normalize_secret(text)`
- `compress(data)` / `decompress(data)`
- 入力上限と圧縮方式を表す型
- raw/compressed bytes と bit count の計測

テスト対象:

- 空文字、ASCII、日本語、結合文字、emoji、100 code points、上限超過
- `decode(encode(x)) == NFC(x)` の randomized test
- 不正 compressed data の拒否

圧縮は短文で肥大化し得るため、raw/圧縮済みの短い方を versioned flag で選択する。

## Phase 2 — shared-key payload

実装前に AEAD/KDF/framing の ADR を追加する。master key 生成、file permissions、
atomic/no-clobber write、nonce generation、key separation をテストする。

成功条件:

- 正しい key で text round-trip
- wrong key、1-bit 改変、truncation、未知 version を拒否
- key/secret が CLI output や logs に漏れない

## Phase 3 — integer coding

LLM の代わりに固定・生成 frequency tables を用いる。1 byte、10 bytes、100 bytes、
1 KiB、10 KiB と境界 payload を多数試験し、内部 arithmetic に float を持ち込まない。

この時点の milestone は次の完全復元である。

```text
secret → payload → AEAD frame → coder → symbols
       ←         ←            ←       ←
```

## Phase 4–5 — model と feasibility

小型 model を algorithm debug 用、より大きい model を日本語品質評価用に分ける。
artifact は name だけでなく revision/hash を固定する。Phase 5 では 100～500 程度の
sample から entropy 分布、tokens/character、総 capacity を測る。

500 cover characters で必要 bit 数に届かない場合は、ここで目標変更を提案する。

## Phase 6–8 — end-to-end steganography

まず 1-bit/token spike で context、tokenizer、prompt、keyed mapping の同期だけを検証。
次に deterministic integer frequencies と Range Coding を接続する。秘密文は小さい値から
始め、100 文字時は cover 上限を 1000 → 800 → 700 → 600 → 500 と縮める。

Phase 6 着手前に prompt/topic の再現方法を ADR で決定する。

## Phase 9–12 — 品質、再現性、配備

normal generation と stego generation を blind comparison できる benchmark 形式にする。
同一 process、別 process、再起動後、別端末の順で互換性を拡張する。API と Docker は
codec が安定してから追加し、server の key は request ではなく secret file または
secret manager から供給する。

## 推奨する変更単位

1 Phase をさらに、小さな testable change に分ける。各変更で docs、実装、test を揃え、
品質チェック結果を確認してから checkpoint/commit を作る。model download や GPU setup は
Phase 4 まで行わない。
