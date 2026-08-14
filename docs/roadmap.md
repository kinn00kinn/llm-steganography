# 実装ロードマップ

各 Phase は独立したテスト可能な増分とする。exit criteria を満たすまで次へ進まない。

現在地: **Phase 4 完了、Phase 5 着手前**。

GitHub Pagesにはprogress viewerを先行配置し、完了済みフェーズのsource/test/sampleだけを
表示する。これはPhase 11/12の完了扱いにはせず、comparison schemaと最終viewerのexit
criteriaは維持する。

| Phase | 成果 | 主な exit criteria | LLM/GPU |
|---:|---|---|:---:|
| 0 | repo、CLI、品質基盤 | 全品質チェック成功、CLI 4 command が見える | 不要 |
| 1 | normalization/圧縮 | randomized text round-trip、bit 数計測 | 不要 |
| 2 | 共有鍵/AEAD/framing | 正常復号、wrong key/改ざん拒否、nonce test | 不要 |
| 3 | integer Range Coder | 多様な長さを数千ケース完全復元 | 不要 |
| 4 | model backend | 固定 model から tokenize/logits を再現 | 必要 |
| 5A | Cover entropy probe | Cover text 100〜1000件の p10 capacity 確定 | 必要 |
| 5B | Prompt/T/model sweep | bits/char が最大化される設定の探索 | 必要 |
| 5C | Secret LM compression | 100秘密文字の実測符号長の確定 (数百bit以下か) | 必要 |
| 5D | End-to-end budget | 実測値に基づき 500文字のGO/NO-GO 判定 | 不要 |
| 6 | 1-bit/token spike | process をまたぐ短い payload の完全復元 | 必要 |
| 7 | LLM + Range Coding | integer frequencies で end-to-end 復元 | 必要 |
| 8 | 100 文字試験 | cover 上限を 1000 から段階的に短縮 | 必要 |
| 9 | 自然さ・capacity | reliability を保った benchmark 比較 | 必要 |
| 10 | 再現性 hardening | PC 再起動後まで互換、artifact mismatch 検出 | 必要 |
| 11 | Sample export/Web UI | sanitized sample schema と静的比較 viewer | 不要 |
| 12 | Pages/Docker deploy | Pages sample site と local runtime を個別検証 | UIのみ不要 |

## Phase 0 — 開発基盤（完了）

成果物:

- Python 3.12.10 pin と `uv.lock`
- `src` layout の installable package
- `pytest`、`ruff`、`mypy`
- `steg keygen|encode|decode|benchmark` の CLI skeleton
- 要件、設計、開発ガイド、repository instructions

実コマンドは stub でよい。ここでは暗号や偽の stego 動作を先回りして実装しない。

## Phase 1 — text payload codec（完了）

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

実装結果:

- NFC後100 code points、最大400 UTF-8 bytes
- 10-byte versioned header
- RAWまたはzlib level 9（小さい方）
- bounded decompression、canonical NFC、UTF-8、lengthの検証
- 1,000件のseeded Unicode round-tripを含む48 tests
- `raw/stored/frame` bytesとbitsのmetrics

wire formatと判断理由は`docs/adr/001-text-payload-frame-v1.md`に固定した。

## Phase 2 — shared-key payload（完了）

実装前に AEAD/KDF/framing の ADR を追加する。master key 生成、file permissions、
atomic/no-clobber write、nonce generation、key separation をテストする。

成功条件:

- 正しい key で text round-trip
- wrong key、1-bit 改変、truncation、未知 version を拒否
- key/secret が CLI output や logs に漏れない

実装結果:

- libsodium CSPRNGで生成する32-byte master key
- versioned 40-byte key file、atomic no-clobber write、owner-only mode
- HKDF-Expand-SHA256による`K_encrypt` / `K_stego`の用途分離
- XChaCha20-Poly1305、24-byte random nonce、16-byte authentication tag
- 10-byte authenticated headerを持つsecure envelope v1
- 50-byte固定overhead、最大460-byte encrypted frame
- wrong key、1-bit改ざん、truncation、未知version/algorithmの拒否
- `steg keygen --output PATH`の実装

wire formatと判断理由は`docs/adr/002-shared-key-aead-v1.md`に固定した。

## Phase 3 — integer coding（完了）

LLM の代わりに固定・生成 frequency tables を用いる。1 byte、10 bytes、100 bytes、
1 KiB、10 KiB と境界 payload を多数試験し、内部 arithmetic に float を持ち込まない。

この時点の milestone は次の完全復元である。

```text
secret → payload → AEAD frame → coder → symbols
       ←         ←            ←       ←
```

実装結果:

- 32-bit integer intervalとE1/E2/E3 renormalization
- total 32,768以下のimmutable cumulative frequency table
- 固定tableとcontext-dependent table provider
- symbol count / coded bit lengthを持つ14-byte finite frame
- arbitrary payload bytes → symbols → exact bytesのtermination-bit protocol
- empty、1 byte、10 bytes、100 bytes、1 KiB、10 KiBのround-trip
- 2,000 seeded random payloads、全256 single-byte値、skewed tableのround-trip
- Phase 2 secure envelope → 4-symbol channel → AEAD復元の統合test

protocolと判断理由は`docs/adr/003-integer-range-coder-v1.md`に固定した。

## Phase 4 — model backend（完了）

実装結果:

- model-neutralな`tokenize` / `detokenize` / `next_logits` interface
- `Qwen/Qwen3-1.7B` model/tokenizerをfull commit SHAへ固定
- Python、Transformers、PyTorch CUDA、dtype、device、numeric policyのstrict manifest
- framework tensorをmodule外へ漏らさないimmutable canonical float32 `Logits`
- deterministic algorithms、cuDNN benchmark無効、TF32無効
- RTX 4060 Laptop GPUでtokenization完全往復と151,936 logitsを取得
- 同一contextの反復logits SHA-256完全一致
- 通常CIから分離したoptional `model` extraと明示`model` test marker

artifact/runtime/numeric boundaryは`docs/adr/004-pinned-model-backend-v1.md`に固定した。

## Phase 5 — 日本語entropyとSecret LM Compression

現在の単純なUTF-8+汎用圧縮では100文字（約2000〜3000 bits）を500文字のカバーテキスト（実測約600 bits）に隠すことは不可能であることが判明した。目標（500文字）を緩和する前に、Secret側にもLLMを用いたlossless圧縮（Arithmetic Coding）を導入し、大幅なオーバーヘッド削減が可能か検証するため、Phase 5を以下の4つに分割する。

### Phase 5A: Cover entropy benchmark
100〜1000件の生成サンプルを用いて、カバーテキストの実測capacityを確定する。
単なる平均ではなく、各サンプルの $\frac{\sum H_t}{\text{Unicode chars}}$ を直接計算し、p10などの分布を出す。また、`enable_thinking=False` と全語彙(`top_k=0`, `top_p=1.0`) を明示的に設定する。

### Phase 5B: Prompt / T / model sweep
temperatureやプロンプト（日記、雑談、旅行記など）を変更し、自然さを損なわずに `bits/char` が最大化される設定を探索する。

### Phase 5C: Secret LM compression
100文字の代表的な秘密文100件について、LLMでlossless圧縮した場合（$- \sum \log_2 P(\text{token})$）の実測bit数を測定し、UTF-8やzlib/brotli等と比較する。数百bit程度に落ちるかを確認する。

### Phase 5D: End-to-end budget
$p90(\text{secret payload} + \text{crypto overhead}) < p10(\text{500-char cover capacity})$ を満たせるか判定し、500文字目標でのGO/NO-GOを最終決定する。無理な場合はカバー文字数や秘密文字数の制約変更を行う。また、AES-SIVの導入など、crypto overheadの削減も検討する。

## Phase 6–8 — end-to-end steganography (Dual-LLM Coding)

Phase 5での実測に基づき、本プロジェクトのステガノグラフィーは**「Secret-sideの圧縮」と「Cover-sideの埋め込み」の両方にLLMのRange Codingを用いる二段階構造（Dual-LLM Coding）**を採用する。秘密文の往復と生成は以下のパイプラインとなる。

```text
             ┌──────────────────────┐
             │ SECRET COMPRESSION   │
             └──────────────────────┘
秘密の自然な日本語100文字
          │
          ▼
       tokenize
          │
          ▼
   Secret Language Model
          │
          ▼
   Arithmetic / Range Coding
          │
          ▼
      数百bit (lossless圧縮)
          │
          ▼
        AEAD
          │
          ▼
      encrypted bits

             ┌──────────────────────┐
             │ COVER GENERATION     │
             └──────────────────────┘
      encrypted bits
          │
          ▼
      RRC / Range
          ▲
          │
       Cover LLM
          │
          ▼
     日本語500文字
```

この構造により、秘密文は「意味圧縮（lossless）」されて極小のbit列となり、暗号化を経てカバーテキストの分布に埋め込まれる。
Phase 6ではまず 1-bit/token spike で context、tokenizer、prompt、keyed mapping の同期を検証する。Phase 7で上記のDual-LLMパイプラインを結合し、Phase 8で100文字秘密文の500文字カバーへの完全往復試験を行う。

## Phase 9–12 — 品質、再現性、UI、配備

normal generation と stego generation を blind comparison できる benchmark 形式にする。
同一 process、別 process、再起動後、別端末の順で互換性を拡張する。Docker/local API は
codec が安定してから追加する。GitHub Pages には、ローカルで生成・検証・redaction 済みの
synthetic sample と静的 viewer だけを配置し、runtime backend へ接続しない。Web UI の
詳細は `docs/web-ui.md` に従う。

## 推奨する変更単位

1 Phase をさらに、小さな testable change に分ける。各変更で docs、実装、test を揃え、
品質チェック結果を確認してから checkpoint/commit を作る。model download や GPU setup は
Phase 4 まで行わない。
