# GitHub Pages sample viewer

## 1. 目的と scope

GitHub Pages では、ローカルで一度生成・検証した synthetic sample を静的に公開する。
利用者が sample ごとに次の 4 点を確認できる viewer を作る。

1. 秘密文を埋め込んだ cover text
2. cover text から復元した秘密文と exact-match 結果
3. 同じ条件で秘密を埋め込まず生成した control text
4. control と stego の差、および各 token が運んだ情報量

Pages 上では encode/decode を実行しない。秘密文・鍵の入力、外部API接続、ブラウザ内推論、
GPU job、upload、自由入力 playground は scope 外とする。同一originにcommitされた静的
sample JSONのGETだけを行う。

## 2. publish pipeline

```text
local trusted environment
    │
    ├── encode / control generation / decode
    ├── exact round-trip verification
    ├── metrics + token alignment
    └── export + schema validation + redaction
                         │
                         ▼
              versioned static sample JSON
                         │
                         ▼
              Pull Request + quality checks
                         │
                         ▼
               GitHub Pages sample viewer
```

生成用 key や private experiment record は publish tree にコピーしない。公開 sample は最初
から公開を前提にした架空の秘密文だけを使用する。

## 3. pages

### Sample index

- sample ID、短い説明、protocol/model revision
- secret/cover の文字数、payload bits、bits/token
- round-trip success、生成日時ではなく reproducible source commit
- model/topic/cover 長などによる filter

### Sample detail: Embed

- 公開用 synthetic secret text
- stego cover text
- payload pipeline: raw → compressed → framed → encrypted bits
- cover characters/tokens、bits/token、encode time
- protocol manifest と generation parameters

### Sample detail: Restore

- 同じ stego cover text
- restored synthetic secret
- exact-match badge
- decoder が消費した token/bit と validation status

### Sample detail: Compare

- control text と stego text の side-by-side 表示
- character diff と tokenizer-aware diff
- token ごとの surface、rank、probability/log-probability、entropy、embedded bits
- total NLL/perplexity、反復率、文字/token 数、生成時間
- LCS alignment による insert/delete/change の表示

### Method/About

- protocol/model/tokenizer/prompt/frequency policy revision
- hardware/backend/runtime
- control generation method
- known limitations、threat model、source commit
- sample export/redaction policy

## 4. fair control generation

「何も埋め込んでいない文章」との比較条件を固定する。

- 同じ model/tokenizer revision
- 同じ initial prompt/topic policy
- 同じ temperature/top-k/top-p と禁止 token
- 同じ終了条件と cover length budget
- versioned control seed/sampler

stego text は最初の異なる token 以降、model context 自体が control と異なる。そのため
後続 token の差を「同じ分布から payload が変更した token」とは解釈しない。viewer は
sequence diff と各系列自身の生成確率を分けて表示する。

## 5. no-payload sample

通常生成した control text を decode した結果も sample に含める。decoder は versioned
framing と AEAD authentication が成立しない限り、偶然の bytes を秘密文として表示せず
`no valid payload` とする。

最低 1 組の公開 sample は次を同じ manifest で示す。

```text
synthetic secret
control text (nothing embedded) → no valid payload
stego text (payload embedded)   → restored secret, exact match
control/stego token-aware diff
```

wrong key、改変、非 stego は公開表示上の詳細を過度に区別せず、内部の test では個別に
原因を検証する。

## 6. public sample schema

静的 viewer は versioned JSON だけを読み込む。最低限、次を含める。

```text
schema_version
sample_id
title + description
source_commit
protocol_manifest
generation_parameters
synthetic_secret
payload_metrics
control.text + control.tokens + control.metrics + control.decode_status
stego.text + stego.tokens + stego.metrics
decode.status + decode.restored_text + decode.exact_match
alignment
timings + public hardware summary
warnings
```

token ごとの field は schema で型と上限を定める。raw logits 全体や巨大 tensor は公開せず、
選択 token と比較に必要な上位 candidate のみに制限する。

## 7. publish safety gate

sample export command は private full record から新しい public record を生成し、上書きで
redaction しない。CI はすべての public sample に対して次を検証する。

- JSON schema に一致し、schema version が support 対象
- key/key path、environment variable、absolute path、private endpoint を含まない
- secret は sample allowlist 内の synthetic text
- restored text が synthetic secret と exact match
- control decode status が `no_valid_payload`
- source commit と artifact revision が存在
- text/token/metrics の size 上限以内

## 8. viewer requirements

- 全 sample は read-onlyで、自由入力formや外部network requestを持たない
- 読み込みは同一originのversioned static JSONだけに限定する
- JavaScript 無効時も概要と cover/control/restored text を読める
- diff を色だけで表現せず、記号・label・screen reader text を併用する
- copy button は control/stego/restored の対象を明示する
- 長い token 列は段階表示し、mobile では summary を先に表示する
- build output は生成物とし、source/sample JSON は Pull Request で review する

### Current information design

Phase 0/1の先行viewerは、参考サイト
[blog.kinn-kinn.com](https://blog.kinn-kinn.com/) と同じく、白黒を中心に、太い罫線、
丸角カード、十分な余白、控えめな緑を使う。参考にするのはvisual grammarであり、viewerの
情報構造は研究結果に合わせて次の3点へ限定する。

1. 現在地: 完了phase数、最後の完了phase、次に実装するphase
2. 現在の成果: 検証済みpayload round-tripと生成済みsample
3. 限界と計画: LLM埋め込みが未実装であること、各phaseの完了条件とevidence

未完了phase、frame hex、source artifactはprogressive disclosureで表示する。擬似console、
装飾用bit列、同じ意味のmetricの重複、実装済みと誤認させるcontrol/stego mockは置かない。

## 9. implementation timing

Phase 0/1ではprogress viewerを先行配置し、完了条件、source artifact、payload
round-trip sampleだけを表示する。これはPhase 11の完了を意味しない。Phase 1～5でprivate
experiment recordとpublic sample export schemaをCLI/JSONとして育て、Phase 9のmetricsが
安定した後にcontrol/stego比較を完成させる。
