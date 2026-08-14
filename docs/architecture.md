# アーキテクチャ

## 1. 基本方針

システムを、独立に round-trip 検証できる 5 層へ分割する。

```text
secret text
    │
    ▼
[payload] normalize → compress → frame → AEAD encrypt
    │ encrypted bytes
    ▼
[coding] bytes/bits ↔ integer arithmetic coder ↔ symbols
    │                                      ▲
    ▼                                      │ integer frequencies
[stego] keyed symbol/token mapping ← [model] tokenizer + logits
    │
    ▼
cover text

[metrics] observes every boundary without owning protocol behavior
```

復号は同じ境界を逆向きに通る。model の確率そのものを保存せず、同じ context から
decoder が再計算するため、決定性は protocol の一部である。

## 2. module responsibilities

### `lsteg.payload`

- NFC normalization と入力上限検証
- UTF-8 conversion と可逆圧縮
- versioned binary framing
- master key から用途別 subkey の導出
- AEAD encryption/decryption と認証失敗の抽象化

この層は token、logits、生成文を知らない。

### `lsteg.coding`

- 正の integer frequency と cumulative interval の構築
- arithmetic/range encoder と decoder
- byte/bit boundaries、termination、padding の一意な規約
- property-based/randomized round-trip tests

この層は浮動小数点確率や LLM tensor を受け取らない。

Phase 3では32-bit inclusive intervalとE1/E2/E3 renormalizationを使用し、frequency totalを
32,768以下に制限する。通常のfinite messageは14-byte headerにsymbol数と有効bit数を持つ。
stego方向ではpayload末尾へtermination bitを加え、必要payload prefixがsettleするまでsymbolを
生成する。詳細は`docs/adr/003-integer-range-coder-v1.md`に固定する。

### `lsteg.model`

概念 interface は次のとおり。

```python
class LanguageModelBackend(Protocol):
    def tokenize(self, text: str) -> list[int]: ...
    def detokenize(self, token_ids: Sequence[int]) -> str: ...
    def next_logits(self, token_ids: Sequence[int]) -> Logits: ...
```

`Logits` から integer frequencies への変換は versioned policy として分離する。
temperature、top-k/top-p、禁止 token、quantization、tie-break をすべて固定する。

Phase 4のbackendはQwen/Transformers固有tensorをmodule内でcanonical float32 `Logits`へ変換し、
model/tokenizer commit、runtime、dtype、deviceをJSON manifestへ固定する。numeric policy v1の
互換範囲は同一runtime/deviceとし、異なるCPU/GPU間のraw logits一致は仮定しない。詳細は
`docs/adr/004-pinned-model-backend-v1.md`に固定する。

### `lsteg.stego`

- prompt/context の構築
- 各 token position で model と coder を同期
- `K_stego` による candidate ordering/permutation
- cover の終了条件と長さ制御
- public API の encode/decode orchestration

### `lsteg.metrics`

- entropy/capacity probe
- reliability と cross-process reproducibility の試験
- quality/capacity/time の記録

metrics は protocol の出力を変更してはならない。

## 3. encode sequence

1. 入力を NFC に正規化し、上限を検証する。
2. UTF-8 bytes を圧縮する。
3. version と length を含む plaintext frame を構築する。
4. `K_encrypt` と一意な nonce で AEAD encryption する。
5. encrypted frame を coding state に投入する。
6. 現在の prompt + generated tokens から logits を得る。
7. 固定 policy で integer frequency table を作る。
8. `K_stego` と再現可能な position/context から candidate ordering を作る。
9. coder が選んだ symbol を token として追加する。
10. frame 終端を符号化し終えたら、protocol 規約に従って文章を終了する。

## 4. decode sequence

1. protocol と同じ initial prompt/context を再構成する。
2. cover text を固定 tokenizer で token IDs にする。
3. token ごとに、それ以前の context から logits と integer frequencies を再計算する。
4. 同じ keyed candidate ordering から symbol interval を特定する。
5. coding decoder で encrypted frame を回収する。
6. version と length を検証し、AEAD authentication/decryption を行う。
7. decompress、UTF-8 decode、NFC invariant の確認を行う。

## 5. 決定性 contract

復号互換性を持つ artifact を protocol manifest として固定する。

```text
protocol_version
payload_codec_version
model artifact + revision/hash
tokenizer artifact + revision/hash
backend/runtime version
prompt template version
frequency policy version
candidate mapping version
numeric/device policy
```

単なる乱数 seed だけでは、GPU kernel、library version、同率 logits の扱いによる差を
防げない。encoder/decoder が最終的に使用する integer frequency table の一致を
cross-process test で検証する。

## 6. key separation

概念的には master key から context label 付きで subkey を導出する。

```text
K_master (32 bytes)
   ├── K_encrypt  (AEAD only)
   └── K_stego    (candidate mapping only)
```

Phase 2ではHKDF-Expand-SHA256とversioned context labelで32-byte subkeyを導出する。
暗号はXChaCha20-Poly1305、nonceは暗号化ごとにCSPRNGで生成する24 bytesとした。wire formatと
判断理由は`docs/adr/002-shared-key-aead-v1.md`に固定する。暗号 primitiveやKDFを独自実装しない。

## 7. prompt/topic に関する制約

LLM の次 token 分布は initial prompt に依存する。したがって encoder が任意 topic を
使い、decoder が `cover_text` と `key` しか受け取らない API は、そのままでは復号不能。

初期実験では固定 versioned prompt を用いるのが最も単純である。topic 指定を追加する
場合は、同じ topic を decode 引数または認証された公開 metadata から復元できる設計に
する。決定前に最終 API を固定しない。

## 8. error model

公開 API では最低限、次を区別する。

- invalid input / secret too long
- unsupported protocol or artifact mismatch
- malformed/truncated cover or frame
- authentication failure（wrong key と改ざんを細分化しすぎない）
- model/backend unavailable
- deterministic reconstruction failure

秘密情報や key material を例外 message、trace、benchmark output へ含めない。
