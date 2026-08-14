共有鍵方式で固定するなら、実装はかなり整理できます。**最初から「100文字→500文字」を狙わず、各層を独立して完成させてから結合する**方針を推します。

現在のCodex CLIはローカルrepo内でコード編集・コマンド実行・レビューまで行え、`/init` で `AGENTS.md` の雛形も生成できます。各フェーズをGit commit単位でCodexに渡す運用と相性がいいです。([OpenAI Developers][1])

# 1. 最終的に作るもの

最終APIはこれくらい単純にします。

```python
steg.encode(
    secret_text="隠したい100文字以内の文章",
    key=key,
    topic="大学生活についての日記"
)

# ->
"今日は研究室に少し早く着いたので、..."
# 400～500文字程度


steg.decode(
    cover_text="今日は研究室に...",
    key=key,
)

# ->
"隠したい100文字以内の文章"
```

CLIなら、

```bash
steg keygen

steg encode \
    --key secret.key \
    --text "秘密の文章"

steg decode \
    --key secret.key \
    --text "生成された文章"
```

を最終形にします。

---

# 2. 全体アーキテクチャ

```text
                 ENCODE

秘密文 ≤100文字
       │
       ▼
Unicode NFC
       │
       ▼
圧縮
       │
       ▼
共有鍵によるAEAD暗号化
       │
       ▼
payload bitstream
       │
       ▼
integer Range Coder
       ▲
       │
LLM ── logits
       │
       ▼
確率 → integer frequency
       │
       ▼
次token決定
       │
       ▼
自然な日本語 400～500文字


                 DECODE

日本語 400～500文字
       │
       ▼
tokenize
       │
       ▼
LLMで各位置のlogits再計算
       │
       ▼
integer frequency
       │
       ▼
Range Decoder
       │
       ▼
encrypted payload
       │
       ▼
AEAD decrypt
       │
       ▼
decompress
       │
       ▼
元の100文字
```

この構造の重要な点は、**暗号・Range Coding・LLMを完全に別moduleにすること**です。

---

# 3. 推奨repo構造

最初からdeployを考えて、こうしておくのがよいです。

```text
linguistic-steg/
│
├─ pyproject.toml
├─ README.md
├─ AGENTS.md
│
├─ src/
│  └─ lsteg/
│     ├─ __init__.py
│     │
│     ├─ payload/
│     │  ├─ codec.py
│     │  ├─ crypto.py
│     │  └─ framing.py
│     │
│     ├─ coding/
│     │  ├─ range_encoder.py
│     │  ├─ range_decoder.py
│     │  └─ frequencies.py
│     │
│     ├─ model/
│     │  ├─ base.py
│     │  ├─ transformers_backend.py
│     │  └─ tokenizer.py
│     │
│     ├─ stego/
│     │  ├─ encoder.py
│     │  └─ decoder.py
│     │
│     ├─ metrics/
│     │  ├─ capacity.py
│     │  ├─ entropy.py
│     │  └─ benchmark.py
│     │
│     └─ cli.py
│
├─ tests/
│
├─ scripts/
│  ├─ probe_model.py
│  └─ benchmark.py
│
└─ deploy/
   ├─ Dockerfile
   └─ compose.yaml
```

これなら途中でLLM backendを交換できます。

---

# 4. Phase 0 — repoとテスト環境

**難易度：★☆☆☆☆**

まだLLMを入れません。

やることは、

* Python project作成
* `pytest`
* `ruff`
* type hint
* CLI skeleton
* Git
* `AGENTS.md`
* CIは後回しでも可

程度です。

最初のcommand：

```bash
steg --help
steg keygen
steg encode
steg decode
steg benchmark
```

ただしencode/decodeはまだstub。

### 完了条件

```bash
pytest
```

が全部通る。

ここでcommit。

```text
feat: initialize project structure
```

---

# 5. Phase 1 — 秘密文をbinaryに変換する

**難易度：★☆☆☆☆**

まだLLMなし。

```text
"こんにちは"
     ↓
NFC normalize
     ↓
UTF-8
     ↓
compress
     ↓
byte stream
```

を実装します。

逆変換も、

```text
byte stream
 ↓
decompress
 ↓
UTF-8 decode
 ↓
"こんにちは"
```

まで。

ここでは暗号化もしません。

### テスト

100～1000個くらいrandomな日本語文字列を生成して、

```python
decode(encode(x)) == x
```

を確認します。

### 同時に測る

秘密文100文字について、

```text
raw bytes
compressed bytes
bit数
```

を記録。

これで初めて、

> 実際に何bitを500文字へ隠す必要があるか

が分かります。

---

# 6. Phase 2 — 共有鍵

**難易度：★☆☆☆☆～★★☆☆☆**

ここで決めた共有鍵方式を追加します。

```text
master.key
= 32 byte random
```

だけを事前共有。

```bash
steg keygen
```

で作成します。

Pythonなら概念的には、

```python
secrets.token_bytes(32)
```

で十分です。

内部ではmaster keyを直接各用途で使わず、

```text
master key
    │
    ├─ K_encrypt
    └─ K_stego
```

と派生させます。

暗号方式はライブラリ実装のAEADを使い、自作暗号はしません。libsodiumはXChaCha20-Poly1305をauthenticated encryptionとして提供しています。([Libsodium Documentation][2])

payloadは例えば、

```text
version
nonce
ciphertext
authentication tag
```

というframeにします。

### 完了条件

```python
decode_payload(
    encode_payload("秘密文章", key),
    key
) == "秘密文章"
```

かつ、

```python
decode_payload(payload, wrong_key)
```

は必ず失敗。

---

# 7. Phase 3 — LLMなしRange Coder

**ここが最重要です。**

**難易度：★★☆☆☆**

まだLLMを触りません。

例えば仮想的な確率を、

```python
freq = {
    "A": 40,
    "B": 30,
    "C": 20,
    "D": 10,
}
```

とします。

秘密bitstreamから、

```text
ABACDBAAC...
```

を生成し、

逆にそこから完全にbitstreamを復元するRange Coderを書く。

### 絶対条件

浮動小数点をRange Coder内部で使用しない。

```text
probability
   ↓
integer frequency

A = 4000
B = 3000
C = 2000
D = 1000
```

として、

```text
cumulative integer interval
```

だけを使います。

### テスト

random payload、

```text
1 byte
10 byte
100 byte
1 KB
10 KB
```

について、

```python
decoded == original
```

を数千回確認。

**ここが100%安定するまでLLMへ進まない**のが重要です。

---

# 8. Phase 4 — LLMをただ動かす

**難易度：★★☆☆☆**

ここで初めてGPUを使います。

私は現在なら、

* 開発時：Qwen3-1.7B
* 本命：Qwen3-4B

という2段階を推します。

Qwen公式はQwen3のdenseモデルとして0.6B、1.7B、4B、8Bなどを公開しており、4Bモデルも32K contextです。([Qwen][3])

1.7Bは**アルゴリズムのデバッグ用**。

4Bは**自然な文章生成の評価用**。

こうすれば毎回4Bを起動する必要がありません。

### このPhaseでは生成しない

まずやるのは、

```python
logits = model.next_logits(tokens)
```

だけ。

APIを、

```python
class LanguageModelBackend:
    def tokenize(text) -> list[int]:
        ...

    def detokenize(tokens) -> str:
        ...

    def next_logits(tokens) -> Tensor:
        ...
```

に固定します。

---

# 9. Phase 5 — 日本語entropy probe

**難易度：★★☆☆☆**

これはかなり重要な実験です。

100～500本くらい普通の日本語文章をLLMに生成させ、

各tokenで、

[
H_t=-\sum_i p_i\log_2p_i
]

を計算します。

出力：

```text
mean entropy     = ? bits/token
median entropy   = ?
p10              = ?
p90              = ?

token / character = ?
```

さらに、

```text
500 Unicode chars
≈ ??? tokens

理論payload
≈ Σ entropy
```

を算出。

ここで初めて、

> 100秘密文字 → 500 cover文字

が実機モデルで可能そうか判断します。

### この時点でGO/NO-GO判定

例えば、

```text
payload = 1650 bit

500文字生成時の総entropy
= 2200 bit
```

ならかなり可能性あり。

逆に、

```text
= 1200 bit
```

なら500文字では物理的に厳しい。

---

# 10. Phase 6 — 超簡単な1 bit/token stego

**難易度：★★★☆☆**

ここでもまだRRCは使いません。

前に話した、

```text
token candidate
 ↓
K_stegoで
0 group / 1 group
```

方式を作る。

これは最終方式ではありません。

目的は、

> **LLMを介した encode → decode が成立することを確認する**

だけ。

例えば秘密情報は、

```text
HELLO
```

くらい。

### 完了条件

```text
secret:
10101100...

        ↓

自然文

        ↓

10101100...
```

が同一processだけでなく、

```bash
steg encode ...
# process終了

steg decode ...
```

でも100%一致。

ここで、

* model
* tokenizer
* prompt
* key

がdecoder側で再現できていることを検証できます。

---

# 11. Phase 7 — LLM + Range Coding

**難易度：★★★★☆**

ここが本プロジェクトの中心です。

```text
LLM logits
   ↓
temperature
   ↓
top-k / top-p
   ↓
softmax
   ↓
integer frequency quantization
   ↓
Range Coder
```

にします。

例えば、

```text
token   float p    integer freq

A       .19233       12605
B       .14012        9183
C       .10211        6692
...
─────────────────────────
total                65536
```

のように、

[
\sum f_i=2^{16}
]

などへ固定。

Range Coderはこのintegerしか見ない。

### ここでK_stegoを使用

共有鍵から、

```text
K_stego
```

を作り、

candidate tokenのrange配置をsecret permutationします。

つまり第三者がLLMを知っていても、

```text
どのrange
→
どのtoken
```

なのかは鍵なしでは分からない設計にします。

---

# 12. Phase 8 — 初めて「100文字」を投入

**難易度：★★★★☆**

ここまでは小さなpayloadだけでいいです。

ここで、

```text
秘密文 = 100 Unicode文字
```

を本当に使います。

最初：

```text
cover max = 1000文字
```

から開始。

成功したら、

```text
1000
↓
800
↓
700
↓
600
↓
500
↓
450
↓
400
```

と削ります。

**最初から500固定にしない**のがポイントです。

---

# 13. Phase 9 — 自然さ

**難易度：★★★★☆**

目標を2つ分離します。

### Reliability

```text
decode success = 100%
```

これは絶対条件。

### Naturalness

その後で、

```text
通常生成
vs
stego生成
```

を比較します。

最低限、

```text
NLL
perplexity
反復率
文字数
token数
bits/token
bits/character
generation time
```

を保存。

CSVに、

```text
experiment_id
model
secret_chars
secret_bits
cover_chars
cover_tokens
bits_per_token
encode_sec
decode_sec
roundtrip_ok
nll
```

を出します。

ここまで来るとかなり研究用ソフトとして扱いやすくなります。

---

# 14. Phase 10 — 再現性問題

**難易度：★★★★★**

ここが最も難しい可能性があります。

### Test A

```text
同一process
encode → decode
```

### Test B

```text
process A encode
process B decode
```

### Test C

```text
PC再起動
encode → decode
```

### Test D

```text
別PC
encode → decode
```

と段階的に試します。

最初の完成条件は **Cまで** でよいです。

Dはdeployフェーズ。

---

# 15. deploymentを考えるのはここから

ここまでPython + Transformersで作ります。

その後、

```python
LanguageModelBackend
```

だけ差し替え可能にします。

候補として `llama.cpp` はGGUFモデルのローカル推論とserver機能を提供しており、現在も継続的に開発されています。([GitHub][4])

ただし、**最初からllama.cppにしない**方がいいです。

今回必要なのは、

```text
full logits
確率分布
独自sampling
```

をかなり細かく触ることなので、まずPyTorch/Transformersでアルゴリズムを完成させる。

その後、

```text
Transformers backend
        ↓
llama.cpp backend
```

を作る方が安全です。

---

# 16. Phase 11 — API

**難易度：★★★☆☆**

FastAPI程度で十分です。

```http
POST /encode

{
  "text": "秘密文",
  "topic": "大学生活"
}
```

↓

```json
{
  "cover_text": "...",
  "cover_chars": 473
}
```

そして、

```http
POST /decode
```

↓

```json
{
  "text": "秘密文"
}
```

### 重要

共有鍵はAPI requestに毎回載せない方がいいです。

server側に、

```text
LSTEG_MASTER_KEY
```

またはsecret fileとして置く。

---

# 17. Phase 12 — Docker / deploy

最終形：

```text
Docker
 │
 ├─ FastAPI
 ├─ model backend
 ├─ model file
 └─ steg codec
```

GPU deploymentなら、

```text
Host GPU
   ↓
NVIDIA runtime
   ↓
Docker container
   ↓
Qwen
```

です。

ここではmodel file/versionも固定します。

```text
protocol_version
model_sha256
tokenizer_sha256
backend_version
frequency_quantizer_version
prompt_version
```

をmanifestとして持たせます。

**これはdeployでは非常に重要です。**

---

# 優先順位をまとめると

| Phase | 内容                     |   難易度 | LLM |
| ----- | ---------------------- | ----: | --- |
| 0     | Repo/CLI/tests         |     ★ | ×   |
| 1     | UTF-8・圧縮               |     ★ | ×   |
| 2     | 共有鍵・AEAD               |    ★★ | ×   |
| 3     | Integer Range Coder    |    ★★ | ×   |
| 4     | Qwen logits取得          |    ★★ | ○   |
| 5     | 日本語entropy測定           |    ★★ | ○   |
| 6     | 1-bit stego            |   ★★★ | ○   |
| 7     | LLM + Range Coding     |  ★★★★ | ○   |
| 8     | 秘密文100文字               |  ★★★★ | ○   |
| 9     | 自然さ・capacity改善         |  ★★★★ | ○   |
| 10    | cross-process/device再現 | ★★★★★ | ○   |
| 11    | FastAPI                |   ★★★ | ○   |
| 12    | Docker/deploy          |  ★★★★ | ○   |

**Phase 0～6までは「本研究の難所に入る前の足場」**です。

Phase 7以降が本番になります。

---

# Codexには一気に全部やらせない

これがかなり重要です。

Codex CLIではrepo単位で継続的に作業でき、公式にもタスク前後でGit checkpointを作ることが推奨されています。また `/review` でworking treeのレビューもできます。([OpenAI Developers][1])

したがって、

```text
Phase 0
↓
test
↓
review
↓
commit

Phase 1
↓
test
↓
review
↓
commit

Phase 2
...
```

とします。

一回のCodex指示では**一Phaseだけ**実装させるのがよいです。

例えば最初にrepoで、

```bash
codex
```

して、

```text
/init
```

で `AGENTS.md` を作れます。これは現在の公式CLIでもサポートされています。([OpenAI Developers][5])

---

# Codexに最初から固定しておく設計原則

`AGENTS.md` には少なくともこれを入れておくとよいです。

```text
Project goals
-------------
- Hide <=100 Unicode characters.
- Generate approximately 400-500 Japanese cover characters.
- Exact decoding is mandatory.
- Naturalness is secondary to reliability until Phase 9.
- Shared-secret-key system only.
- 256-bit random master key.
- Local-first implementation.
- Target hardware: RTX 4060 Laptop 8GB / RAM 16GB.
- Deployment must remain possible.

Architecture
------------
payload -> compression -> encryption
        -> range coding
        -> LLM token generation

Rules
-----
- Never implement custom cryptographic primitives.
- Range coder must use integer arithmetic.
- LLM probability quantization must be deterministic.
- Keep model backend independent from steganography logic.
- Every encoder feature must have a decoder test.
- Never advance a phase while round-trip tests fail.
- Pin model/tokenizer/runtime versions.
- Benchmark every optimization.
```

Qwen3には1.7B/4Bという小型denseモデルがあるので、**1.7BをCI/開発、4Bを実際の品質試験**に分ける構成はこのPC環境とも相性がよいです。([Qwen][3])

## 最初の到達目標

まずCodexには **Phase 0～3だけ**をやらせるのを勧めます。

そこまで完了した時点で、

```text
100文字
 ↓
圧縮
 ↓
共有鍵暗号化
 ↓
bitstream
 ↓
integer Range Coding
 ↓
symbol列
 ↓
完全復元
```

が **LLMなしで100%動く**状態になります。

その状態をこちらで一度確認してからQwenを接続すれば、以後のバグはほぼ「LLM側」と断定できます。これはデバッグ上かなり大きいです。

**次の一手としては、Codexにそのまま渡せる `AGENTS.md` と「Phase 0〜3を順番に実装させる具体的なCodex指示書」まで作る**のがよいです。これを作っておけば、そのまま新規repoに置いて実装開始できます。

[1]: https://developers.openai.com/codex/cli "Codex CLI | ChatGPT Learn"
[2]: https://doc.libsodium.org/quickstart?utm_source=chatgpt.com "Quickstart and FAQ - Libsodium documentation - GitBook"
[3]: https://qwenlm.github.io/blog/qwen3/?utm_source=chatgpt.com "Qwen3: Think Deeper, Act Faster"
[4]: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md?utm_source=chatgpt.com "llama.cpp/tools/server/README.md at master · ggml-org ..."
[5]: https://developers.openai.com/codex/developer-commands "Developer commands | ChatGPT Learn"
