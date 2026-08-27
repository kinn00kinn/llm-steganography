# llm-steganography

[![CI](https://github.com/kinn00kinn/llm-steganography/actions/workflows/ci.yml/badge.svg)](https://github.com/kinn00kinn/llm-steganography/actions/workflows/ci.yml)

[フェーズ別の成果と公開sampleを見る](http://s.kinn-kinn.com/llm-steganography/)

共有鍵とローカル LLM を用いて、短い秘密文を自然な日本語の文章へ埋め込み、
同じ鍵で完全に復元するための研究開発プロジェクトです。

**Phase 6 mock channel** まで実装済みで、Phase 7の実Qwen end-to-end検証へ進んでいます。
payload、暗号、integer coding、model推論、stego channelを独立に検証し、`K_stego`による
keyed candidate mappingとarbitrary-byte transportまで接続しています。

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

`keygen`は実装済みです。既存fileを上書きせず、key materialを標準出力へ表示しません。

```powershell
uv run steg keygen --output shared.key
```

`encode`、`decode`、`benchmark`は、対応するend-to-end phaseまで意図的にstubです。

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
- [ADR-002: shared-key and AEAD envelope v1](docs/adr/002-shared-key-aead-v1.md)
- [ADR-003: integer Range Coder v1](docs/adr/003-integer-range-coder-v1.md)
- [ADR-004: pinned model backend v1](docs/adr/004-pinned-model-backend-v1.md)
- [ADR-005: keyed cover channel v1](docs/adr/005-keyed-cover-channel-v1.md)
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

Phase 0〜2のprogress viewerを公開します。各数値は`pages/data/phase-results.json`として
commitし、`scripts/export_phase_results.py --check`で現在のcodec出力と一致することをCIで
検証します。最終的なcontrol/stego比較は実装が成立するPhase 6以降に追加します。

## パッケージ構成

```text
src/lsteg/
  payload/    # 正規化、圧縮、framing、鍵導出、AEAD（Phase 2 実装済み）
  coding/     # integer frequencies と Range Coding（Phase 3 実装済み）
  model/      # tokenizer / pinned LLM backend（Phase 4 実装済み）
  stego/      # 各層を結合する encoder / decoder
  metrics/    # capacity、entropy、benchmark
```

各ディレクトリは、対応する Phase に入るときに追加します。先行して空の構造を
量産せず、テストと一緒に実装します。

## Phase 2 API

```python
from lsteg.payload import (
    create_master_key_file,
    decode_secure_text_payload,
    encode_secure_text_payload,
    read_master_key,
)

create_master_key_file("shared.key")
key = read_master_key("shared.key")
encoded = encode_secure_text_payload("e\u0301 を含む秘密", key)
assert decode_secure_text_payload(encoded.frame, key) == encoded.normalized_text

print(encoded.text_metrics.raw_bits)
print(encoded.secure_metrics.secure_frame_bits)
```

master keyは32 bytesで、暗号用と将来のstego用subkeyへ用途分離します。secure frameは
XChaCha20-Poly1305で暗号化・認証され、wrong keyと改ざんを同じ認証失敗として拒否します。

## Phase 3 API

```python
from lsteg.coding import FrequencyTable, map_bytes_to_symbols, recover_bytes_from_symbols

payload = b"authenticated payload"
table = FrequencyTable([40, 30, 20, 10])
symbols = map_bytes_to_symbols(payload, table)
assert recover_bytes_from_symbols(symbols, len(payload), table) == payload
```

coder内部はinteger演算だけを使用します。Range Coder自体は改ざんを検出しないため、復元した
bytesはPhase 2のAEAD envelopeで必ず認証します。

## Phase 4 model probe

通常のpayload/coding開発はmodel dependencyなしで動く。固定Qwen backendを明示的に使う場合だけ、
optional extraを同期する。

```powershell
uv sync --extra model
uv run --extra model python scripts/probe_model_backend.py
# 取得後のoffline再検証
uv run --extra model python scripts/probe_model_backend.py --local-files-only
```

debug modelは`Qwen/Qwen3-1.7B`のfull commit SHAへ固定する。weightはignoredな
`artifacts/model-cache/`に置き、GitHub Pagesやrepositoryへ含めない。

### Qwen2.5-7B GPTQ quality probe

Qwen3-1.7Bはcontrolとして残したまま、より自然な日本語coverを狙うquality candidateとして
公式4-bit GPTQ checkpoint `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` を直接Transformersから
ロードできます。Ollama APIは使わず、既存の`past_key_values` incremental sessionをそのまま
使うため、各生成位置で全文promptを再評価しません。

GPTQ runtimeは通常のmodel extraから分離しています。

```powershell
uv sync --extra model
uv pip install --python .venv\Scripts\python.exe -r requirements-gptq.txt

# 初回は約5.6 GBのpinned artifactをartifacts/model-cacheへ取得
.venv\Scripts\python.exe scripts/probe_qwen25_gptq.py

# 取得後のoffline probe
.venv\Scripts\python.exe scripts/probe_qwen25_gptq.py --local-files-only
```

probeは4-bit GPTQ artifact、完全GPU常駐、同一prefixのtop-k決定性、KV-cache incremental
latency、peak VRAM、channel entropyを確認します。probeが通った後は既存quality benchmarkへ
manifestを差し替えてexact round-tripを比較できます。

```powershell
.venv\Scripts\python.exe scripts/compare_quality_first_range.py `
    --key-file shared.key `
    --manifest config/models/qwen2.5-7b-gptq-int4-quality.json `
    --semantic-plan guarded
```

## Phase 7 local smoke test

Phase 7では、共有master keyから導出した`K_stego`でcandidate intervalを並べ替え、
任意の暗号化bytesをQwenの生成tokenへ埋め込みます。GPU/model artifactを取得済みなら、
まず実装済みchannelそのものの容量を測定します。Phase 5Bのraw-LM entropy値と、
実際のtop-k/temperature/integer-frequency channel値は区別します。

```powershell
uv run --extra model python scripts/probe_channel_capacity.py --local-files-only --samples 100
```

続いて実モデルのtransport testを実行します。

```powershell
$env:LSTEG_RUN_MODEL_TESTS = "1"
uv run --extra model pytest tests/steg/test_tokenizer_transport.py -v
```

次にsynthetic secretだけを使うsecure-payload smoke testを実行できます。

```powershell
uv run steg keygen --output shared.key
uv run --extra model python scripts/demo_e2e_steg.py --key-file shared.key
```

この段階ではreceiverへencrypted frameのbyte長を内部的に渡しています。最終CLIでは
secure frameのheaderから必要長を自己復元する設計へ置き換えます。

## Quality-first Range-Coder benchmark

The production-oriented quality experiment keeps direct Range Coding and uses the compact
authenticated payload plus a sparse semantic scaffold.  Capacity failures print settled bits,
mean channel entropy, selected-token surprisal, and semantic phase instead of a raw model-context
traceback.

```powershell
uv run --extra model python scripts/compare_quality_first_range.py `
    --key-file shared.key
```

Before testing a new patch candidate locally, run the combined verification gate.

```powershell
uv run --extra model python scripts/verify_release_candidate.py --full --model
```

The `--model` portion requires the pinned local Qwen artifact and CUDA runtime; deterministic
model-neutral integration tests run without it.

## Experimental SynthID-inspired quality channel

The Range-Coder channel remains the baseline.  For cover-quality research, an
experimental candidate-pool channel is also available.  It follows the
quality-first idea behind non-distortionary generative watermarking: first fix
the normal Qwen decoding distribution, then let secret data choose only among
iid samples drawn from that same distribution.  Candidate collisions become
erasures rather than forcing a low-probability token.

```powershell
uv run --extra model python scripts/compare_synthid_channel.py `
    --key-file shared.key
```

The script defaults to Qwen3-1.7B non-thinking decoding parameters
`temperature=0.7`, `top_p=0.8`, `top_k=20`, uses a sentence-scale semantic
scaffold, compares STEG with ordinary sampling from the same base distribution,
and verifies exact secret recovery across the Unicode transport boundary.

## Experimental Japanese-prose LoRA pipeline

The fixed deployment model remains `Qwen/Qwen3-1.7B`.  Naturalness research may adapt its
weights with a tokenizer-preserving LoRA, but the steganography core is not changed by the
training pipeline.  A local `qwen2.5:7b` Ollama model is used only as a teacher and several
accepted completions are collected per deterministic scenario.

The preferred **v2** recipe trains an `all-linear` rank-8 QLoRA adapter with completion-only
cross entropy and sparse exact full-vocabulary `KL(P_base || P_adapter)` anchoring.  Unlike the
original v1 smoke recipe, v2 deliberately sets uniform label smoothing to zero.  PyTorch label
smoothing mixes the one-hot target with a uniform distribution over the whole vocabulary; for
the 151,936-token multilingual Qwen vocabulary that is not the distribution we want to teach.
The v1 recipe is retained only so old smoke adapters remain reproducible.

Training-only dependencies are intentionally kept out of the normal project lock.  On Windows,
prepare the existing model environment and then install the pinned training stack with `uv`:

```powershell
uv sync --extra model
uv pip install --python .venv\Scripts\python.exe -r requirements-training.txt
```

Fail fast before generating a large dataset or starting a long training run:

```powershell
.venv\Scripts\python.exe scripts/probe_lora_training_stack.py --backward-smoke
.venv\Scripts\python.exe scripts/verify_release_candidate.py --full --model --training
```

Generate four filtered teacher completions for each deterministic scenario, then audit the
result before GPU training.  The generator records the local Ollama version and installed
teacher-model digest so the mutable `qwen2.5:7b` tag is not the only provenance record:

```powershell
.venv\Scripts\python.exe scripts/generate_japanese_teacher_data.py
.venv\Scripts\python.exe scripts/audit_japanese_teacher_data.py `
    data/training/japanese-prose-v1.jsonl
```

Train the preferred v2 rank-8/KL recipe:

```powershell
.venv\Scripts\python.exe scripts/train_japanese_lora.py `
    data/training/japanese-prose-v1.jsonl
```

The legacy v1 recipe (`qwen3-1.7b-japanese-prose-lora-v1.json`) preserves the first
label-smoothed experiment for reproducibility.  `qwen3-1.7b-japanese-prose-lora-r8-sft.json`
removes KL as an ablation, and the rank-16 recipe remains an intentionally stronger legacy
ablation.

Evaluation now reports two entropy views.  Full-vocabulary entropy remains diagnostic, while
**channel entropy** applies the same quality profile used by the cover experiment
(`T=0.9`, `top_k=128`, `top_p=0.92`).  Both ratios are two-sided gates: an adapter is rejected
when entropy collapses *or* explodes.  A quality gate also requires at least eight independent
held-out scenarios; the 10-scenario/40-example dataset is therefore a pipeline smoke test only
and can never authorize merge.  Generated samples also receive mechanical Japanese quality
audits for foreign scripts, long Latin runs, exact sentence repetition, and meta text.

The v2 smoke run showed that removing label smoothing alone did **not** fix entropy expansion.
Before changing KL direction, rank, or learning rate, use the one-variable KL sweep to compare
weights `0.2`, `0.5`, and `1.0` with every other recipe field held constant.  The sweep accepts
quality-gate exit code 2 as data rather than aborting, writes one evaluation report per profile,
and produces a compact `kl-sweep-summary.json`.  On the 10-scenario smoke dataset this is only a
distribution diagnostic; use at least 60 scenarios (eight held-out scenarios with the fixed split)
before treating the result as a model-selection signal.

```powershell
.venv\Scripts\python.exe scripts/sweep_japanese_lora_kl.py `
    data\training\japanese-prose-smoke.jsonl
```

For a gate-capable pilot, first generate/audit 60 scenarios and then run the same sweep on that
single frozen dataset.  Do not regenerate teacher completions between KL profiles.

```powershell
.venv\Scripts\python.exe scripts/evaluate_japanese_lora.py `
    data/training/japanese-prose-v1.jsonl `
    artifacts/training/qwen3-1.7b-japanese-prose-lora-v2/adapter
```

Only a passing adapter should be merged:

```powershell
.venv\Scripts\python.exe scripts/merge_japanese_lora.py `
    artifacts/training/qwen3-1.7b-japanese-prose-lora-v2/adapter
```

Merging produces a local artifact plus `MERGED_ARTIFACT.json`.  The merge command checks that
the passing `evaluation.json` contains the SHA-256 of the exact adapter being merged.  Since
QLoRA is trained/evaluated on a 4-bit base but deployment uses a merged floating-point artifact,
validate the merged model once more with the same two-sided entropy gate:

```powershell
.venv\Scripts\python.exe scripts/evaluate_merged_japanese_model.py `
    data/training/japanese-prose-v1.jsonl `
    artifacts/models/qwen3-1.7b-japanese-prose-merged-v1
```

The merged model still does **not** automatically replace the pinned inference manifest.  A later,
explicit model-registration change must record the merged artifact hash and rerun logits, Unicode
transport, capacity, naturalness A/B, and exact sender/receiver round-trip baselines before it can
be used by the steganographic encoder or decoder.

## Direct Ollama Qwen2.5 7B Range-Coder experiment

For Japanese-cover quality, the installed quantized `qwen2.5:7b` Ollama model can be used
directly instead of distilling it back into Qwen3-1.7B.  This path is deliberately separate from
the existing token-ID engine.  Ollama exposes at most 20 `top_logprobs`, so the direct channel
constructs a small prefix-free UTF-8 byte alphabet (16 candidates by default) and Range-codes over
that alphabet.

The performance-sensitive detail is the raw prompt layout.  The Qwen2.5 ChatML prefix is rendered
explicitly and each request is the previous raw prompt plus the selected visible candidate.  With
the model kept resident (`keep_alive=30m`), this layout is intended to maximize Ollama's longest-
common-prefix prompt/KV-cache reuse instead of re-evaluating the whole cover at every token.

First record the exact local Ollama runtime/model identity and measure determinism/cache behavior:

```powershell
.venv\Scripts\python.exe scripts/probe_ollama_direct_backend.py `
    --write-identity
```

The probe reports both raw single-shot determinism and **settled-channel determinism**.  Some
Ollama/CUDA prompt-cache paths can make the first logprob table for an otherwise identical prompt
differ from the warmed-cache table.  Exact transport therefore re-queries the same prefix until
the derived integer frequency table repeats on consecutive calls (two confirmations within four
queries by default), and fails closed if it never settles.  The probe also scans several logprob
quantization levels so numerical jitter can be distinguished from candidate-set churn.

After the determinism gate passes, the probe reports steady-state wall time, requests per accepted
channel step, prompt-eval token count, prompt-eval time, token-eval time, candidate count, and
channel entropy.  `prompt_eval_count` per HTTP request should normally collapse to only a few
tokens after warm-up; a large steady value means the intended prefix cache is not helping on that
runtime.  Confirmation requests preserve prefix-cache reuse, so correctness does not require
disabling caching and falling back to full-prefix evaluation.

Then run the compact-payload exact round-trip and NORMAL/STEG A/B:

```powershell
.venv\Scripts\python.exe scripts/compare_ollama_qwen25_range.py `
    --key-file shared.key
```

The direct path pins both the Ollama model digest and Ollama version through the generated identity
lock.  It uses a persistent localhost HTTP connection, keeps the model loaded, neutralizes
server-side sampling transforms, quantizes returned logprobs before integer frequency allocation,
and accepts only individually valid, prefix-free UTF-8 candidate byte strings.  Existing Qwen3
and LoRA experiments remain available for regression comparison.
