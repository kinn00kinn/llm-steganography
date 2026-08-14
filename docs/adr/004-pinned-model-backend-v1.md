# ADR-004: pinned Qwen model backend and numeric boundary v1

- Status: accepted
- Date: 2026-08-14
- Owners: repository owner

## Context

Phase 4ではcoderやstego orchestrationへLLM固有tensorを漏らさず、固定tokenizerからtoken IDs、
固定causal LMからnext-token logitsを取得する。Phase 7以降のdecoderは各token位置で同じinteger
frequency tableを再構成するため、model名だけでなくartifact commit、runtime、dtype、deviceを
明示する必要がある。

PyTorchはrelease、platform、CPU/GPUが異なる実行間の完全再現を保証せず、同じ数学的処理でも
浮動小数点の演算順序で結果が変わり得ると明記している。そのためseedだけを互換性根拠にしない。

Sources:

- <https://huggingface.co/Qwen/Qwen3-1.7B>
- <https://huggingface.co/docs/transformers/model_doc/qwen3>
- <https://docs.pytorch.org/docs/stable/notes/randomness>
- <https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html>
- <https://docs.astral.sh/uv/guides/integration/pytorch/>

## Decision

### Debug artifact

Phase 4/5のdebug modelを次で固定する。

| Field | Value |
|---|---|
| Model/tokenizer | `Qwen/Qwen3-1.7B` |
| Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| License | Apache-2.0 |
| Transformers | `5.15.0` |
| PyTorch | `2.13.0+cu130` |
| Python | `3.12.10` |
| dtype/device | float16 / `cuda:0` |
| max context in v1 | 2,048 tokens |
| remote code | disabled |

Qwen repositoryのmutable `main`は使わず、modelとtokenizerを同じfull commit SHAへ固定する。
quality評価modelはPhase 5のcapacity probe要件を確認してから別manifestで決定し、このdebug
manifestを暗黙に置換しない。

### Dependency boundary

`torch`と`transformers`は`model` optional extraへ分離する。通常のunit test、CI、payload/coding
利用者はmodel runtimeやGPUを必要としない。Windows/LinuxのCUDA wheelはPyTorch公式
`cu130` indexだけから取得し、他のdependencyはそのindexから解決しない。

weightは`artifacts/model-cache/`へ置き、Git/Pagesへ含めない。repositoryには公開manifest、
bounded probe結果、source/test evidenceだけを残す。

### Interface boundary

公開interfaceは次だけを返す。

```python
token_ids = backend.tokenize(text)
text = backend.detokenize(token_ids)
logits = backend.next_logits(context_token_ids)
```

framework tensorは`lsteg.model`の外へ出さず、next-token vectorをimmutable canonical float32
`Logits`へ変換する。`Logits.sha256`は回帰証跡であり、wire formatや認証値ではない。

### Numeric compatibility

numeric policy v1を`cuda-float16-same-runtime-device-v1`とする。

- deterministic algorithmsを要求する
- cuDNN benchmarkとTF32を無効化する
- Python、OS、PyTorch、Transformers、CUDA、GPU名、compute capabilityを記録する
- 同一processで同じcontextのlogits digestが一致することをPhase 4で検証する
- CPU/GPU、GPU世代、runtime versionをまたぐbitwise一致は主張しない

Phase 7のprotocol互換判定はraw logits digestではなく、最終的に使用するversioned integer
frequency tableの一致で行う。不一致環境でのdecodeはbest effortにせず、manifest mismatchとして
停止させる。

## Consequences

- model変更とcoder変更を独立にtestできる。
- 8 GB VRAM端末で1.7B modelをfloat16実行できる。
- 通常CIは高速・offlineのまま維持できる。
- model integration testは明示markerとlocal artifactを必要とする。
- 異なるdeviceへの互換拡張には別numeric policyとcross-process evidenceが必要になる。

## Security / privacy

`trust_remote_code`は常にfalseとする。probeは入力本文、raw logits、model weightを保存せず、
入力/token列/logitsのdigestとbounded metadataだけを出力する。model出力は秘密性や
steganography耐性を提供しない。
