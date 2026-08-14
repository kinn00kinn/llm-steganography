# Model manifests

このdirectoryにはmodel weightではなく、再現に必要な公開manifestだけを置く。weightとcacheは
`artifacts/model-cache/`に保存され、Gitへ追加しない。

`qwen3-1.7b-debug.json`はPhase 4のdebug backendを固定する。

- artifact: `Qwen/Qwen3-1.7B`
- revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- license: Apache-2.0
- purpose: tokenizer / next-token logits interfaceと同一環境再現性の検証

`qwen3-1.7b-debug-baseline.json`は既定の公開probe入力について、入力/token列/logitsの
SHA-256、token数、runtime fingerprintだけを記録する。入力本文、raw logits、weightは含めない。

`main`やversion rangeをmanifestに書かない。artifact、runtime、numeric policyのいずれかを変更する
場合は、新しいmanifest revisionとしてreviewし、既存の復号互換性を暗黙に主張しない。
