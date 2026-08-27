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
| D-011 | HMAC-SHA256 keyed candidate permutation + arbitrary-byte cover channel | `K_stego`を使用しつつtoken確率質量を維持し、Phase 7のbyte transportを可能にする |
| D-012 | Qwen3 backendは`logits_to_keep=1`でlast-token logitsのみ計算 | 生成時に未使用の全系列logitsを保持せずVRAMを削減。RTX 4060 Laptop上で2 process連続probeが同一SHA256/argmaxを再現したためbaselineを更新 |
| D-013 | stego逐次推論はoptional incremental backend + KV cacheを使用 | promptを1回prefillし、以後は新規tokenだけを`past_key_values`へ入力する。従来`next_logits()`はbaseline/互換用に維持し、mock/backend非対応時はfull-prefix再計算へfallbackする |

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

### D-013: semantic re-anchors use assistant-prefill continuation

Semantic phase changes must not quote the accumulated cover text inside a new user message.  The
Qwen chat template is rendered with the visible cover as the final assistant message and
`continue_final_message=True`, so the model predicts a literal continuation rather than a new
answer that may summarize or repeat the quoted prefix.  Sender and receiver rebuild the same
hidden continuation prompt from the transmitted cover prefix.

## D-015: Anti-loop channel filtering and bounded sentence-tail closure

The small Qwen cover model can enter exact token n-gram loops, especially after
semantic phases have nearly finished.  The stego channel may optionally remove
candidate tokens that would recreate an already-seen token n-gram.  The filter
is deterministic from the visible cover prefix, preserves surviving integer
frequencies, and fails open only if every active candidate would be removed.

Non-payload tail generation uses a dedicated assistant-continuation closure
prompt and stops when the newly generated tail contains the first sentence
terminator.  ASCII `.` is accepted as an emergency terminator so an accidental
English loop cannot consume the full tail budget.  The university diary scaffold
is split into eight smaller phases so a ~300-token cover remains anchored for
most of its payload-bearing span.

### D-016: Qwen3 semantic continuation uses literal assistant prefill, not `continue_final_message`

Qwen3's chat template can rewrite final assistant content around `</think>`, which is
incompatible with Transformers' exact final-message validation for
`continue_final_message=True`. Semantic re-anchors therefore render only the system/user
messages with `add_generation_prompt=True` and `enable_thinking=False`, then append the
visible cover text literally. Qwen's `</think>` token (151668) is also excluded from the
script-level stego candidate alphabet so reasoning delimiters cannot be selected as visible
cover tokens.

### D-018 (experimental): SynthID-inspired empirical candidate-pool Range channel

To separate cover quality from payload mapping, retain the existing integer Range Coder but feed it
small *empirical* distributions constructed from iid samples of the quality-first base decoder.  At
each position the protocol draws `2**b` candidate tokens from the active Qwen distribution using a
context/key-bound PRF.  Duplicate draws are collapsed into frequency counts rather than discarded.
For example `[A,A,A,B,B,C,D,D]` becomes `{A:3,B:2,C:1,D:2}` and that count table is the Range-Coder
alphabet for the position.

For a uniformly random coded bitstream, the conditional output distribution is the empirical count
distribution.  Averaging over iid candidate pools gives the original base-model distribution because
`E[count(x)/M] = p_LM(x)`.  The secure frame is XOR-whitened with a domain-separated HMAC stream before
Range Coding so its deterministic framing bytes behave pseudorandomly to the mapper without adding
transport bytes or modifying AEAD.  The first slot-selection experiment is superseded because
candidate collisions caused complete erasures and could exhaust Qwen3-1.7B's 2048-token context.

This is not an implementation of Google's watermark and has a different goal (exact secret recovery
rather than watermark detection).  It borrows SynthID-Text's quality-first principle: define the base
decoder first, sample candidates from it, and avoid deliberately widening/reweighting the model only
to chase capacity.  A four-token sliding visible context remains part of the PRF seed.  The ordinary
Range-Coder channel remains the production baseline until real-Qwen A/B results justify migration.

### D-019 (experimental): sentence-scale semantic scaffold for Qwen3-1.7B

Because the development model is fixed at Qwen3-1.7B, long-range semantic planning is treated as a
cover-generator limitation rather than something the stego mapper should compensate for by widening
its alphabet.  A separate `university_lunch_micro_cover_plan` refreshes one concrete first-person
fact at nearly every sentence boundary.  It avoids abstract narrator language and fixes time,
characters, route, meal, and return-to-work details.  The older eight-phase scaffold remains
available for regression comparison.

## D-015 — Quality-first stego returns to direct Range Coding; SynthID candidate pools remain experimental

The SynthID-inspired candidate-pool experiments are not the production path for arbitrary secret
payloads.  On the fixed Qwen3-1.7B quality distribution, a 16-draw empirical pool settled only 105
of 528 payload bits before the 2048-token model context was exhausted.  Candidate sampling is useful
for watermark detection signals, but Monte-Carlo pools unnecessarily throw away capacity when the
actual requirement is exact transport of hundreds of chosen bits.

The quality-first path therefore uses the active Qwen distribution directly as the Range-Coder table.
Encrypted frames are transport-whitened before arithmetic coding so fixed protocol headers do not
bias the source bitstream.  To reduce the amount of information that must be carried, an additive
compact secure frame uses AES-256-GCM-SIV with a random 96-bit nonce, a 128-bit tag, and two bytes of
authenticated metadata.  The conservative XChaCha20-Poly1305 v1 frame remains supported unchanged.

## D-020 — Patch verification is based on the actual supplied working tree

Phase-7 research changes are currently ahead of the last committed checkpoint and include a
substantial uncommitted working tree.  Patch candidates must therefore be generated against an
explicit snapshot of the user's actual working tree, not against a reconstructed sequence of older
patches.  Before delivery, the patch is applied to a fresh copy of that snapshot with
`git apply --check`, then the deterministic release gate is executed.  Real-Qwen/CUDA tests remain
a separate local gate because the model artifact is intentionally not committed to the repository.

## D-021 — Quality-first semantic scaffolding constrains state, not exact prose

The fixed Qwen3-1.7B development model loses both naturalness and channel entropy when a hidden
outline prescribes exact sentence content.  The quality-first Range-Coder benchmark therefore uses
a sparse current-phase-only scaffold: future phases are not exposed in the initial prompt, lexical
and mundane factual details remain model choices, and the final carrier phase is open-ended and may
be re-anchored repeatedly until the payload settles.  No payload phase asks the model to conclude;
short sentence closure is performed only after the payload is recoverable.


## D-022 — Japanese naturalness adaptation uses tokenizer-preserving, KL-anchored QLoRA

The development/deployment model family remains `Qwen/Qwen3-1.7B`; naturalness experiments do not
change the tokenizer or the steganographic protocol.  Normal-generation A/B tests showed that many
Japanese failures (mixed scripts, malformed words, time/place drift) also occur without Range
Coding, so a model-distribution adaptation experiment is justified before adding more channel
heuristics.

The first training recipe uses a local `qwen2.5:7b` model only to distill several acceptable
Japanese prose completions for each deterministic scenario.  Mechanical filters reject thinking
markers, disallowed scripts, long Latin runs, markdown/meta responses, and exact sentence loops.
Scenario-level train/eval splitting prevents the four completions of one prompt from leaking across
the evaluation boundary.

QLoRA uses `target_modules="all-linear"`, rank 8, alpha 16, dropout 0.05, completion-only cross
entropy with label smoothing 0.05, and a sparse exact full-vocabulary `KL(P_base || P_adapter)`
term (weight 0.2) at deterministic completion positions.  The KL anchor is a quality/capacity
regularizer: plain SFT is allowed as an ablation, but an adapter is not preferred merely because it
fits the teacher data if its next-token entropy collapses.  Rank 16 is also retained only as an
ablation.

Training artifacts are not automatically trusted by the inference boundary.  A held-out evaluation
must pass the configured NLL/entropy/KL screen before merge.  Merging writes provenance and an
artifact hash, but a separate registration change is required to pin that artifact and rerun the
model logits fingerprint, Unicode transport, capacity, and exact sender/receiver round-trip tests.

The teacher tag is not treated as an immutable identifier: dataset metadata records the local
Ollama model digest.  Evaluation reports bind to the adapter directory hash, and merge refuses a
passing report produced for different adapter bytes.  Since QLoRA optimization occurs against a
4-bit base while deployment uses a floating-point merged artifact, the merged checkpoint receives
a second held-out NLL/entropy/KL and tokenizer-equivalence gate before it can be registered for
steganographic inference.

## D-023 — LoRA v2 removes uniform label smoothing and gates the actual cover-channel entropy

The first 40-example QLoRA smoke adapter improved teacher-forced held-out NLL but did not improve
free Japanese generation reliably.  More importantly, its measured full-vocabulary entropy rose
from about 1.27 to 2.86 bits at sampled held-out positions (ratio about 2.25).  The v1 evaluation
only enforced a *minimum* entropy ratio, so this pathological over-dispersion incorrectly passed.

The preferred v2 recipe therefore removes PyTorch uniform label smoothing (`0.05 -> 0.0`) while
retaining the sparse exact `KL(P_base || P_adapter)` anchor at weight 0.2.  Uniform smoothing mixes
5% target mass over the entire 151,936-token multilingual vocabulary, which is not a Japanese-prose
prior and works against the objective of changing only the useful part of the next-token
probability distribution.  The old v1 recipe remains in the repository solely for reproducibility.

Evaluation is also aligned with the deployed steganographic objective.  In addition to full-vocab
entropy, it computes entropy after the fixed quality profile `temperature=0.9`, `top_k=128`,
`top_p=0.92`.  Both full and channel entropy ratios use two-sided corridors, so an adapter is
rejected for either collapse or explosion.  The merge-quality gate additionally requires at least
eight independent held-out scenarios; the 40-example smoke dataset has only one eval scenario and
therefore cannot produce a passing quality gate.  Mechanical audits of generated samples are
reported as diagnostics; semantic quality still requires human/paired A/B review before model
registration.

## D-024 — Keep forward KL and sweep only its weight after v2 entropy expansion

The v2 40-example smoke adapter removed uniform label smoothing but still increased
full-vocabulary entropy from about 1.27 to 2.60 bits (ratio about 2.04) and the deployed
`T=0.9, top_k=128, top_p=0.92` channel entropy from about 0.79 to 1.54 bits (ratio about 1.96).
Teacher-forced NLL improved and `KL(P_base || P_adapter)` remained about 0.41 nats, so the result
shows that removing label smoothing was insufficient; it does not by itself justify changing the
KL direction or LoRA rank.

Small synthetic low-rank experiments were therefore repeated with the adapter objective varied
independently.  Across clean, moderately broadened, and noisy-teacher simulations, increasing the
existing forward-KL weight consistently pulled full/channel entropy ratios back toward 1.0.
Reverse or symmetric KL did not show a consistent advantage over forward KL at matched weights.
The next real-model experiment therefore changes **only** `kl_weight`: 0.2 (v2 control), 0.5, and
1.0.  Rank 8, learning rate 1e-4, all-linear targets, dropout, completion-only CE, KL row count,
seed, and teacher dataset remain fixed.

`sweep_japanese_lora_kl.py` automates this controlled comparison and deliberately continues when
an evaluation returns quality-gate exit code 2.  A smoke dataset with fewer than eight held-out
scenarios remains incapable of selecting a release adapter; the KL sweep there is useful only for
checking whether entropy explosion responds in the predicted direction.  Final KL selection must
be repeated on one frozen gate-capable pilot dataset so teacher-data variation cannot be confused
with objective variation.

## D-025 — Prefer direct Ollama Qwen2.5-7B evaluation before further 1.7B distillation

The local Japanese-prose LoRA experiments show that Qwen3-1.7B can fit Qwen2.5-7B teacher text
(NLL improves) while free-generation quality remains weak and the deployed cover-channel entropy
is substantially broadened.  Before adding more distillation machinery, the stronger installed
quantized `qwen2.5:7b` model is therefore evaluated directly as the carrier distribution.

The direct integration does **not** pretend Ollama exposes the 128-candidate vocabulary slice used
by the Transformers backend: the native API caps `top_logprobs` at 20.  The channel requests 20
neutral one-step alternatives, deterministically quantizes their logprobs, retains at most 16
high-ranked candidates that are individually valid UTF-8 and mutually prefix-free, applies the
local temperature/nucleus policy, and Range-codes over those byte strings.  Prefix-free bytes make
the received Unicode cover uniquely parseable without importing a second copy of the 7B model or
requiring Ollama to expose token IDs.

Performance is a first-class acceptance condition.  The Qwen2.5 ChatML generation prefix is
rendered explicitly and sent with `raw=true`; generated cover bytes are appended directly to that
raw prompt.  Consecutive requests are therefore exact prefix extensions.  A persistent HTTP/1.1
connection removes localhost reconnect overhead and `keep_alive=30m` keeps weights resident.  The
real-runtime probe records first/steady wall latency and Ollama's prompt-eval metrics; a median
steady `prompt_eval_count` above a few tokens is treated as evidence that prefix/KV caching is not
working effectively and must be investigated before accepting the direct backend.

Reproducibility is pinned by a local identity lock containing the Ollama version, model digest,
parameter size, and quantization level.  Each active logprob is additionally rounded to a fixed
quantum before frequency allocation so insignificant floating-point reporting noise does not alter
the integer Range-Coder table.  Repeated identical-prefix queries must produce the same channel
fingerprint before an E2E benchmark is trusted.

## D-026 — Settle Ollama's cached logprob table instead of trusting one seeded request

The first real Qwen2.5-7B probe on Windows/NVIDIA with Ollama 0.30.11 failed the repeated
single-shot channel fingerprint despite a fixed seed and neutral server-side sampling.  This is a
correctness blocker: Range Coding cannot tolerate even one sender/receiver frequency-table
mismatch.  Ollama's public reproducibility claim for seeded generation is not strong enough for
an exact probability channel, and prompt-cache related first-versus-warmed differences have been
reported on GPU runtimes.

The direct backend therefore keeps prompt caching for speed but no longer trusts a single query.
For each visible prefix it requests the same one-token logprob table until the **derived integer
channel fingerprint** repeats on consecutive calls.  Two consecutive confirmations within four
queries are required by default.  The final accepted table is used by both encoder and decoder;
all request timings are accumulated so benchmark latency reflects the correctness overhead.  A
runtime that alternates or otherwise fails to settle is rejected rather than made probabilistic.

This is preferred to disabling cache because the latter defeats the main performance rationale for
using Ollama one token at a time.  The runtime probe separately reports raw single-shot stability,
raw top-20 candidate membership stability, a logprob-quantum sweep, settled stability, requests per
accepted step, and prompt-cache metrics.  If the settled path remains unstable or requires too many
queries, the Ollama native API is considered unsuitable for production exact transport and the next
backend should expose an explicit incremental KV/logits interface over the installed GGUF model.
