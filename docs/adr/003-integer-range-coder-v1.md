# ADR-003: integer Range Coder and finite-message protocol v1

- Status: accepted
- Date: 2026-08-14
- Owners: repository owner

## Context

Phase 3ではLLMを使わず、integer frequency tableとpayloadの間を可逆変換する。最終的な
steganographyでは、senderがpayload bitsをarithmetic-decodeしてtoken symbolを選び、receiverが
同じsymbolをarithmetic-encodeしてpayload bitsを回収する。coderと確率modelを分離し、同じ
frequency table列ならprocessをまたいで同じ結果になるprotocolが必要である。

有限精度arithmetic codingの基本はWitten, Neal, Clearyのinteger implementationと、bit-level
renormalizationのE1/E2/E3条件に従う。range codingはarithmetic codingを任意radixへ一般化した
同系統の手法であり、本実装はbitwise rangeを使用する。

Sources:

- <https://doi.org/10.1145/214762.214771>
- <https://people.xiph.org/~tterribe/notes/range.html>

## Decision

### Integer state

- state precision: 32 bits
- interval: inclusive `[low, high]`
- initial interval: `[0, 2^32 - 1]`
- renormalization: E1 lower half、E2 upper half、E3 middle-half underflow
- coder内の更新、比較、乗算、除算はintegerのみ
- frequency total上限: `2^15 = 32,768`

Python自体はunbounded integerを持つが、`low`、`high`、`code`は各renormalizationで32-bitへ
maskし、protocol stateを固定する。symbol interval更新の中間積だけは正確なintegerとして計算する。

### Frequency table

`FrequencyTable`はsymbol ID順の非負integer列である。totalは1以上32,768以下とする。frequency
zeroのsymbolは選択不能だが、alphabet indexを保持するためtable内に存在してよい。cumulative
intervalと同値時の順序はsymbol ID昇順で固定する。

固定tableに加えて、次の形のproviderを許可する。

```python
table = provider(position, decoded_prefix)
```

encoderとdecoderは同じposition/prefixから同じtableを再構成しなければならない。Phase 7では
model logitsのfilter/quantization結果をこのinterfaceへ接続する。

### Finite range frame v1

通常のsymbol圧縮round-tripでは、symbol数を外部の曖昧な状態に置かずframeへ格納する。

```text
offset  bytes  field
0       4      magic = "LRNG"
4       1      version = 1
5       1      algorithm = 1 (32-bit integer E1/E2/E3)
6       4      symbol count, unsigned big-endian
10      4      meaningful coded bit length, unsigned big-endian
14      N      MSB-first coded bits, final unused bits are zero
```

symbol数上限は1,048,576。bit数上限は`symbol_count_limit * 32 + 2`。空symbol列はbit数0の
canonical frameだけを許可する。decoderはdeclared symbol countだけ復元し、bitstream末尾を
arithmetic decoder規約のzeroで仮想的に延長する。

### Payload-to-symbol mapping

steganography方向では、任意のpayload bytesをarithmetic decoderの入力とする。有限payloadを
そのままzeroで延長すると、dyadic interval境界上で最後のpayload bitsが永久にsettleしない場合が
ある。そのためsenderはpayload末尾へtermination bit `1`を追加し、その後をzeroで延長する。

senderは次を同時に進める。

1. terminated payloadを`RangeDecoder`へ入力してsymbolを選ぶ。
2. 同じtable/symbolをmirror `RangeEncoder`へ入力する。
3. mirrorが元payload長以上のprefix bitsをsettleした時点で終了する。

receiverはsymbol列を`RangeEncoder`へ入力し、settled prefixの先頭`payload_size * 8` bitsを返す。
termination bitは返さない。Phase 3 APIではpayload byte lengthを明示する。Phase 7では最初に
復元される`LSEC` headerのciphertext lengthからsecure frame全長を判断する接続層を追加する。

channel capacityが不足し、指定symbol budgetまでに必要bitsがsettleしない場合は
`InsufficientRangeDataError`とする。

### Integrity boundary

Range Coderは誤り検出や認証を提供しない。symbol変更で別の有効bitstreamになる場合がある。
回収したbytesはPhase 2のsecure envelope parserとAEAD authenticationで必ず検証する。

## Alternatives considered

### Floating-point interval

長いmessageでrounding差が蓄積し、backend/device差をdecoderが再現できないため不採用。

### EOF symbolだけで終了

LLM token alphabetへprotocol専用symbolを追加できず、modelのEOS tokenは生成終了の意味も持つ。
payload lengthとgeneration terminationを混同しないため、Phase 3ではlength-framed messageを採用。

### Frequency tableをframeへ格納

Phase 7ではtableを各token contextのmodel logitsから再計算するため、巨大になりprotocol目的と
一致しない。frameはcoder/lengthだけを保持し、frequency policyはprotocol manifestで固定する。

### rANS

bitstreamとsymbol列の双方向mappingに適するが、project briefでinteger Range Coderを選択済みで
あり、LLMの逐次contextとFIFO処理を明示的に検証するため今回は不採用。

## Consequences

- 固定table、zero-frequency symbol、context-dependent tableを同じcoreで扱える。
- 1 byte〜10 KiBと2,000 seeded random casesをCPUだけで試験できる。
- secure payloadをsymbol channelへ通して完全復元できる。
- compression ratioはmodel品質の結果であり、coderの成功条件には含めない。
- Phase 7までにfrequency quantizationとpayload length discoveryをversioned policyとして追加する。

## Compatibility/security impact

state bits、renormalization、frequency上限、symbol ordering、frame layout、termination bitの変更は
新しいprotocol versionを必要とする。Range Coder単体の出力を認証済みdataとして扱わない。
