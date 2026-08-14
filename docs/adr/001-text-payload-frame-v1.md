# ADR-001: Text payload frame version 1

- Status: accepted
- Date: 2026-08-14
- Owners: repository owner

## Context

暗号やLLMを接続する前に、100文字以内のUnicode秘密文をbounded binary payloadへ変換し、
完全復元とbit数計測を独立に成立させる必要がある。短文では圧縮headerの方が大きくなる
場合があり、decoderはuntrusted bytesから過剰なmemoryを確保してはならない。

## Decision

### Input contract

- Python `str`をUnicode NFCへ正規化する。
- 正規化後のPython `len(text)`をcode point数とし、最大100とする。
- UTF-8 strict encodingを使い、unpaired surrogateを拒否する。
- UTF-8 bytesは最大400とする。
- 空文字、改行、NUL、emojiを許可する。

### Storage selection

raw UTF-8と`zlib.compress(raw, level=9)`を比較し、zlib bodyがstrictly smallerの場合だけ
圧縮する。同じ長さならRAWを選ぶ。decoderは圧縮bodyがraw宣言長以上の非canonical frameを
拒否する。

### Binary layout

すべての整数はunsigned big-endian。

| Offset | Size | Field | Version 1 value |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `LSTG` |
| 4 | 1 | version | `1` |
| 5 | 1 | compression | `0=RAW`, `1=ZLIB` |
| 6 | 2 | decoded UTF-8 byte length | `0..400` |
| 8 | 2 | stored body byte length | `0..400` |
| 10 | variable | body | raw UTF-8またはzlib stream |

frameはheaderが宣言する長さとexactly一致しなければならない。trailing bytes、未知version、
未知compression IDを拒否する。

### Decode validation

1. magic、version、compression、exact frame lengthを検査する。
2. decoded/stored sizeが400 bytes以下か検査する。
3. zlibは最大`declared_size + 1` bytesまで展開し、bomb、truncation、trailing streamを拒否する。
4. decoded sizeとの一致を検査する。
5. UTF-8 strict decodeを行う。
6. textが既にNFCで、100 code points以下であることを検査する。

### Metrics

encoderはnormalized code points、raw/stored/frame bytes、各bit数、compression method、
saved bytes、compression ratioを返す。秘密文そのものを通常logへ出さない。

## Measured examples

Python 3.12.10、zlib level 9での境界確認値。反復入力なので日本語一般の圧縮率を示す
benchmarkではない。

| Input | Code points | Raw bytes | Method | Stored bytes | Frame bits |
|---|---:|---:|---|---:|---:|
| empty | 0 | 0 | RAW | 0 | 80 |
| `秘密` | 2 | 6 | RAW | 6 | 128 |
| `あ` × 100 | 100 | 300 | ZLIB | 15 | 200 |
| decomposed `e + ◌́` × 100 | 100 after NFC | 200 | ZLIB | 13 | 184 |
| `😀` × 100 | 100 | 400 | ZLIB | 17 | 216 |

## Alternatives considered

### Always compress

短い秘密文でframeが不要に大きくなるため不採用。

### Compression flag without decoded length

bounded decompressionと後続coderのtermination判断が難しくなるため不採用。

### CRC/checksum

Phase 2のAEAD authenticationと役割が重複し、最終payload overheadを増やすため不採用。
Phase 1ではstructural/UTF-8 validationのみを行う。

### Grapheme cluster limit

libraryとUnicode versionへの依存が増えるためprotocol limitには使わない。将来UIで参考値を
表示しても、wire contractはNFC後code pointsのままとする。

## Consequences

- inner frame overheadは10 bytes。
- text frame単体には機密性も完全な改ざん検知もない。
- zlib encoder outputはruntime差の影響を受け得るが、decoderは標準zlib streamを受理する。
- Phase 2はこのframeをplaintextとしてAEAD envelopeへ格納する。
- wire behaviorを変える場合はversionを増やし、version 1 decoderを暗黙変更しない。
