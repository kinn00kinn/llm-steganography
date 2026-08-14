# 要件と成功条件

## 1. 目的

本プロジェクトは、共有鍵を持つ送信者と受信者の間で、短い秘密文を日本語の
cover text に埋め込み、完全に復元できるローカル実行の linguistic
steganography システムを研究・実装する。

暗号化と steganography は別の性質を扱う。AEAD は秘密文の機密性と改ざん検知を
担当するが、生成文が stego text だと見抜かれないことまでは保証しない。

## 2. 用語と計数規則

- **secret text**: 埋め込む元の Unicode 文字列。
- **cover text**: LLM が生成する、payload を運ぶ日本語文字列。
- **master key**: 事前共有する 32-byte の一様ランダム値。
- **payload**: 正規化、圧縮、framing、暗号化後の byte sequence。
- **文字数**: 特記がなければ NFC 正規化後の Python `len(text)`、すなわち
  Unicode code point 数。grapheme cluster 数ではない。
- **token 数**: 固定した tokenizer revision による token ID 数。

## 3. 機能要件

### 3.1 鍵管理

- `steg keygen` は CSPRNG から 32-byte master key を生成する。
- master key を暗号用途と stego 用途に直接兼用せず、用途別の subkey を導出する。
- key file の誤上書きを既定動作にしない。
- key や秘密文を通常ログへ出力しない。

### 3.2 payload codec

- secret text を NFC 正規化し、UTF-8 bytes へ変換する。
- 可逆圧縮を適用し、効果がない場合を含めて deterministic に framing する。
- library-provided AEAD で暗号化・認証する。
- frame は少なくとも protocol version、圧縮方式、nonce、payload length、
  ciphertext と authentication tag を復元可能に表現する。
- wrong key、破損 frame、未知 version、上限超過を明確なエラーとして拒否する。

### 3.3 coding

- Range Coder 内部は整数演算のみを用いる。
- 同じ frequency table と payload から同じ symbol sequence を得る。
- 空、短い入力、境界値、大きな入力を含む round-trip を保証する。
- payload の終端または長さを、decoder が曖昧なく判断できる framing にする。

### 3.4 model backend

- tokenize、detokenize、next-token logits の最小 interface を提供する。
- model 固有の実装を stego/coding/payload 層から隔離する。
- model、tokenizer、prompt、推論 runtime の revision を固定・記録する。
- probability filtering、frequency quantization、同値時の順序を決定的にする。

### 3.5 stego encode/decode

- encoder と decoder は、同じ鍵、protocol manifest、prompt context から各位置の
  frequency table と keyed mapping を再現する。
- process 終了後や PC 再起動後も復号できることを最初の完成条件とする。
- 別端末での一致は deployment phase の条件とする。

### 3.6 metrics

各実験について最低限、以下を機械可読な形式で保存する。

- experiment ID と protocol/model/tokenizer/prompt revision
- secret characters、raw/compressed/encrypted bits
- cover characters、cover tokens、bits/token、bits/character
- encode/decode time と round-trip result
- NLL、perplexity、反復率

## 4. 品質属性

優先順位は次のとおり。

1. 正しい条件での exact round-trip
2. 誤った鍵・破損データの安全な拒否
3. 同一環境での再現性
4. 測定可能性とデバッグ可能性
5. 容量
6. 自然さと速度

単体テストでは LLM やネットワークを必須にしない。GPU/model を使う試験は
marker を分け、明示したときだけ実行する。

## 5. 最終目標と判定方法

研究上の目標は、100 secret characters を約 400～500 cover characters に
埋め込むことである。ただしこれは、実測 entropy と暗号化 overhead に依存する。

Phase 5 で次を測定し、容量の GO/NO-GO を判断する。

```text
required_bits = encrypted_frame_bits
available_bits = 対象文字数までの各 token entropy の合計
```

安全余裕を含めて `available_bits > required_bits` とならない場合、cover 長の延長、
secret 上限の縮小、model/prompt の変更のいずれかを選ぶ。成功率を下げて 500 文字に
見せかけることはしない。

## 6. 想定 threat model

- 攻撃者はアルゴリズム、model、tokenizer、ソースコードを知っていてよい。
- 攻撃者は master key を知らない。
- cover text の観測・改変を想定する。
- endpoint の侵害、keylogger、実行中の memory disclosure は初期 scope 外とする。
- traffic analysis と高度な steganalysis への耐性は、測定対象だが初期保証外とする。

## 7. 未確定事項

Phase 6 の前に、以下を ADR として決定する。

1. **prompt/topic の再現方法**: 任意 topic を使う場合、decoder にも同じ情報が必要。
   固定 prompt、decode 引数、公開 header のどれを採るか。
2. **AEAD と KDF の具体的 library/API**: XChaCha20-Poly1305 を第一候補とし、
   nonce 管理と subkey derivation を含めて決める。
3. **Range Coding の finite-message 規約**: length framing、termination、padding。
4. **文字数上限の UI 定義**: code points のほか grapheme clusters も表示するか。
5. **model baseline**: 小型デバッグ用と品質評価用の artifact/revision。
