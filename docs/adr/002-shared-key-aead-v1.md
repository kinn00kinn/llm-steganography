# ADR-002: shared-key and AEAD envelope v1

- Status: accepted
- Date: 2026-08-14
- Owners: repository owner

## Context

Phase 1のtext frameはNFC、圧縮、lengthを検証できるが、秘密文の機密性と改ざん検知を
提供しない。Phase 2では、32-byte master keyだけを事前共有し、inner text frameを
認証付き暗号化する。暗号primitiveは自作せず、wrong keyと改ざんを公開errorから区別しない。

## Decision

### Library and primitive

- AEAD: PyNaCl `Aead` / libsodium XChaCha20-Poly1305-IETF
- KDF: cryptography `HKDFExpand` with SHA-256
- master key: libsodium CSPRNGで生成する32 bytes
- nonce: 暗号化ごとにlibsodium CSPRNGで生成する24 bytes
- authentication tag: 16 bytes

PyNaClの`Aead`は24-byte nonceを持つXChaCha20-Poly1305を使用する。libsodiumは
XChaCha20-Poly1305でrandom nonceを使用可能としている。

Sources:

- <https://pynacl.readthedocs.io/en/latest/secret/>
- <https://doc.libsodium.org/secret-key_cryptography/aead/chacha20-poly1305/xchacha20-poly1305_construction>
- <https://cryptography.io/en/stable/hazmat/primitives/key-derivation-functions/#hkdf>

### Key separation

master keyは一様ランダムな256-bit値なので、HKDF-ExpandのPRKとして使用する。用途ごとに
独立した`info`を使用し、32-byte subkeyを導出する。

```text
K_encrypt = HKDF-Expand-SHA256(K_master, 32,
            "llm-steganography/v1/encryption")
K_stego   = HKDF-Expand-SHA256(K_master, 32,
            "llm-steganography/v1/steganography")
```

context文字列と出力長はprotocolの一部であり、互換性を壊さず変更しない。

### Master-key file v1

key fileは40-byte binary formatとする。

```text
offset  bytes  field
0       4      magic = "LSTK"
4       1      version = 1
5       3      reserved = zero
8       32     master key
```

`steg keygen`は同じdirectoryにmode 0600のtemporary fileを完全に書き、hard linkで最終pathへ
installする。link作成は既存targetを置換しないため、atomic visibilityとno-clobberを同時に
満たす。hard linkを安全に作れないfilesystemでは失敗し、非atomicなfallbackは行わない。
WindowsのACLは`chmod`だけで完全に表現できないため、配置directory自体も利用者が保護する。

### Secure envelope v1

```text
offset  bytes  field
0       4      magic = "LSEC"
4       1      version = 1
5       1      algorithm = 1 (XChaCha20-Poly1305-IETF)
6       2      plaintext inner-frame length, unsigned big-endian
8       2      ciphertext + tag length, unsigned big-endian
10      24     nonce
34      N      ciphertext followed by 16-byte authentication tag
```

10-byte header全体をAEAD additional authenticated dataとする。nonce変更はAEAD検証を失敗させる。
compression methodとdecoded lengthは暗号化されたPhase 1 inner frameに保持する。100文字上限で
inner frameは最大410 bytes、secure envelopeは最大460 bytes、固定overheadは50 bytesとなる。

parseは復号前にmagic、version、algorithm、length、上限を検証する。AEAD検証後にのみinner
frameをdecodeする。wrong key、nonce/ciphertext/tag改ざんはすべて
`AuthenticationError("secure payload authentication failed")`へ変換する。

## Alternatives considered

### ChaCha20-Poly1305 with 96-bit random nonce

cryptographyだけで実装できるが、永続keyでrandom nonceを使う今回のlocal-first用途では、
192-bit nonceを持つXChaCha20-Poly1305の方がnonce衝突riskを小さくできるため不採用。

### Counter nonce

再起動、backup rollback、複数process間でcounter stateをatomicに共有する必要がある。Phase 2の
単一message envelopeには24-byte random nonceの方が状態を持たず安全に運用できるため不採用。

### Master keyをAEADへ直接渡す

将来のcandidate mappingと暗号化で同じkey materialを再利用するため不採用。最初から用途別
contextでsubkeyを分離する。

### Key bytesをhex/base64で標準出力する

shell history、CI log、terminal captureへ残りやすいため不採用。`keygen`は保存先だけを表示する。

## Consequences

- encrypted payloadは同じsecret/keyでも毎回異なる。
- secure frameの50-byte overheadをPhase 5のcapacity計算に含める必要がある。
- PyNaClとcryptographyがruntime dependencyになる。
- key fileはportableだが、OS account、directory ACL、backupの保護は利用者の責任となる。
- AEADは機密性と完全性を提供するが、cover textのsteganalysis耐性は保証しない。

## Compatibility/security impact

version、algorithm、KDF context、header layoutの変更は新しいprotocol versionを必要とする。
nonce再利用は禁止する。APIはcaller指定nonceを公開せず、暗号化ごとにCSPRNGから生成する。
