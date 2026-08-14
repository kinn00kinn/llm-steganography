# ADR-005: keyed cover channel v1

- Status: accepted
- Date: 2026-08-15
- Owners: repository owner

## Context

Phase 6 established that canonical Range-Coder streams can be transported through a
mock LLM channel, but the implementation did not yet consume `K_stego`.  The threat
model assumes the model, tokenizer, source code, and protocol are public, while the
shared master key remains secret.  Therefore the symbol-to-token interval ordering
must depend on the derived steganography key without changing the LLM-assigned
probability mass of each candidate token.

The earlier low-level `CodedBits` spike also accepts only a canonical Range-Coder
stream.  Phase 7 needs to transport arbitrary encrypted payload bytes.

## Decision

### Candidate mapping

For each cover position:

1. obtain canonical float32 next-token logits from the pinned backend;
2. build the deterministic top-k integer frequency table;
3. keep each `(token_id, frequency)` pair intact;
4. compute a context-bound HMAC-SHA256 rank using `K_stego`, cover position,
   SHA-256 of the complete model context token IDs, and candidate token ID;
5. sort the token/frequency pairs by that rank, with token ID as the collision
   tie-breaker;
6. feed the permuted frequency sequence to the integer Range Coder.

Because the frequency travels with its token, the permutation changes only the
secret interval ordering.  It does not intentionally change the marginal
probability assigned to a candidate token.

The mapping version is `1`; key size is 32 bytes, matching the Phase-2 HKDF output.

### Arbitrary byte channel

`hide_bytes()` appends one termination bit to arbitrary payload bytes and uses a
mirror `RangeEncoder` to stop as soon as the complete payload prefix has settled.
`extract_bytes()` arithmetic-encodes the received cover tokens and returns the exact
requested byte prefix after enough bits have settled.

In Phase 7 the byte count is an explicit argument.  A later orchestration layer will
infer the encrypted-frame length from versioned framing so the final public decode
API does not require an out-of-band byte count.

### Control/special tokens

The model-neutral stego config can exclude explicit token IDs from the active
alphabet.  Model-specific EOS/control IDs are protocol configuration, not hard-coded
into the generic engine.  Sender and receiver must use the same exclusion set.

The engine never treats an emitted EOS token as a successful early termination;
ending before the payload settles is an error.

## Alternatives considered

### Unkeyed candidate order

Rejected.  It would leave the interval mapping fully public and would not use the
purpose-separated `K_stego` established in Phase 2.

### Permute token IDs but leave frequencies in place

Rejected.  That would assign a token another token's interval width and would alter
the intended model distribution.

### Stop when an EOS token is selected

Rejected for reliability.  A cover may end only after the payload is completely
recoverable.  EOS/control tokens must instead be excluded by versioned config when
necessary.

## Consequences

- Sender and receiver now require the same 32-byte derived steganography key.
- Wrong-key extraction should yield unrelated bytes; the higher AEAD layer remains
  responsible for authoritative wrong-key/tamper rejection.
- HMAC work is proportional to `top_k` per generated token, which is expected to be
  small compared with an LLM forward pass.
- Exact Unicode text transport remains a separate invariant.  Real-model Phase 6.5
  tests must verify `tokenize(detokenize(generated_ids)) == generated_ids`.
- `hide_bytes`/`extract_bytes` form the Phase-7 channel primitive, but secure-frame
  self-delimiting extraction and public CLI orchestration are not yet complete.
