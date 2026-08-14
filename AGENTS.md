# Repository instructions

## Project goals

- Hide a secret of at most 100 Unicode code points after NFC normalization.
- Target approximately 400-500 Japanese cover characters.
- Require exact decoding before optimizing naturalness or capacity.
- Use a shared-secret-key system with a random 256-bit master key.
- Keep the implementation local-first and deployable.
- Target an RTX 4060 Laptop GPU with 8 GB VRAM and 16 GB system RAM.

## Architecture boundaries

Keep these layers independent:

1. payload normalization, compression, framing, and encryption;
2. deterministic integer frequency construction and Range Coding;
3. tokenizer and language-model backend;
4. steganographic orchestration;
5. metrics and benchmarks.

Dependencies must point inward through explicit interfaces. Model-specific tensor
types must not leak into payload or coding modules.

## Non-negotiable rules

- Never implement custom cryptographic primitives. Use audited library APIs.
- Never reuse an AEAD nonce with the same encryption key.
- Derive purpose-specific encryption and steganography keys from the master key.
- Use integer arithmetic inside the Range Coder.
- Make probability filtering, quantization, tie-breaking, and token ordering deterministic.
- Pin model, tokenizer, prompt, runtime, and protocol revisions for reproducibility.
- Every encoder path must have a matching decoder round-trip test.
- Do not advance a phase while its exit criteria or round-trip tests fail.
- Reliability takes precedence over naturalness through Phase 8.
- Benchmark every capacity or naturalness optimization.
- Do not claim that encryption alone makes generated text undetectable.

## Development workflow

- Supported Python: exactly the 3.12 minor series; `.python-version` pins the local patch.
- Manage environments and lock files with `uv`.
- Put importable code under `src/lsteg` and tests under `tests`.
- Prefer small, typed functions and explicit immutable data structures.
- Keep protocol constants versioned; do not silently change wire behavior.
- Treat `first.md` as the source brief. Keep maintained decisions in `docs/`.
- Update documentation and tests in the same change as behavior.

## Git workflow

- Never commit or push directly to `main`.
- Start work from an up-to-date `main` and create one short-lived branch per concern.
- Use `feat/`, `fix/`, `docs/`, `chore/`, `research/`, `refactor/`, `test/`, or `ci/`
  followed by a lowercase kebab-case description. `hotfix/`, `release/`, and automated
  `dependabot/` branches are also allowed.
- Keep commits reviewable and use Conventional Commit prefixes.
- Open a pull request using the repository template and wait for the required `quality` check.
- Resolve review conversations and squash merge. Delete the branch after merge.
- Do not weaken or bypass the `main` ruleset to land a change.

Run before handing off a change:

```text
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

For an individual phase, implement only that phase and prerequisites explicitly
required by its acceptance tests. Do not download an LLM before Phase 4.
