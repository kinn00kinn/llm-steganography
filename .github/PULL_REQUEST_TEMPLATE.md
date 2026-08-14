## Purpose

<!-- What problem, issue, or research question does this PR address? -->

## Changes

<!-- Keep this to one concern. -->

## Verification

- [ ] `uv run --frozen pytest`
- [ ] `uv run --frozen ruff check .`
- [ ] `uv run --frozen ruff format --check .`
- [ ] `uv run --frozen mypy src tests`
- [ ] Encoder changes include decoder round-trip tests, or not applicable
- [ ] Phase completion updates the static Pages result artifact, or not applicable

## Protocol / security / reproducibility impact

<!-- State none, or describe versioning, key handling, determinism, and migration effects. -->

## Artifacts

- [ ] No key, secret, `.env`, model weight, or sensitive experiment record is included
- [ ] Documentation and experiment manifest/schema are updated where needed
