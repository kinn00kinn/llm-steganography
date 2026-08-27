"""Deterministic selection of sparse KL-anchor positions."""

from __future__ import annotations

import hashlib
import random


def select_kl_rows(
    completion_token_count: int,
    requested: int,
    *,
    seed: int,
    example_key: str,
) -> tuple[int, ...]:
    """Select row indices inside completion logits without global RNG state."""
    if completion_token_count < 1:
        raise ValueError("completion_token_count must be positive")
    if requested < 0:
        raise ValueError("requested must be non-negative")
    count = min(completion_token_count, requested)
    if count == 0:
        return ()
    digest = hashlib.sha256(f"{seed}:{example_key}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(local_seed)
    return tuple(sorted(rng.sample(range(completion_token_count), count)))
