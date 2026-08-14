"""Deterministic conversion of LLM logits to a FrequencyTable.

Design constraints
------------------
* Pure Python + PyTorch; no model-specific types leak out.
* Deterministic and reproducible: given identical float32 logits, this function
  always produces an identical FrequencyTable regardless of process or device.
* Uses only integer arithmetic after the initial softmax, so the Range Coder
  invariants are preserved.
* Token ordering is deterministic: top-k tokens are ordered by descending
  probability, with ties broken by ascending token ID.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from lsteg.coding.frequencies import MAX_FREQUENCY_TOTAL, FrequencyTable

# The total frequency budget must fit within FrequencyTable.MAX_FREQUENCY_TOTAL.
# We use 2^15 = 32768 to leave headroom for the mandatory minimum-1 allocation.
FREQUENCY_TOTAL: int = MAX_FREQUENCY_TOTAL  # 32768

# Number of tokens kept in the active alphabet.  This is the steganographic
# alphabet size: larger = more capacity per token but rarer tokens appear.
# 256 is a reasonable default; the engine config can override it.
DEFAULT_TOP_K: int = 256


def logits_to_frequency_table(
    logits: Sequence[float],
    *,
    top_k: int = DEFAULT_TOP_K,
    total: int = FREQUENCY_TOTAL,
) -> tuple[list[int], FrequencyTable]:
    """Convert raw logits into a (token_ids, FrequencyTable) pair.

    Parameters
    ----------
    logits:
        Raw float32 logit vector; length must equal the vocabulary size.
    top_k:
        Number of highest-probability tokens to keep in the active alphabet.
        Must satisfy 1 <= top_k <= total.
    total:
        Integer frequency budget.  Must equal MAX_FREQUENCY_TOTAL or less.

    Returns
    -------
    token_ids:
        The top-k token IDs in the order they appear in the FrequencyTable
        (descending probability, ascending ID on ties).
    table:
        FrequencyTable of length top_k whose cumulative total equals `total`.
    """
    if not (1 <= top_k <= total):
        raise ValueError(f"top_k must satisfy 1 <= top_k <= total, got {top_k}, {total}")
    if total > MAX_FREQUENCY_TOTAL:
        raise ValueError(f"total must be <= {MAX_FREQUENCY_TOTAL}, got {total}")
    if len(logits) < top_k:
        raise ValueError(f"vocabulary size {len(logits)} is smaller than top_k {top_k}")

    # --- softmax in float64 for numerical stability ---------------------------
    # We deliberately avoid torch here so this module has no GPU dependency at
    # runtime; callers already computed logits via the backend.
    max_logit = max(logits)
    exp_values = [math.exp(v - max_logit) for v in logits]
    exp_sum = sum(exp_values)
    probs = [v / exp_sum for v in exp_values]

    # --- select top-k by descending probability, ascending token ID -----------
    indexed = sorted(
        enumerate(probs),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    top_ids = [idx for idx, _ in indexed]
    top_probs = [p for _, p in indexed]

    # Re-normalize among the selected tokens
    top_sum = sum(top_probs)
    top_probs_norm = [p / top_sum for p in top_probs]

    # --- integer allocation (largest-remainder method) ------------------------
    # Give each token at least 1 frequency unit, then distribute the remainder
    # proportionally using the largest-remainder (Hamilton) method.
    remaining = total - top_k  # budget after mandatory minimums
    raw_alloc = [p * remaining for p in top_probs_norm]
    floor_alloc = [int(f) for f in raw_alloc]
    remainders = [raw - floor for raw, floor in zip(raw_alloc, floor_alloc, strict=True)]

    # How many extra units to distribute?
    distributed = sum(floor_alloc)
    extra = remaining - distributed  # guaranteed >= 0 due to floor

    # Rank by remainder descending, breaking ties by index (ascending)
    extra_order = sorted(range(top_k), key=lambda i: (-remainders[i], i))

    frequencies = [1 + f for f in floor_alloc]
    for i in extra_order[:extra]:
        frequencies[i] += 1

    assert sum(frequencies) == total, "frequency allocation invariant violated"

    return top_ids, FrequencyTable(frequencies)
