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
from lsteg.model.interface import RankedLogits

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
    temperature: float = 1.0,
    top_p: float = 1.0,
    excluded_token_ids: Sequence[int] = (),
    penalized_token_ids: Sequence[int] = (),
    presence_penalty: float = 0.0,
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
    temperature:
        Positive logit temperature applied before softmax.
    top_p:
        Nucleus cutoff applied *within the selected top-k subset*.  ``1.0``
        disables nucleus truncation.  At least two symbols are retained when
        possible so the steganographic channel does not collapse needlessly.
    excluded_token_ids:
        Token IDs that must never appear in the active alphabet (for example,
        model-specific EOS/control tokens). IDs outside the vocabulary are ignored.
    penalized_token_ids:
        Token IDs already present in the visible cover.  When ``presence_penalty``
        is positive, that constant is subtracted once from each such token's logit.
    presence_penalty:
        Deterministic presence penalty in ``[0, 2]``.  A token is penalized once
        regardless of how many times it has appeared.

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
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    top_p_value = _validate_top_p(top_p)
    penalty = _validate_presence_penalty(presence_penalty)
    excluded: set[int] = set()
    for token_id in excluded_token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("excluded token IDs must be integers")
        if token_id < 0:
            raise ValueError("excluded token IDs must not be negative")
        if token_id < len(logits):
            excluded.add(token_id)
    penalized: set[int] = set()
    for token_id in penalized_token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("penalized token IDs must be integers")
        if token_id < 0:
            raise ValueError("penalized token IDs must not be negative")
        if token_id < len(logits):
            penalized.add(token_id)
    available = len(logits) - len(excluded)
    if available < top_k:
        raise ValueError(f"available vocabulary size {available} is smaller than top_k {top_k}")

    # Select top-k after deterministic presence adjustment. Softmax is
    # monotonic, so ranking adjusted logits is equivalent to ranking the
    # adjusted probabilities before nucleus truncation.
    indexed = sorted(
        (
            (
                token_id,
                float(value) - (penalty if token_id in penalized else 0.0),
            )
            for token_id, value in enumerate(logits)
            if token_id not in excluded
        ),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]
    ranked = RankedLogits(
        tuple(token_id for token_id, _ in indexed),
        tuple(value for _, value in indexed),
    )
    return ranked_logits_to_frequency_table(
        ranked,
        total=total,
        temperature=temperature_value,
        top_p=top_p_value,
    )


def ranked_logits_to_frequency_table(
    ranked: RankedLogits,
    *,
    total: int = FREQUENCY_TOTAL,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[list[int], FrequencyTable]:
    """Quantize an already-ranked top-k subset without full-vocabulary softmax.

    ``logits_to_frequency_table`` selects top-k by raw logit and then
    renormalizes probabilities *within that selected subset*.  Therefore the
    full-vocabulary softmax denominator cancels exactly: the same integer table
    can be constructed from only the selected logits.
    """
    candidate_count = len(ranked.token_ids)
    if not (1 <= candidate_count <= total):
        raise ValueError(f"top_k must satisfy 1 <= top_k <= total, got {candidate_count}, {total}")
    if total > MAX_FREQUENCY_TOTAL:
        raise ValueError(f"total must be <= {MAX_FREQUENCY_TOTAL}, got {total}")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    top_p_value = _validate_top_p(top_p)

    scaled = [value / temperature_value for value in ranked.values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]

    keep = candidate_count
    if top_p_value < 1.0 and candidate_count > 1:
        cumulative = 0.0
        keep = 0
        for probability in probabilities:
            cumulative += probability
            keep += 1
            if cumulative >= top_p_value:
                break
        keep = max(2, keep)

    token_ids = list(ranked.token_ids[:keep])
    selected_weights = weights[:keep]
    selected_sum = sum(selected_weights)
    selected_probabilities = [weight / selected_sum for weight in selected_weights]

    remaining = total - keep
    raw_alloc = [probability * remaining for probability in selected_probabilities]
    floor_alloc = [int(value) for value in raw_alloc]
    remainders = [raw - floor for raw, floor in zip(raw_alloc, floor_alloc, strict=True)]
    extra = remaining - sum(floor_alloc)
    extra_order = sorted(range(keep), key=lambda index: (-remainders[index], index))
    frequencies = [1 + value for value in floor_alloc]
    for index in extra_order[:extra]:
        frequencies[index] += 1
    assert sum(frequencies) == total, "frequency allocation invariant violated"
    return token_ids, FrequencyTable(frequencies)


def _validate_top_p(top_p: float) -> float:
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be a number")
    value = float(top_p)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")
    return value


def _validate_presence_penalty(presence_penalty: float) -> float:
    if isinstance(presence_penalty, bool) or not isinstance(presence_penalty, (int, float)):
        raise TypeError("presence_penalty must be a number")
    value = float(presence_penalty)
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise ValueError("presence_penalty must satisfy 0 <= presence_penalty <= 2")
    return value


def frequency_table_entropy(table: FrequencyTable) -> float:
    """Return Shannon entropy in bits of an integer frequency table."""
    total = table.total
    entropy = 0.0
    for frequency in table.frequencies:
        if frequency == 0:
            continue
        probability = frequency / total
        entropy -= probability * math.log2(probability)
    return entropy
