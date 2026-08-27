"""Distribution diagnostics for LoRA training/evaluation.

These helpers intentionally evaluate the *cover-channel* head distribution used
by quality-first steganography rather than only the full multilingual vocabulary.
Heavy tensor libraries are passed in by the caller so importing ``lsteg.training``
remains lightweight for the normal test suite.
"""

from __future__ import annotations

import math
from typing import Any


def full_entropy_nats(torch: Any, logits: Any) -> Any:
    """Return per-row Shannon entropy (nats) over the full vocabulary."""
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def forward_kl_nats(torch: Any, base_logits: Any, adapted_logits: Any) -> Any:
    """Return per-row ``KL(P_base || P_adapted)`` over the full vocabulary."""
    base_log_probs = torch.nn.functional.log_softmax(base_logits.float(), dim=-1)
    adapted_log_probs = torch.nn.functional.log_softmax(adapted_logits.float(), dim=-1)
    return (base_log_probs.exp() * (base_log_probs - adapted_log_probs)).sum(dim=-1)


def channel_entropy_bits(
    torch: Any,
    logits: Any,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Any:
    """Return per-row entropy after the same top-k/top-p shape used by the cover channel.

    This is a floating-point diagnostic, not the integer Range-Coder table.  It
    intentionally ignores transport-safe filtering because evaluation does not
    have a visible cover prefix at each teacher-forced row.  Base and adapted
    models are compared with the same transformation, making the ratio useful
    for detecting both entropy collapse and entropy explosion.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [rows, vocabulary]")

    keep_k = min(int(top_k), int(logits.shape[-1]))
    ranked, _ = torch.topk(logits.float() / float(temperature), keep_k, dim=-1)
    probabilities = torch.softmax(ranked, dim=-1)
    if top_p < 1.0 and keep_k > 1:
        cumulative = torch.cumsum(probabilities, dim=-1)
        # Keep the crossing token, matching ranked_logits_to_frequency_table.
        keep_counts = (cumulative < float(top_p)).sum(dim=-1) + 1
        keep_counts = torch.clamp(keep_counts, min=min(2, keep_k), max=keep_k)
        positions = torch.arange(keep_k, device=logits.device).unsqueeze(0)
        mask = positions < keep_counts.unsqueeze(1)
        probabilities = probabilities * mask
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    safe = torch.where(probabilities > 0, probabilities, torch.ones_like(probabilities))
    terms = torch.where(
        probabilities > 0,
        probabilities * torch.log2(safe),
        torch.zeros_like(probabilities),
    )
    return -terms.sum(dim=-1)


def ratio_within_bounds(value: float, *, minimum: float, maximum: float) -> bool:
    """Return whether a finite ratio lies inside an inclusive positive corridor."""
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("ratio bounds must be finite")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("ratio bounds must satisfy 0 < minimum <= maximum")
    return math.isfinite(value) and minimum <= value <= maximum
