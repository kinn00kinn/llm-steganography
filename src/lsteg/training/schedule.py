"""Model-neutral helpers for deterministic training-step scheduling."""

from __future__ import annotations


def should_optimizer_step(
    accumulated_examples: int,
    gradient_accumulation_steps: int,
    *,
    is_last_example: bool,
) -> bool:
    """Return whether the current accumulated gradients should be applied.

    The accumulator is intentionally epoch-local.  If the final partial batch is
    stepped at an epoch boundary, the next epoch must start from zero rather than
    inheriting the previous global example modulo.
    """
    if accumulated_examples < 1:
        raise ValueError("accumulated_examples must be positive")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    return accumulated_examples >= gradient_accumulation_steps or is_last_example
