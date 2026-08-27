from __future__ import annotations

import pytest

from lsteg.training.schedule import should_optimizer_step


def test_gradient_accumulation_steps_on_full_and_partial_batches() -> None:
    assert not should_optimizer_step(1, 8, is_last_example=False)
    assert should_optimizer_step(8, 8, is_last_example=False)
    assert should_optimizer_step(2, 8, is_last_example=True)


def test_epoch_local_accumulator_can_restart_after_partial_step() -> None:
    # Regression: a global-example modulo caused the next epoch to step after
    # only six examples when the prior epoch ended with a 2/8 partial batch.
    assert should_optimizer_step(2, 8, is_last_example=True)
    assert not should_optimizer_step(1, 8, is_last_example=False)
    assert not should_optimizer_step(6, 8, is_last_example=False)
    assert should_optimizer_step(8, 8, is_last_example=False)


@pytest.mark.parametrize("accumulated,steps", [(0, 8), (1, 0)])
def test_gradient_accumulation_rejects_invalid_counts(accumulated: int, steps: int) -> None:
    with pytest.raises(ValueError):
        should_optimizer_step(accumulated, steps, is_last_example=False)
