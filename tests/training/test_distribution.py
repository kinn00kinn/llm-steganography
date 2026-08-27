from __future__ import annotations

import pytest

from lsteg.training.distribution import channel_entropy_bits, ratio_within_bounds


def test_ratio_within_bounds_is_two_sided() -> None:
    assert ratio_within_bounds(1.0, minimum=0.75, maximum=1.35)
    assert not ratio_within_bounds(0.5, minimum=0.75, maximum=1.35)
    assert not ratio_within_bounds(2.0, minimum=0.75, maximum=1.35)


def test_ratio_within_bounds_rejects_bad_corridor() -> None:
    with pytest.raises(ValueError):
        ratio_within_bounds(1.0, minimum=0.0, maximum=1.0)
    with pytest.raises(ValueError):
        ratio_within_bounds(1.0, minimum=1.2, maximum=1.1)


def test_channel_entropy_matches_nucleus_shape() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    entropy = channel_entropy_bits(
        torch,
        logits,
        temperature=1.0,
        top_k=4,
        top_p=0.70,
    )
    # top-p must retain at least two symbols. Entropy is therefore positive
    # and below the 2-bit entropy of a uniform four-symbol alphabet.
    assert entropy.shape == (1,)
    assert 0.0 < float(entropy[0]) < 2.0


def test_channel_entropy_rejects_invalid_parameters() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4))
    with pytest.raises(ValueError):
        channel_entropy_bits(torch, logits, temperature=0.0, top_k=4, top_p=1.0)
    with pytest.raises(ValueError):
        channel_entropy_bits(torch, logits, temperature=1.0, top_k=0, top_p=1.0)
    with pytest.raises(ValueError):
        channel_entropy_bits(torch, logits, temperature=1.0, top_k=4, top_p=0.0)
