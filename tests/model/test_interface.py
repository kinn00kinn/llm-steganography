from __future__ import annotations

import pytest

from lsteg.model import Logits, RuntimeFingerprint


def test_logits_are_canonical_float32_with_stable_digest() -> None:
    logits = Logits.from_values([0.1, 4, -2.5])

    assert len(logits) == 3
    assert logits[1] == 4.0
    assert logits[:] == pytest.approx((0.1, 4.0, -2.5))
    assert logits.argmax_token_id == 1
    assert logits.sha256 == "8e63e56b5589c0b0f8d7d0b8fcab48d6469d5300d16b99eaacb539bc5b9e7893"


def test_argmax_breaks_exact_ties_by_lowest_token_id() -> None:
    assert Logits.from_values([1.0, 2.0, 2.0]).argmax_token_id == 1


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [1e100]])
def test_logits_reject_empty_or_non_float32_values(values: list[float]) -> None:
    with pytest.raises(ValueError):
        Logits.from_values(values)


def test_runtime_fingerprint_is_json_compatible() -> None:
    fingerprint = RuntimeFingerprint(
        python_version="3.12.10",
        platform="Windows-11-AMD64",
        torch_version="2.13.0+cu130",
        transformers_version="5.15.0",
        cuda_version="13.0",
        device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        compute_capability="8.9",
    )

    assert fingerprint.as_dict()["compute_capability"] == "8.9"
