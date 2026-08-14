from __future__ import annotations

import pytest

from lsteg.coding import CodedBits, FrequencyTable, RangeDecoder, RangeEncoder


def test_incremental_encoder_decoder_round_trip() -> None:
    table = FrequencyTable([40, 30, 20, 10])
    symbols = [0, 1, 0, 2, 3, 0, 0]
    encoder = RangeEncoder()

    for symbol in symbols:
        encoder.encode(table, symbol)
    coded = encoder.finish()

    assert coded == CodedBits(bytes.fromhex("33c8"), 13)

    decoder = RangeDecoder(coded)
    assert [decoder.decode(table) for _ in symbols] == symbols


def test_underflow_renormalization_is_symmetric() -> None:
    table = FrequencyTable([1, 2, 1])
    symbols = [1, 0, 2, 1] * 100
    encoder = RangeEncoder()
    for symbol in symbols:
        encoder.encode(table, symbol)
    coded = encoder.finish()

    decoder = RangeDecoder(coded)
    assert [decoder.decode(table) for _ in symbols] == symbols


def test_single_symbol_alphabet_round_trip() -> None:
    table = FrequencyTable([1])
    encoder = RangeEncoder()
    for _ in range(1_000):
        encoder.encode(table, 0)
    coded = encoder.finish()

    decoder = RangeDecoder(coded)
    assert [decoder.decode(table) for _ in range(1_000)] == [0] * 1_000


def test_encoder_cannot_be_reused_after_finish() -> None:
    encoder = RangeEncoder()
    encoder.encode(FrequencyTable([1, 1]), 0)
    encoder.finish()

    with pytest.raises(RuntimeError, match="finished"):
        encoder.finish()
    with pytest.raises(RuntimeError, match="finished"):
        encoder.encode(FrequencyTable([1, 1]), 1)


def test_settled_bit_snapshot_does_not_finish_encoder() -> None:
    encoder = RangeEncoder()
    table = FrequencyTable.uniform(2)
    encoder.encode(table, 1)

    assert encoder.settled_bit_length == 1
    assert encoder.settled_bits() == CodedBits(b"\x80", 1)

    encoder.encode(table, 0)
    assert encoder.settled_bits() == CodedBits(b"\x80", 2)
