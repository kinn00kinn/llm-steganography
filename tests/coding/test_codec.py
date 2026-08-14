from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from lsteg.coding import (
    FrequencyTable,
    InsufficientRangeDataError,
    MalformedRangeFrameError,
    UnavailableSymbolError,
    decode_bytes,
    decode_symbols,
    encode_bytes,
    encode_symbols,
    map_bytes_to_symbols,
    parse_range_frame,
    recover_bytes_from_symbols,
)
from lsteg.payload import (
    decode_secure_text_payload,
    encode_secure_text_payload,
    generate_master_key,
)


@pytest.mark.parametrize("size", [0, 1, 10, 100, 1_024, 10_240])
def test_boundary_sized_byte_round_trips(size: int) -> None:
    source = random.Random(size)
    payload = source.randbytes(size)

    frame = encode_bytes(payload)

    assert decode_bytes(frame) == payload
    assert parse_range_frame(frame).symbol_count == size


def test_all_byte_values_round_trip() -> None:
    payload = bytes(range(256)) * 4

    assert decode_bytes(encode_bytes(payload)) == payload


def test_encoding_is_deterministic() -> None:
    table = FrequencyTable([40, 30, 20, 10])
    symbols = (0, 1, 0, 2, 3, 0, 0)

    expected = bytes.fromhex("4c524e470101000000070000000d33c8")
    assert encode_symbols(symbols, table) == expected
    assert encode_symbols(symbols, table) == expected


def test_skewed_integer_frequencies_round_trip() -> None:
    frequencies = [1] * 256
    frequencies[0] = 10_000
    frequencies[32] = 5_000
    table = FrequencyTable(frequencies)
    payload = (b"\x00 " * 1_000) + bytes(range(256))

    frame = encode_bytes(payload, table)

    assert decode_bytes(frame, table) == payload
    assert parse_range_frame(frame).coded.bit_length < len(payload) * 8


def test_context_dependent_frequency_provider_round_trip() -> None:
    tables = (FrequencyTable([7, 2, 1]), FrequencyTable([1, 3, 6]))
    symbols = tuple(index % 3 for index in range(300))

    def provider(index: int, prefix: Sequence[int]) -> FrequencyTable:
        assert index == len(prefix)
        previous = prefix[-1] if prefix else 0
        return tables[(index + previous) % len(tables)]

    frame = encode_symbols(symbols, provider)

    assert decode_symbols(frame, provider) == symbols


def test_two_thousand_seeded_random_round_trips() -> None:
    source = random.Random(20260814)

    for _ in range(2_000):
        payload = source.randbytes(source.randint(0, 256))
        assert decode_bytes(encode_bytes(payload)) == payload


def test_secure_payload_range_coder_milestone() -> None:
    key = generate_master_key()
    encrypted = encode_secure_text_payload("Phase 3の完全復元", key).frame

    coded = encode_bytes(encrypted)
    recovered_encrypted = decode_bytes(coded)

    assert recovered_encrypted == encrypted
    assert decode_secure_text_payload(recovered_encrypted, key) == "Phase 3の完全復元"


@pytest.mark.parametrize("size", [1, 10, 100, 1_024, 10_240])
def test_payload_bits_map_to_symbols_and_back(size: int) -> None:
    payload = random.Random(size + 100).randbytes(size)
    table = FrequencyTable.uniform(256)

    symbols = map_bytes_to_symbols(payload, table)

    assert recover_bytes_from_symbols(symbols, size, table) == payload


def test_every_single_byte_value_maps_to_symbols_and_back() -> None:
    table = FrequencyTable([40, 30, 20, 10])

    for value in range(256):
        payload = bytes((value,))
        symbols = map_bytes_to_symbols(payload, table)
        assert recover_bytes_from_symbols(symbols, 1, table) == payload


def test_empty_payload_uses_no_symbols() -> None:
    table = FrequencyTable.uniform(2)

    assert map_bytes_to_symbols(b"", table) == ()
    assert recover_bytes_from_symbols((), 0, table) == b""


def test_dynamic_tables_map_payload_bits_to_symbols_and_back() -> None:
    payload = bytes.fromhex("4c5345430101001a002a")
    tables = (FrequencyTable([5, 3, 2]), FrequencyTable([1, 4, 5]))

    def provider(index: int, prefix: Sequence[int]) -> FrequencyTable:
        previous = prefix[-1] if prefix else 0
        return tables[(index + previous) % 2]

    symbols = map_bytes_to_symbols(payload, provider)

    assert recover_bytes_from_symbols(symbols, len(payload), provider) == payload


def test_secure_payload_maps_through_symbol_channel() -> None:
    key = generate_master_key()
    encrypted = encode_secure_text_payload("payloadからsymbolへ", key).frame
    table = FrequencyTable([40, 30, 20, 10])

    symbols = map_bytes_to_symbols(encrypted, table)
    recovered = recover_bytes_from_symbols(symbols, len(encrypted), table)

    assert recovered == encrypted
    assert decode_secure_text_payload(recovered, key) == "payloadからsymbolへ"


def test_mapping_rejects_channel_without_capacity() -> None:
    with pytest.raises(InsufficientRangeDataError, match="cannot carry"):
        map_bytes_to_symbols(b"payload", FrequencyTable([1]))


def test_mapping_respects_symbol_budget() -> None:
    with pytest.raises(InsufficientRangeDataError, match="within 1 symbols"):
        map_bytes_to_symbols(b"payload", FrequencyTable.uniform(2), max_symbols=1)


def test_recovery_rejects_too_short_symbol_stream() -> None:
    with pytest.raises(InsufficientRangeDataError, match="required bits"):
        recover_bytes_from_symbols([0], 10, FrequencyTable.uniform(2))


def test_zero_frequency_symbol_is_rejected() -> None:
    with pytest.raises(UnavailableSymbolError, match="zero"):
        encode_symbols([1], FrequencyTable([1, 0]))


def test_symbols_must_be_integers() -> None:
    with pytest.raises(TypeError, match="integers"):
        encode_symbols(["0"], FrequencyTable([1]))  # type: ignore[list-item]


def test_frequency_provider_must_return_a_table() -> None:
    def invalid_provider(index: int, prefix: Sequence[int]) -> object:
        del index, prefix
        return [1, 1]

    with pytest.raises(TypeError, match="return FrequencyTable"):
        encode_symbols([0], invalid_provider)  # type: ignore[arg-type]


def test_byte_codec_rejects_non_byte_symbols() -> None:
    frame = encode_symbols([256], FrequencyTable.uniform(257))

    with pytest.raises(MalformedRangeFrameError, match="byte alphabet"):
        decode_bytes(frame, FrequencyTable.uniform(257))


def test_byte_encoder_requires_bytes() -> None:
    with pytest.raises(TypeError, match="bytes"):
        encode_bytes(bytearray(b"data"))  # type: ignore[arg-type]
