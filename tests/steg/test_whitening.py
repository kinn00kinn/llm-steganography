from lsteg.steg.whitening import xor_whiten


def test_whitening_round_trip() -> None:
    key = bytes(range(32))
    data = bytes(range(255))
    whitened = xor_whiten(data, key)
    assert whitened != data
    assert xor_whiten(whitened, key) == data


def test_whitening_is_deterministic() -> None:
    key = b"k" * 32
    assert xor_whiten(b"payload", key) == xor_whiten(b"payload", key)
