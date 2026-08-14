"""Text payload normalization, compression, and framing."""

from lsteg.payload.codec import (
    MAX_SECRET_CODE_POINTS,
    MAX_SECRET_UTF8_BYTES,
    EncodedTextPayload,
    PayloadMetrics,
    decode_text_payload,
    encode_text_payload,
    normalize_secret,
)
from lsteg.payload.errors import (
    InvalidSecretTextError,
    MalformedPayloadError,
    PayloadCodecError,
    SecretTooLongError,
    UnsupportedPayloadVersionError,
)
from lsteg.payload.framing import TEXT_FRAME_VERSION, CompressionMethod

__all__ = [
    "MAX_SECRET_CODE_POINTS",
    "MAX_SECRET_UTF8_BYTES",
    "TEXT_FRAME_VERSION",
    "CompressionMethod",
    "EncodedTextPayload",
    "InvalidSecretTextError",
    "MalformedPayloadError",
    "PayloadCodecError",
    "PayloadMetrics",
    "SecretTooLongError",
    "UnsupportedPayloadVersionError",
    "decode_text_payload",
    "encode_text_payload",
    "normalize_secret",
]
