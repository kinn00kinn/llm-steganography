"""Steganographic encoding and decoding using an LLM as the cover distribution."""

from lsteg.steg.engine import (
    CoverTokenError,
    InsufficientCoverCapacityError,
    SteganographyConfig,
    extract,
    extract_bytes,
    hide,
    hide_bytes,
)
from lsteg.steg.frequencies import frequency_table_entropy, logits_to_frequency_table
from lsteg.steg.mapping import (
    CANDIDATE_MAPPING_VERSION,
    STEGANOGRAPHY_KEY_SIZE,
    keyed_candidate_permutation,
)

__all__ = [
    "CANDIDATE_MAPPING_VERSION",
    "STEGANOGRAPHY_KEY_SIZE",
    "CoverTokenError",
    "InsufficientCoverCapacityError",
    "SteganographyConfig",
    "extract",
    "extract_bytes",
    "hide",
    "hide_bytes",
    "frequency_table_entropy",
    "keyed_candidate_permutation",
    "logits_to_frequency_table",
]
