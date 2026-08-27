"""Steganographic encoding and decoding using an LLM as the cover distribution."""

from lsteg.steg.engine import (
    CoverCapacityDiagnostics,
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
from lsteg.steg.ollama_direct import (
    OllamaDirectCapacityError,
    OllamaDirectConfig,
    OllamaDirectCover,
    OllamaDirectCoverError,
    OllamaDirectDeterminismError,
    OllamaDirectDiagnostics,
    build_ollama_frequency_table,
    channel_fingerprint,
    extract_bytes_ollama,
    hide_bytes_ollama,
    render_qwen25_raw_prompt,
    sample_ollama_cover,
    settled_next_token_candidates,
)
from lsteg.steg.repetition import filter_no_repeat_ngram_candidates
from lsteg.steg.semantic import (
    SemanticAnchorPlan,
    SemanticAnchorState,
    university_lunch_cover_plan,
    university_lunch_guarded_sparse_cover_plan,
    university_lunch_micro_cover_plan,
    university_lunch_sparse_cover_plan,
)
from lsteg.steg.tail import append_sentence_tail
from lsteg.steg.tournament_channel import (
    extract_bytes_tournament,
    hide_bytes_tournament,
)
from lsteg.steg.transport import (
    NoTransportSafeCandidatesError,
    filter_transport_safe_candidates,
)

__all__ = [
    "CANDIDATE_MAPPING_VERSION",
    "STEGANOGRAPHY_KEY_SIZE",
    "CoverCapacityDiagnostics",
    "CoverTokenError",
    "InsufficientCoverCapacityError",
    "NoTransportSafeCandidatesError",
    "OllamaDirectCapacityError",
    "OllamaDirectConfig",
    "OllamaDirectCover",
    "OllamaDirectCoverError",
    "OllamaDirectDeterminismError",
    "OllamaDirectDiagnostics",
    "SemanticAnchorPlan",
    "SemanticAnchorState",
    "SteganographyConfig",
    "append_sentence_tail",
    "build_ollama_frequency_table",
    "channel_fingerprint",
    "extract",
    "extract_bytes",
    "extract_bytes_ollama",
    "extract_bytes_tournament",
    "filter_no_repeat_ngram_candidates",
    "filter_transport_safe_candidates",
    "frequency_table_entropy",
    "hide",
    "hide_bytes",
    "hide_bytes_ollama",
    "hide_bytes_tournament",
    "keyed_candidate_permutation",
    "logits_to_frequency_table",
    "render_qwen25_raw_prompt",
    "sample_ollama_cover",
    "settled_next_token_candidates",
    "university_lunch_cover_plan",
    "university_lunch_guarded_sparse_cover_plan",
    "university_lunch_micro_cover_plan",
    "university_lunch_sparse_cover_plan",
]
