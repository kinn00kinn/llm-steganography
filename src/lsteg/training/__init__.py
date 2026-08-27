"""Utilities for Japanese-prose LoRA research.

The training package deliberately contains only model-neutral recipe/data logic.
Heavy optional dependencies (Transformers/PEFT/bitsandbytes) live in scripts and
are imported lazily so the normal steganography test suite stays lightweight.
"""

from lsteg.training.artifacts import hash_file, hash_tree
from lsteg.training.config import TrainingRecipe
from lsteg.training.data import (
    TEACHER_DATA_SCHEMA_VERSION,
    TeacherExample,
    TeacherValidationPolicy,
    audit_teacher_examples,
    load_teacher_examples,
    validate_teacher_completion,
)
from lsteg.training.distribution import (
    channel_entropy_bits,
    forward_kl_nats,
    full_entropy_nats,
    ratio_within_bounds,
)
from lsteg.training.ollama import parse_model_identity
from lsteg.training.scenarios import Scenario, build_scenarios
from lsteg.training.schedule import should_optimizer_step
from lsteg.training.selection import select_kl_rows
from lsteg.training.tokenization import build_completion_training_ids

__all__ = [
    "TEACHER_DATA_SCHEMA_VERSION",
    "Scenario",
    "TeacherExample",
    "TeacherValidationPolicy",
    "TrainingRecipe",
    "audit_teacher_examples",
    "build_completion_training_ids",
    "build_scenarios",
    "channel_entropy_bits",
    "forward_kl_nats",
    "full_entropy_nats",
    "hash_file",
    "hash_tree",
    "load_teacher_examples",
    "parse_model_identity",
    "ratio_within_bounds",
    "select_kl_rows",
    "should_optimizer_step",
    "validate_teacher_completion",
]
