"""
Benchmark Generators

This module contains classes and functions for generating dynamic benchmarks
for safety, security, and privacy evaluation.

Sub-modules
-----------
base       – abstract BaseGenerator class
privacy    – PrivacyGenerator
variations – factuality-engine question variation utilities

Safety mutation operators have moved to ``sspbench.safety.safety_mutations``.
Re-exports below are kept for backwards compatibility.
"""

from .base import BaseGenerator
from .privacy import PrivacyGenerator

from ..safety.safety_mutations import (
    mine_source_prompts,
    build_source_context,
    apply_safety_mutations,
    generate_grounded_safety_prompts,
    mutate_persona_injection,
    mutate_hypothetical_frame,
    mutate_entailment_shift,
    mutate_topic_transplant,
    mutate_escalation_ladder,
    mutate_compositional_blend,
    MUTATION_OPERATORS,
)


def __getattr__(name):
    """Lazy attribute access for modules that require heavy deps (vllm)."""
    if name == "apply_variations_to_dataset":
        from .variations import apply_variations_to_dataset
        return apply_variations_to_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseGenerator",
    "PrivacyGenerator",
    # Re-exported from sspbench.safety.safety_mutations
    "mine_source_prompts",
    "build_source_context",
    "apply_safety_mutations",
    "generate_grounded_safety_prompts",
    "mutate_persona_injection",
    "mutate_hypothetical_frame",
    "mutate_entailment_shift",
    "mutate_topic_transplant",
    "mutate_escalation_ladder",
    "mutate_compositional_blend",
    "MUTATION_OPERATORS",
]