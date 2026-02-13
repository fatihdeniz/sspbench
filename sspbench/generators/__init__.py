"""
Benchmark Generators

This module contains classes and functions for generating dynamic benchmarks
for safety, security, and privacy evaluation.

Sub-modules
-----------
base            – abstract BaseGenerator class
privacy         – PrivacyGenerator
variations      – factuality-engine question variation utilities
safety_mutations – safety prompt source-mining & mutation operators (new)
"""

from .base import BaseGenerator
from .privacy import PrivacyGenerator

# Safety mutations — these use lazy imports internally so they are safe
# to import without vllm being available.
from .safety_mutations import (
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

# Variations module requires llm_utils (which needs vllm) — import lazily.
# Use: ``from sspbench.generators.variations import apply_variations_to_dataset``


def __getattr__(name):
    """Lazy attribute access for modules that require heavy deps (vllm)."""
    if name == "apply_variations_to_dataset":
        from .variations import apply_variations_to_dataset
        return apply_variations_to_dataset
    if name in ("SafetyGenerator", "SecurityGenerator"):
        raise ImportError(
            f"{name} is referenced but not yet implemented. "
            f"Use sspbench.generators.safety_mutations instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseGenerator",
    "PrivacyGenerator",
    # Safety mutation & source-mining API
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