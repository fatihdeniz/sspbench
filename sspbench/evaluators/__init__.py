"""
Benchmark Evaluators

This module contains classes and functions for evaluating models on generated benchmarks.

Hierarchy
---------
BaseEvaluator           (abstract root)
  └─ LLMEvaluator       (adds LLM judge helpers)
       ├─ ScopeEvaluator
       ├─ SalienceEvaluator
       ├─ DirectFaithfulnessEvaluator
       ├─ SafetyEvaluator
       ├─ SecurityEvaluator
       └─ PrivacyEvaluator
  └─ RagasFaithfulnessEvaluator  (uses RAGAS toolkit, not a single LLM)
"""

from .base import BaseEvaluator
from .llm_evaluator import LLMEvaluator
from .safety import SafetyEvaluator
from .security import SecurityEvaluator
from .privacy import PrivacyEvaluator
from .faithfulness import DirectFaithfulnessEvaluator, RagasFaithfulnessEvaluator
from .scope import ScopeEvaluator
from .salience import SalienceEvaluator

__all__ = [
	"BaseEvaluator",
	"LLMEvaluator",
	"SafetyEvaluator",
	"SecurityEvaluator",
	"PrivacyEvaluator",
	"DirectFaithfulnessEvaluator",
	"RagasFaithfulnessEvaluator",
	"ScopeEvaluator",
	"SalienceEvaluator",
]