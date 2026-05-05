"""
Benchmark Evaluators

This module contains classes and functions for evaluating models on generated benchmarks.

Hierarchy
---------
BaseEvaluator           (abstract root)
  ├─ AnswerLeakageEvaluator   (deterministic string filter)
  ├─ PrivacyEvaluator         (deterministic PII leak detection)
  └─ LLMEvaluator             (adds LLM judge helpers)
       ├─ DuplicateEvaluator
       ├─ ScopeEvaluator
       ├─ SalienceEvaluator
       ├─ DirectFaithfulnessEvaluator
       ├─ SafetyEvaluator
       └─ SecurityEvaluator
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
from .answer_leakage import AnswerLeakageEvaluator
from .dedup import DuplicateEvaluator

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
	"AnswerLeakageEvaluator",
	"DuplicateEvaluator",
]