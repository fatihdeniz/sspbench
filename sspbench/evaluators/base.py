"""
Base Evaluator Class

Provides the base interface for all benchmark evaluators.

Design Notes
------------
This hierarchy is modelled after the ``Evaluator`` / ``LLMEvaluator`` pattern
used in the aiXamine airflow-tasks evaluator plugin so that SSPBench evaluators
share the same conventions:

* ``BaseEvaluator``  – abstract root; defines the contract every evaluator must
  satisfy (``evaluate``, ``evaluate_single``, ``name``).  Instances are
  *callable* (``__call__`` delegates to ``evaluate``).

* ``LLMEvaluator`` (in ``llm_evaluator.py``) – intermediate class for any
  evaluator that needs an LLM judge.  Provides ``_generate()`` helper and a
  default ``evaluate()`` that loops over samples calling ``evaluate_single()``.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseEvaluator(ABC):
    """
    Abstract base class for benchmark evaluators.

    Subclasses **must** implement:
    * ``evaluate_single`` – evaluate one sample and return a result dict.
    * ``evaluate``        – evaluate a batch; default impl loops ``evaluate_single``.
    * ``name``            – short identifier used in logs / result keys.

    ``compute_metrics`` and ``_summarize`` have sensible defaults but can be
    overridden.
    """

    def __init__(self, model: Any = None, **kwargs):
        self.model = model
        self.config = kwargs

    # ------------------------------------------------------------------
    # Callable interface (mirrors aiXamine Evaluator.__call__)
    # ------------------------------------------------------------------

    def __call__(self, samples: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Convenience callable that delegates to :meth:`evaluate`."""
        return self.evaluate(samples, **kwargs)

    # ------------------------------------------------------------------
    # Core contract
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the evaluator (e.g. ``'scope'``, ``'salience'``)."""
        ...

    @abstractmethod
    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a **single** sample.

        Args:
            sample: One benchmark sample dict.

        Returns:
            A result dict for this sample (schema is evaluator-specific).
        """
        ...

    @abstractmethod
    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the model on the given samples.

        Args:
            samples: List of benchmark samples.

        Returns:
            Dictionary with at least ``"results"`` (list) and ``"summary"`` (dict).
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks with sensible defaults
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute aggregate metrics from a list of per-sample result dicts.

        The default implementation returns an empty dict; override to add
        domain-specific metrics.
        """
        return {}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute a summary dict from per-sample results.

        Default: return total sample count.  Override for richer summaries.
        """
        return {"total_samples": len(results)}