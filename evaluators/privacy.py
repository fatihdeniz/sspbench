"""
Privacy Evaluator

Evaluates model responses on PII awareness benchmarks using deterministic
PII leak detection.  Checks whether ground truth PII strings appear in the
model's response text.

Extends :class:`BaseEvaluator` (not ``LLMEvaluator``) because the core
evaluation is string matching -- no LLM judge is needed.
"""

from typing import Any, Dict, List, Optional

from .base import BaseEvaluator


class PrivacyEvaluator(BaseEvaluator):
    """Evaluator for PII awareness benchmarks.

    Determines whether a model's response leaks PII from the ground truth.
    Each sample must include a ``ground_truth`` dict and a ``response`` field
    (the model's output).  If ``response`` is missing, a placeholder is used.

    Parameters
    ----------
    model : Any, optional
        Not required for evaluation (string-matching based).  Accepted for
        API compatibility with the evaluator hierarchy.
    **kwargs
        Forwarded to :class:`BaseEvaluator`.
    """

    @property
    def name(self) -> str:
        return "privacy"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate one sample for PII leakage.

        Parameters
        ----------
        sample : dict
            Must contain ``ground_truth`` (dict[str, list[str]]) and
            ``response`` (str).

        Returns
        -------
        dict
            Result with ``passed``, ``score``, ``leaked_types``, etc.
        """
        from ..privacy.privacy_core import evaluate_pii_sample

        response = sample.get("response", sample.get("prediction", ""))
        return evaluate_pii_sample(sample, response)

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of samples.

        Parameters
        ----------
        samples : list[dict]
            Each dict should have ``ground_truth`` and ``response``.

        Returns
        -------
        dict
            ``{"results": [...], "summary": {...}}`` with per-sample results
            and aggregate privacy metrics.
        """
        results = []
        for idx, sample in enumerate(samples):
            result = self.evaluate_single(sample)
            if "sample_id" not in result or not result["sample_id"]:
                result["sample_id"] = sample.get("id", idx)
            results.append(result)

        summary = self._summarize(results)
        return {"results": results, "summary": summary}

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate pass rate."""
        total = len(results)
        if not total:
            return {"pass_rate": 0.0}
        passed = sum(1 for r in results if r.get("passed"))
        return {"pass_rate": passed / total}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary with full privacy metrics."""
        from ..privacy.privacy_core import compute_privacy_metrics

        return compute_privacy_metrics(results)
