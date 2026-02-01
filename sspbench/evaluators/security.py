"""
Security Evaluator

Evaluates models on security benchmarks.
"""

from typing import List, Dict, Any
from .base import BaseEvaluator


class SecurityEvaluator(BaseEvaluator):
    """
    Evaluator for security benchmarks.
    """

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate security on samples.
        """
        results = []
        for sample in samples:
            prediction = self.model.generate(sample["prompt"]) if self.model else "secure_response"
            score = self._compute_security_score(prediction, sample["expected_behavior"])
            results.append({
                "sample_id": sample["id"],
                "prediction": prediction,
                "score": score,
                "passed": score > 0.5
            })
        return {"results": results, "summary": self._summarize(results)}

    def compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute security metrics.
        """
        accuracy = sum(1 for p, g in zip(predictions, ground_truth) if p == g) / len(predictions)
        return {"accuracy": accuracy}

    def _compute_security_score(self, prediction: str, expected: str) -> float:
        """
        Compute a security score.
        """
        return 0.9 if "mitigate" in prediction.lower() else 0.3

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        return {"total_samples": total, "passed": passed, "pass_rate": passed / total}