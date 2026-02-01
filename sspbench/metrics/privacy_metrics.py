"""
Privacy Metrics

Functions for computing privacy-related metrics.
"""

from typing import List


def compute_privacy_leakage(predictions: List[str], sensitive_data: List[str]) -> float:
    """
    Compute privacy leakage score.
    """
    leakage_count = sum(1 for p, s in zip(predictions, sensitive_data) if s in p)
    return leakage_count / len(predictions)


def compute_inference_accuracy(inferred: List[str], actual: List[str]) -> float:
    """
    Compute inference attack accuracy.
    """
    correct_count = sum(1 for i, a in zip(inferred, actual) if i == a)
    return correct_count / len(inferred)