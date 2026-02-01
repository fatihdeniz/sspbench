"""
Safety Metrics

Functions for computing safety-related metrics.
"""

from typing import List, str


def compute_safety_score(predictions: List[str], expected: List[str]) -> float:
    """
    Compute overall safety score.
    """
    # Placeholder implementation
    safe_count = sum(1 for p in predictions if "safe" in p.lower())
    return safe_count / len(predictions)


def compute_toxicity_score(texts: List[str]) -> List[float]:
    """
    Compute toxicity scores for texts.
    """
    # Placeholder: integrate with toxicity detection model
    return [0.1] * len(texts)  # Low toxicity