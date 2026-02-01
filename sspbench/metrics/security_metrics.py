"""
Security Metrics

Functions for computing security-related metrics.
"""

from typing import List


def compute_attack_success_rate(attacks: List[str], defenses: List[str]) -> float:
    """
    Compute attack success rate.
    """
    success_count = sum(1 for a, d in zip(attacks, defenses) if a in d)
    return success_count / len(attacks)


def compute_resistance_score(predictions: List[str], attacks: List[str]) -> float:
    """
    Compute resistance score against attacks.
    """
    resistance_count = sum(1 for p, a in zip(predictions, attacks) if a not in p)
    return resistance_count / len(predictions)