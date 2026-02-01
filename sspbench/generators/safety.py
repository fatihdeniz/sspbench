"""
Safety Benchmark Generator

Generates benchmarks for evaluating model safety, such as resistance to harmful content,
toxicity, and ethical concerns.
"""

from typing import List, Dict, Any
from .base import BaseGenerator


class SafetyGenerator(BaseGenerator):
    """
    Generator for safety-related benchmarks.
    """

    def generate(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate safety benchmark samples.

        Examples: Jailbreak attempts, harmful queries, toxic prompts.
        """
        samples = []
        for i in range(num_samples):
            sample = {
                "id": f"safety_{i}",
                "type": "safety",
                "prompt": f"Sample safety prompt {i}",
                "expected_behavior": "Refuse or handle safely",
                "category": "harmful_content"  # e.g., violence, hate, etc.
            }
            samples.append(sample)
        return samples

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a safety sample.
        """
        required_keys = ["id", "type", "prompt", "expected_behavior"]
        return all(key in sample for key in required_keys) and sample["type"] == "safety"