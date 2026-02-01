"""
Security Benchmark Generator

Generates benchmarks for evaluating model security, such as resistance to attacks,
data leakage, and adversarial inputs.
"""

from typing import List, Dict, Any
from .base import BaseGenerator


class SecurityGenerator(BaseGenerator):
    """
    Generator for security-related benchmarks.
    """

    def generate(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate security benchmark samples.

        Examples: Adversarial prompts, injection attacks, data exfiltration attempts.
        """
        samples = []
        for i in range(num_samples):
            sample = {
                "id": f"security_{i}",
                "type": "security",
                "prompt": f"Sample security prompt {i}",
                "expected_behavior": "Detect and mitigate",
                "category": "adversarial_attack"
            }
            samples.append(sample)
        return samples

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a security sample.
        """
        required_keys = ["id", "type", "prompt", "expected_behavior"]
        return all(key in sample for key in required_keys) and sample["type"] == "security"