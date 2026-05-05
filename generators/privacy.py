"""
Privacy Benchmark Generator

Generates PII awareness benchmarks using SPY Dataset templates filled with
Faker-generated synthetic entities and structural prompt patterns.
"""

from typing import Any, Dict, List, Optional

from .base import BaseGenerator


class PrivacyGenerator(BaseGenerator):
    """Generator for PII awareness benchmarks.

    Delegates to :func:`sspbench.privacy.privacy_core.generate_pii_benchmark`
    which uses SPY Dataset templates + Faker + 6 prompt patterns.

    Parameters
    ----------
    config : dict, optional
        Configuration overrides.  Supported keys:

        * ``template_dir`` – path to SPY JSONL template directory.
        * ``faker_seed``   – seed for Faker reproducibility (default 42).
        * ``domains``      – list of domains to process (default all).
        * ``patterns``     – list of prompt patterns to apply (default all).
    """

    def generate(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate PII awareness benchmark prompts.

        Parameters
        ----------
        num_samples : int
            Maximum number of prompts to return.  The underlying SPY-based
            pipeline generates all prompts first; this parameter truncates
            the result.

        Returns
        -------
        list[dict]
            Prompt dicts with fields: ``id``, ``group_id``, ``prompt``,
            ``domain``, ``pattern``, ``target_pii_type``, ``pii_types``,
            ``ground_truth``, ``all_ground_truth``, ``expected_behavior``.
        """
        from ..privacy.privacy_core import generate_pii_benchmark

        prompts = generate_pii_benchmark(
            template_dir=self.config.get("template_dir"),
            faker_seed=self.config.get("faker_seed", 42),
            domains=self.config.get("domains"),
            patterns=self.config.get("patterns"),
            num_samples=num_samples,
        )
        return prompts

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a PII awareness benchmark sample.

        Checks for the required fields defined in the benchmark spec.
        """
        required_keys = {
            "id", "prompt", "pattern", "target_pii_type",
            "ground_truth", "expected_behavior",
        }
        return required_keys.issubset(sample.keys())
