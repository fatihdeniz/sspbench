#!/usr/bin/env python3
"""
Example usage of SSP Bench
"""

from sspbench.generators import SafetyGenerator, SecurityGenerator, PrivacyGenerator
from sspbench.evaluators import SafetyEvaluator, SecurityEvaluator, PrivacyEvaluator
from sspbench.utils import setup_logging

def main():
    setup_logging()

    # Generate benchmarks
    safety_gen = SafetyGenerator()
    security_gen = SecurityGenerator()
    privacy_gen = PrivacyGenerator()

    safety_samples = safety_gen.generate(10)
    security_samples = security_gen.generate(10)
    privacy_samples = privacy_gen.generate(10)

    print(f"Generated {len(safety_samples)} safety samples")
    print(f"Generated {len(security_samples)} security samples")
    print(f"Generated {len(privacy_samples)} privacy samples")

    # Evaluate (with mock model)
    safety_eval = SafetyEvaluator()
    security_eval = SecurityEvaluator()
    privacy_eval = PrivacyEvaluator()

    safety_results = safety_eval.evaluate(safety_samples)
    security_results = security_eval.evaluate(security_samples)
    privacy_results = privacy_eval.evaluate(privacy_samples)

    print("Safety evaluation summary:", safety_results["summary"])
    print("Security evaluation summary:", security_results["summary"])
    print("Privacy evaluation summary:", privacy_results["summary"])

if __name__ == "__main__":
    main()