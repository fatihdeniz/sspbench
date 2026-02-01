"""
Tests for generators
"""

import pytest
from sspbench.generators import SafetyGenerator


def test_safety_generator():
    generator = SafetyGenerator()
    samples = generator.generate(num_samples=5)
    assert len(samples) == 5
    for sample in samples:
        assert generator.validate_sample(sample)
        assert sample["type"] == "safety"