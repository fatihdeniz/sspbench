"""
Over-Refusal Benchmark Generation Module

This module generates benign boundary questions to test for over-refusal
(false positives) where models incorrectly refuse legitimate requests.

Key Components:
- or_config.py: Taxonomy, prompts, thresholds
- or_core.py: Category generation, prompt generation, filtering
- or_eval.py: Response pattern classification, over-refusal detection

Pipeline Flow:
1. Generate benign boundary categories
2. Mine similar benign examples from corpus
3. Generate grounded benign questions
4. Strict benign validation
5. Evaluate with test model (detect over-refusal)

Target: HIGH refusal (0.7-0.9) on BENIGN questions = tough boundary cases
"""

__version__ = "0.1.0"
