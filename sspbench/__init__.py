"""
SSP Bench: Safety, Security, and Privacy Dynamic Benchmark Generation Framework

This framework provides tools for generating and evaluating dynamic benchmarks
for assessing the safety, security, and privacy aspects of large language models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import generators
from . import evaluators
from . import metrics
from . import utils

__all__ = ["generators", "evaluators", "metrics", "utils"]