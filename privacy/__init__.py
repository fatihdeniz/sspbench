"""
Privacy domain module for SSPBench.

Provides PII awareness benchmark generation and evaluation using the SPY
Dataset with Faker-generated synthetic entities and structural prompt patterns.
"""

from .privacy_config import (
    PRIVACY_TAXONOMY,
    PRIVACY_DOMAINS,
    PRIVACY_PATTERNS,
    DEFAULT_FAKER_SEED,
)
from .privacy_core import (
    generate_pii_benchmark,
    export_pii_benchmark,
)

__all__ = [
    "PRIVACY_TAXONOMY",
    "PRIVACY_DOMAINS",
    "PRIVACY_PATTERNS",
    "DEFAULT_FAKER_SEED",
    "generate_pii_benchmark",
    "export_pii_benchmark",
]
