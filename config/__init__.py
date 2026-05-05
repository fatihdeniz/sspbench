"""
Configuration management for benchmark engines.

This module provides centralized configuration classes for managing
engine settings, model configurations, and runtime parameters.

Examples:
    # Create config from environment variables
    >>> from sspbench.config import BenchmarkConfig
    >>> config = BenchmarkConfig.from_env()

    # Create config programmatically
    >>> config = BenchmarkConfig(
    ...     agent_model="gpt-4",
    ...     test_model="gemma-2-2b",
    ...     max_iterations=5
    ... )

    # Use with engine
    >>> from sspbench.engines import NoveltyEngine
    >>> engine = NoveltyEngine(config)
"""

from .base import BenchmarkConfig
from .models import ModelManager

__all__ = [
    "BenchmarkConfig",
    "ModelManager",
]
