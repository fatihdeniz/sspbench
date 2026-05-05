"""
High-level engine API for benchmark generation.

This module provides user-friendly classes for running benchmark generation
pipelines. These engines wrap the lower-level functions in sspbench.novelty
and sspbench.safety with a cleaner, more consistent API.

Examples:
    # Novelty engine - single theme
    >>> from sspbench.engines import NoveltyEngine
    >>> engine = NoveltyEngine()
    >>> results = engine.run(theme="science", max_iterations=3)

    # Novelty engine - multiple themes
    >>> results = engine.run(themes=["science", "history", "geography"])

    # Safety engine
    >>> from sspbench.engines import SafetyEngine
    >>> engine = SafetyEngine()
    >>> results = engine.run(theme="safety alignment", max_iterations=5)

    # Over-refusal engine
    >>> from sspbench.engines import OverRefusalEngine
    >>> engine = OverRefusalEngine()
    >>> results = engine.run(theme="benign boundary questions", max_iterations=5)
"""

from .base import BaseEngine
from .novelty import NoveltyEngine
from .safety import SafetyEngine
from .over_refusal import OverRefusalEngine

__all__ = [
    "BaseEngine",
    "NoveltyEngine",
    "SafetyEngine",
    "OverRefusalEngine",
]
