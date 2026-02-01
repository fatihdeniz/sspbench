"""
Base Generator Class

Provides the base interface for all benchmark generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseGenerator(ABC):
    """
    Abstract base class for benchmark generators.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @abstractmethod
    def generate(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a list of benchmark samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of dictionaries, each representing a benchmark sample
        """
        pass

    @abstractmethod
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a generated sample.

        Args:
            sample: The sample to validate

        Returns:
            True if valid, False otherwise
        """
        pass