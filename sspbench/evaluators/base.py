"""
Base Evaluator Class

Provides the base interface for all benchmark evaluators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseEvaluator(ABC):
    """
    Abstract base class for benchmark evaluators.
    """

    def __init__(self, model: Any = None, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the model on the given samples.

        Args:
            samples: List of benchmark samples

        Returns:
            Dictionary containing evaluation results
        """
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Model predictions
            ground_truth: Expected outputs

        Returns:
            Dictionary of metric names to values
        """
        pass