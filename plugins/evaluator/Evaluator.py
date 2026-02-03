"""
Evaluator class
=================
"""
from tqdm import tqdm
from abc import ABC, abstractmethod
from custom_datasets import Dataset

class Evaluator(ABC):
    """
    Abstract base class for implementing evaluation strategies for conversations.
    """
    def __init__(self, eval_model = None):
        """
        :param ~ModelBase eval_model: The evaluation model to be used for dataset evaluation. Can be any model compatible with the evaluation strategy.
        """
        self.eval_model = eval_model

    def __call__(self, dataset: Dataset, **kwargs):
        """
        Execute the evaluation process on the provided Dataset.

        :param ~Dataset dataset: The dataset to be evaluated.
        :param **kwargs: Additional keyword arguments that can be passed to the _evaluate method.
        """
        for instance in tqdm(dataset, desc="Evaluating responses"):
            self._evaluate(instance, **kwargs)

    @abstractmethod
    def _evaluate(self, instance, **kwargs):
        """
        Abstract method to be implemented by subclasses for specific evaluation strategies.

        :param ~Instance instance: A single instance from the aiXamine custom dataset to be evaluated.
        :param **kwargs: Additional keyword arguments relevant to the specific evaluation strategy.
        :return: The result of the evaluation, the nature of which depends on the specific implementation in subclasses.
        """
        return NotImplementedError()
