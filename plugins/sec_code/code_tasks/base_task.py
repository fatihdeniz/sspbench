from __future__ import annotations
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sec_code.metrics.error_codes import ErrorCode
from sec_code.metrics.refusal import is_response_llm_refusal

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    raw_data: Any
    id_: str
    task: Task
    subtask: dict[str, str]
    subtask_desc: str
    reference: str | list[str] | None
    messages: list[dict[str, str]]
    response: str | None = None
    raw_response: str | None = None
    rejected: bool | None = None
    metrics: dict[str, float | None] | None = None


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    FEWSHOT_RESERVE = 10

    AVAIL_METRICS: list[str] | None = None
    AVAIL_SUBTASKS: dict[str, list[str]] | None = None

    TASK_FULL_NAME: str | None = None

    def __str__(self):
        return self.TASK_FULL_NAME

    def __init__(
        self,
        metric_functions: dict[str, callable] | None = None,
        subtasks: dict[str, list[str]] | None = None,
        num_data: int | None = None,
        fewshot_num: int | None = None,
        shuffle_data: bool = False,
        batch_size: int = 1,
    ):
        """
        :param subtasks: dict[str, list[str]]
        :param metric_functions: dict[str, callable]
        :param num_data: int
            load the first num_data examples
        :param shuffle_data: bool
            load the examples in random order
        :param stop_words: list[str]
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            whether the task requires code execution during evaluation or not
        :param language: str
            the language of the task, all represents all languages
        """
        self.subtasks = subtasks
        self.shuffle_data = shuffle_data
        self.batch_size = batch_size
        self.fewshot_num = fewshot_num if fewshot_num else 0

        # init metrics
        self.metric_functions = {}
        
    @staticmethod
    def describe_subtask(subtask: dict[str, str]) -> str:
        """
        Describe the subtask.
        """
        return ", ".join(f"{k}: {v}" for k, v in subtask.items())


    @abstractmethod
    def get_prompt(self, doc, include_security_policy) -> str:
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    def get_system_prompt(self, doc) -> str | None:
        return None

    @abstractmethod
    def get_reference(self, doc) -> str | list[str]:
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    def get_reference_model_output(self, doc) -> str:
        """Builds the reference raw answer from the model for the doc.
        Use it when you want to reformat the reference answer like in a code block or json format.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        return self.get_reference(doc)

    @abstractmethod
    def get_id(self, doc) -> str:
        """
        Get the id of the doc
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

      
    def postprocess_generation(self, response: str):
        """Defines the postprocessing for an LM generation.
        :param response: str
            Code generation from the language model
        """
        response = re.sub(r'```(?:\w+)?\s*\n```', '```', response)
        pattern = re.compile(r"```(?:\w+)?\s*\n([\s\S]*?)```", re.DOTALL)
        code_blocks = pattern.findall(response)
        if code_blocks:
            return max((block for block in code_blocks if block.strip()), key=len, default=response)
        return response

    def evaluate(self, data: DataPoint):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param data: DataPoint
        """
        results = {}
        # before everything, we need to examine if llm reject to answer the prompt
        if is_response_llm_refusal(data.response):
            data.rejected = True
            # all metrics equals to REFUSAL
            data.metrics = {
                name: ErrorCode.RESPONSE_REFUSAL.value for name in self.metric_functions
            }
        else:
            data.rejected = False
            # if llm does not refuse to answer the prompt, we can evaluate the response
            for name, metric in self.metric_functions.items():
                score = metric(data)
                results[name] = score
            data.metrics = results


class MultiTurnTask(Task, ABC):
    """A multi-turn task represents an entire benchmark including its dataset, problems, answers, generation settings and evaluation methods."""

    FEWSHOT_RESERVE = 0

    AVAIL_METRICS: list[str] | None = None
    AVAIL_SUBTASKS: dict[str, list[str]] | None = None

    TASK_FULL_NAME: str | None = None

    def __str__(self):
        return self.TASK_FULL_NAME

    def __init__(
        self,
        metric_functions: dict[str, callable],
        subtasks: dict[str, list[str]] | None = None,
        num_data: int | None = None,
        fewshot_num: int | None = None,
        shuffle_data: bool = False,
        batch_size: int = 1,
    ):
        """
        :param subtasks: dict[str, list[str]]
        :param metric_functions: dict[str, callable]
        :param num_data: int
            load the first num_data examples
        :param shuffle_data: bool
            load the examples in random order
        :param stop_words: list[str]
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            whether the task requires code execution during evaluation or not
        :param language: str
            the language of the task, all represents all languages
        """
        # check parameters
        assert self.FEWSHOT_RESERVE, "Multi-turn task does not support fewshot"
        super().__init__(
            metric_functions, subtasks, num_data, fewshot_num, shuffle_data, batch_size
        )


    @abstractmethod
    def initial_env(self, data: DataPoint):
        pass

    @abstractmethod
    def update_prompt(self, feedback: str, data: DataPoint):
        pass

    @abstractmethod
    def feedback(self, data: DataPoint) -> str:
        pass


class TaskRegistry:
    def __init__(self, tasks: dict[str, type[Task]]):
        self.tasks = tasks

    def __getitem__(self, task_name: str) -> type[Task]:
        # directly match with task name
        task_cand = self.tasks.get(task_name)
        if task_cand is not None:
            return task_cand

        # TODO: try to match with regex pattern
        raise ValueError(f"Task {task_name} not found in the registry.")
