"""
LLM Evaluator Base Class

Intermediate base for evaluators that rely on an LLM judge, following the
``LLMEvaluator`` pattern from the aiXamine airflow-tasks evaluator plugin.

Concrete sub-classes only need to implement ``evaluate_single`` and ``name``;
``evaluate`` loops automatically and builds a ``{"results", "summary"}`` dict.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base import BaseEvaluator


class LLMEvaluator(BaseEvaluator):
    """
    Base class for evaluators that use an LLM to judge each sample.

    Parameters
    ----------
    model : Any
        An object with a ``generate()`` method (e.g. a vLLM model wrapper).
    prompt_template : str, optional
        A format-string with placeholders that ``evaluate_single`` can fill
        (e.g. ``"{question}"``).
    system_prompt : str, optional
        System-level instruction sent with every LLM call.
    **kwargs
        Forwarded to :class:`BaseEvaluator`.
    """

    def __init__(
        self,
        model: Any,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError(f"{self.__class__.__name__} requires an evaluation model")
        super().__init__(model, **kwargs)
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Default batch evaluate — loops evaluate_single
    # ------------------------------------------------------------------

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch by calling :meth:`evaluate_single` on each sample.

        Returns ``{"results": [...], "summary": {...}}``.
        """
        results: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            result = self.evaluate_single(sample)
            # Ensure every result carries a sample id
            if "sample_id" not in result:
                result["sample_id"] = sample.get("id", idx)
            results.append(result)

        summary = self._summarize(results)
        return {"results": results, "summary": summary}

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Send a single prompt to the evaluation LLM and return the response.

        Uses ``gen_from_prompt`` from the project's LLM utilities so that
        conversation wrapping and sampling-param handling stay consistent.
        """
        from ..utils.llm_utils import gen_from_prompt

        response = gen_from_prompt(
            model=self.model,
            prompt=prompt,
            system_prompt=kwargs.get("system_prompt", self.system_prompt),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        if isinstance(response, list):
            response = response[0]
        return response
