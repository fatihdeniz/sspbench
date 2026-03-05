"""
Salience Evaluator

Rates the importance / salience of generated factual QA questions on a 1–5
scale so that benchmark questions focus on facts that actually matter.

The scoring rubric comes from ``docs/salience_instructions.txt`` and is
stored as ``SALIENCE_JUDGE_SYSTEM_PROMPT`` in ``novelty/config.py``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .llm_evaluator import LLMEvaluator


class SalienceEvaluator(LLMEvaluator):
    """
    LLM-based evaluator that assigns an importance score (1–5) to each
    question–answer pair.

    Parameters
    ----------
    model : Any
        Evaluation LLM (must support ``generate``).
    min_score : int
        Minimum salience score to consider a question *salient* (default 3).
    prompt_template : str, optional
        Override the default user-prompt template.  Must contain
        ``{question}`` placeholder.
    system_prompt : str, optional
        Override the system-level rubric.  When ``None`` the constant
        ``SALIENCE_JUDGE_SYSTEM_PROMPT`` from ``novelty.config`` is used.
    """

    DEFAULT_USER_TEMPLATE = (
        "Rate the importance of the following question according to the instructions.\n\n"
        "Question: {question}\n\n"
        "Respond with ONLY a single line in the format:\n"
        "Answer: <score>: <importance level>.\n"
        "Explanation: <brief explanation>\n"
    )

    def __init__(
        self,
        model: Any,
        min_score: int = 3,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        key_field: str = "question",
        **kwargs,
    ):
        if system_prompt is None:
            from ..novelty.config import SALIENCE_JUDGE_SYSTEM_PROMPT
            system_prompt = SALIENCE_JUDGE_SYSTEM_PROMPT
        super().__init__(
            model,
            prompt_template=prompt_template or self.DEFAULT_USER_TEMPLATE,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.min_score = min_score
        self.key_field = key_field

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "salience"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get(self.key_field, "")
        prompt = self.prompt_template.format(question=question)

        try:
            response = self._generate(prompt)
        except Exception as exc:
            response = f"Error: {exc}"

        score, explanation = self._parse_response(response)
        is_salient = score >= self.min_score

        return {
            "question": question,
            "salience_score": score,
            "salience_explanation": explanation,
            "is_salient": is_salient,
            "raw_response": response,
        }

    # ------------------------------------------------------------------
    # Metrics / summary
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if not total:
            return {"average_salience": 0.0, "salient_ratio": 0.0}
        avg = sum(r.get("salience_score", 0) for r in results) / total
        salient = sum(1 for r in results if r.get("is_salient"))
        return {
            "average_salience": round(avg, 3),
            "salient_ratio": round(salient / total, 3),
        }

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = self.compute_metrics(results)
        metrics["total_samples"] = len(results)
        return metrics

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    _SCORE_RE = re.compile(r"(\d)\s*:")

    def _parse_response(self, text: str) -> tuple[int, str]:
        """Return ``(score, explanation)`` from the LLM response.

        Tries to find a line like ``Answer: 4: High importance.`` and
        extracts the leading digit.  Falls back to scanning the whole
        response for a bare digit 1-5.
        """
        explanation = ""
        score = 0

        if not text:
            return score, explanation

        lines = text.strip().splitlines()

        # Look for an "Answer:" or "Explanation:" line
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("answer"):
                m = self._SCORE_RE.search(stripped)
                if m:
                    score = int(m.group(1))
            if stripped.lower().startswith("explanation"):
                explanation = stripped.split(":", 1)[-1].strip()

        # Fallback: grab first digit 1-5 in the response
        if score == 0:
            m = re.search(r"\b([1-5])\b", text)
            if m:
                score = int(m.group(1))

        if not explanation:
            explanation = text.strip()

        return score, explanation
