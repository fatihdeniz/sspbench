"""
Scope Evaluator

Assesses whether generated questions fall within the factuality-hallucination scope.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm_evaluator import LLMEvaluator


class ScopeEvaluator(LLMEvaluator):
    """Evaluator that checks if a QA pair is in scope for factuality tests."""

    def __init__(self, model: Any, prompt_template: str | None = None, **kwargs):
        super().__init__(model, prompt_template=prompt_template, **kwargs)

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "scope"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        answer = sample.get("gold_answer") or sample.get("answer", "")
        prompt = self.prompt_template.format(question=question, answer=answer)

        try:
            response = self._generate(prompt)
        except Exception as exc:
            response = f'{{"in_scope": false, "reason": "model_error: {exc}"}}'

        parsed = self._parse_response(response)
        in_scope = bool(parsed.get("in_scope", False))
        reason = parsed.get("reason", "")

        return {
            "question": question,
            "answer": answer,
            "in_scope": in_scope,
            "in_scope_reason": reason,
            "raw_response": response,
        }

    # ------------------------------------------------------------------
    # Metrics / summary
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if not total:
            return {"in_scope_ratio": 0.0}
        in_scope = sum(1 for r in results if r.get("in_scope"))
        return {"in_scope_ratio": in_scope / total}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        in_scope_count = sum(1 for r in results if r.get("in_scope"))
        return {
            "in_scope_ratio": in_scope_count / total if total else 0.0,
            "total_samples": total,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]
        except ValueError:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
