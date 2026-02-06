"""
Scope Evaluator

Assesses whether generated questions fall within the factuality-hallucination scope.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import BaseEvaluator


class ScopeEvaluator(BaseEvaluator):
    """Evaluator that checks if a QA pair is in scope for factuality tests."""

    def __init__(self, model: Any = None, prompt_template: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.prompt_template = prompt_template

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.model:
            raise ValueError("ScopeEvaluator requires an evaluation model with a generate() method")

        results = []
        in_scope_count = 0

        for idx, sample in enumerate(samples):
            question = sample.get("question", "")
            answer = sample.get("gold_answer") or sample.get("answer", "")
            prompt = self.prompt_template.format(question=question, answer=answer)

            try:
                response = self.model.generate(prompt)
                if isinstance(response, list):
                    response = response[0]
            except Exception as exc:
                response = f"{{\"in_scope\": false, \"reason\": \"model_error: {exc}\"}}"

            parsed = self._parse_response(response)
            in_scope = bool(parsed.get("in_scope", False))
            reason = parsed.get("reason", "")
            in_scope_count += 1 if in_scope else 0

            results.append({
                "sample_id": sample.get("id", idx),
                "question": question,
                "answer": answer,
                "in_scope": in_scope,
                "in_scope_reason": reason,
                "raw_response": response,
            })

        summary = self._summarize(results, len(samples), in_scope_count)
        return {"results": results, "summary": summary}

    def compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        if not predictions or not ground_truth:
            return {"scope_accuracy": 0.0}

        total = min(len(predictions), len(ground_truth))
        correct = 0
        for pred, gold in zip(predictions[:total], ground_truth[:total]):
            pred_bool = str(pred).lower() in {"true", "1", "yes", "in_scope"}
            gold_bool = str(gold).lower() in {"true", "1", "yes", "in_scope"}
            if pred_bool == gold_bool:
                correct += 1
        return {"scope_accuracy": correct / total if total else 0.0}

    def _parse_response(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            # isolate JSON snippet if model adds wrapping text
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]
        except ValueError:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _summarize(self, results: List[Dict[str, Any]], total: int, in_scope_count: int) -> Dict[str, Any]:
        if total == 0:
            return {"in_scope_ratio": 0.0, "total_samples": 0}
        return {
            "in_scope_ratio": in_scope_count / total,
            "total_samples": total,
        }
