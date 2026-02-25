"""
Answer Leakage Evaluator

Detects questions where the gold answer appears inside the question text,
which invalidates them as knowledge probes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import BaseEvaluator


class AnswerLeakageEvaluator(BaseEvaluator):

    def __init__(
        self,
        key_question: str = "question",
        key_answer: str = "gold_answer",
        word_overlap_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key_question = key_question
        self.key_answer = key_answer
        self.word_overlap_threshold = word_overlap_threshold

    @property
    def name(self) -> str:
        return "answer_leakage"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get(self.key_question, "").lower()
        answer = sample.get(self.key_answer, sample.get("answer", "")).strip().lower()

        if not answer or len(answer) < 3:
            return {"has_leakage": False, "leakage_type": None}

        if answer in question:
            return {"has_leakage": True, "leakage_type": "substring"}

        ans_words = [w for w in re.split(r"\W+", answer) if len(w) > 3]
        if len(ans_words) > 1:
            q_words = set(re.split(r"\W+", question))
            overlap = sum(1 for w in ans_words if w in q_words)
            if overlap / len(ans_words) > self.word_overlap_threshold:
                return {"has_leakage": True, "leakage_type": "word_overlap"}

        return {"has_leakage": False, "leakage_type": None}

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.evaluate_single(s) for s in samples]
        return {"results": results, "summary": self._summarize(results)}

    def filter(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output = self.evaluate(samples)
        kept = [s for s, r in zip(samples, output["results"]) if not r["has_leakage"]]
        removed = len(samples) - len(kept)
        if removed:
            print(f"  [{self.name}] Removed {removed} questions ({len(kept)} remaining)")
        return kept

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if not total:
            return {"leak_ratio": 0.0}
        leaked = sum(1 for r in results if r.get("has_leakage"))
        return {"leak_ratio": round(leaked / total, 3)}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        leaked = sum(1 for r in results if r.get("has_leakage"))
        return {"total": total, "leaked": leaked, "kept": total - leaked}
