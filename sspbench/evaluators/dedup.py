"""
Duplicate Evaluator

LLM-based deduplication for QA pairs generated from the same context.
The eval model reviews all questions and identifies which ones are redundant
duplicates of another question, keeping all distinct formulations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .llm_evaluator import LLMEvaluator


DEDUP_SYSTEM_PROMPT = """\
You are a question quality judge for a factual QA benchmark.

You will receive a numbered list of questions generated from the same source \
passage. Some of them may ask about the same fact using different wording — \
these are duplicates.

Your task:
1. Identify questions that are duplicates of each other (same underlying fact).
2. For each set of duplicates, keep the ONE with the clearest, most specific \
wording and mark the others for removal.
3. Questions that ask about different facts are NOT duplicates — keep all of them.

Respond with ONLY valid JSON:
{"keep": [<1-based ids to keep>], "remove": [<1-based ids to remove>], \
"reason": "<brief explanation>"}

Example — given 3 questions where #1 and #3 ask the same thing:
{"keep": [1, 2], "remove": [3], "reason": "#3 is a less clear duplicate of #1"}
"""

DEDUP_USER_TEMPLATE = """\
Review the following {n} questions for duplicates. Remove only the redundant \
ones and keep all distinct questions.

{questions_block}
"""


class DuplicateEvaluator(LLMEvaluator):

    def __init__(
        self,
        model: Any,
        key_question: str = "question",
        **kwargs,
    ):
        super().__init__(
            model,
            prompt_template=DEDUP_USER_TEMPLATE,
            system_prompt=DEDUP_SYSTEM_PROMPT,
            **kwargs,
        )
        self.key_question = key_question

    @property
    def name(self) -> str:
        return "dedup"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"is_duplicate": False}

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.evaluate_single(s) for s in samples]
        return {"results": results, "summary": self._summarize(results)}

    def select(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ask the LLM to identify and remove only duplicate questions."""
        if len(qa_pairs) <= 1:
            return qa_pairs

        questions_block = "\n".join(
            f"{i+1}. {q.get(self.key_question, '')}" for i, q in enumerate(qa_pairs)
        )
        prompt = self.prompt_template.format(n=len(qa_pairs), questions_block=questions_block)

        try:
            response = self._generate(prompt)
            keep_ids = self._parse_response(response, len(qa_pairs))
            kept = [qa_pairs[i] for i in keep_ids]
        except Exception:
            kept = qa_pairs

        removed = len(qa_pairs) - len(kept)
        if removed:
            print(f"  [{self.name}] Filtered {removed} duplicate(s) "
                  f"from {len(qa_pairs)} questions, kept {len(kept)}")
        return kept

    @staticmethod
    def _parse_response(text: str, total: int) -> List[int]:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return list(range(total))

        keep_raw = data.get("keep")
        if isinstance(keep_raw, list) and keep_raw:
            ids = [int(x) - 1 for x in keep_raw if isinstance(x, (int, float)) and 1 <= int(x) <= total]
            if ids:
                return sorted(set(ids))

        remove_raw = data.get("remove")
        if isinstance(remove_raw, list):
            remove_set = {int(x) - 1 for x in remove_raw if isinstance(x, (int, float)) and 1 <= int(x) <= total}
            return [i for i in range(total) if i not in remove_set]

        return list(range(total))

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if not total:
            return {"duplicate_ratio": 0.0}
        dupes = sum(1 for r in results if r.get("is_duplicate"))
        return {"duplicate_ratio": round(dupes / total, 3)}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        dupes = sum(1 for r in results if r.get("is_duplicate"))
        return {"total": total, "duplicates": dupes, "kept": total - dupes}
