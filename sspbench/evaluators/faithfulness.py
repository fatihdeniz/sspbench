"""
Faithfulness Evaluators

Evaluators for assessing the faithfulness of model-generated answers to given contexts.
"""

from typing import List, Dict, Any

from .llm_evaluator import LLMEvaluator
from .base import BaseEvaluator


class DirectFaithfulnessEvaluator(LLMEvaluator):
    """
    Evaluator that uses direct prompting to assess faithfulness and relevancy.
    """

    def __init__(self, eval_model: Any = None, embedding_model: Any = None, **kwargs):
        # eval_model may be None for offline / placeholder usage
        super().__init__(eval_model or _PlaceholderModel(), **kwargs)
        self._has_real_model = eval_model is not None
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "direct_faithfulness"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        context = sample.get("context", "")

        eval_prompt = (
            f'Given the question: "{question}"\n'
            f'And the answer: "{answer}"\n'
            f'And the context: "{context}"\n\n'
            "Rate the faithfulness of the answer on a scale of 0-1, where:\n"
            "- 1.0 = The answer is completely faithful to the context\n"
            "- 0.0 = The answer contradicts or is not supported by the context\n\n"
            "Also rate the answer relevancy on a scale of 0-1, where:\n"
            "- 1.0 = The answer directly and completely answers the question\n"
            "- 0.0 = The answer does not address the question\n\n"
            "Respond with just two numbers separated by a space, like: 0.95 0.88\n"
        )

        try:
            if self._has_real_model:
                eval_response = self._generate(eval_prompt)
                parts = eval_response.strip().split()
                if len(parts) >= 2:
                    faithfulness_score = float(parts[0])
                    relevancy_score = float(parts[1])
                else:
                    faithfulness_score = 0.5
                    relevancy_score = 0.5
            else:
                faithfulness_score = 0.5
                relevancy_score = 0.5
        except Exception as e:
            print(f"Error in direct faithfulness evaluation: {e}")
            faithfulness_score = 0.5
            relevancy_score = 0.5

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
        }

    # ------------------------------------------------------------------
    # Metrics / summary
    # ------------------------------------------------------------------

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"average_faithfulness": 0.0, "average_relevancy": 0.0}
        return {
            "average_faithfulness": sum(r["faithfulness"] for r in results) / len(results),
            "average_relevancy": sum(r["answer_relevancy"] for r in results) / len(results),
        }

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"average_faithfulness": 0.0, "average_relevancy": 0.0, "total_samples": 0}
        metrics = self.compute_metrics(results)
        metrics["total_samples"] = len(results)
        return metrics


class RagasFaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluator that uses RAGAS library for faithfulness and relevancy assessment.

    This evaluator wraps the RAGAS toolkit rather than prompting a single LLM,
    so it inherits directly from :class:`BaseEvaluator` (not ``LLMEvaluator``).
    """

    def __init__(self, eval_model: Any = None, embedding_model: Any = None, **kwargs):
        super().__init__(eval_model, **kwargs)
        self.embedding_model = embedding_model

    @property
    def name(self) -> str:
        return "ragas_faithfulness"

    def evaluate_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        from ..novelty.ragas_utils import evaluate_qa_faithfulness

        question = sample.get("question", "")
        answer = sample.get("answer", "")
        context_text = sample.get("context", "")
        ground_truth = sample.get("ground_truth", None)

        if not context_text.strip():
            return {
                "question": question,
                "answer": answer,
                "context": context_text,
                "ground_truth": ground_truth,
                "faithfulness": None,
                "answer_relevancy": None,
            }

        try:
            metrics = evaluate_qa_faithfulness(
                question=question,
                answer=answer,
                context=context_text,
                eval_model=self.model,
                embedding_model=self.embedding_model,
            )
            faithfulness_score = metrics.get("faithfulness", 0.0)
            relevancy_score = metrics.get("answer_relevancy", 0.0)
        except Exception as e:
            print(f"Error in RAGAS faithfulness evaluation: {e}")
            faithfulness_score = 0.0
            relevancy_score = 0.0

        return {
            "question": question,
            "answer": answer,
            "context": context_text,
            "ground_truth": ground_truth,
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
        }

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for idx, sample in enumerate(samples):
            result = self.evaluate_single(sample)
            result["sample_id"] = sample.get("id", idx)
            results.append(result)
        summary = self._summarize(results)
        return {"results": results, "summary": summary}

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"average_faithfulness": 0.0, "average_relevancy": 0.0}
        return {
            "average_faithfulness": sum(r["faithfulness"] for r in results if r["faithfulness"] is not None) / max(1, len(results)),
            "average_relevancy": sum(r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None) / max(1, len(results)),
        }

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"average_faithfulness": 0.0, "average_relevancy": 0.0, "total_samples": 0}
        metrics = self.compute_metrics(results)
        metrics["total_samples"] = len(results)
        return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class _PlaceholderModel:
    """Dummy model so ``DirectFaithfulnessEvaluator`` can be instantiated
    without a real model (returns neutral scores)."""

    def generate(self, *a, **kw):
        return "0.5 0.5"