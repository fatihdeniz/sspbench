"""
Faithfulness Evaluators

Evaluators for assessing the faithfulness of model-generated answers to given contexts.
"""

from typing import List, Dict, Any
from .base import BaseEvaluator


class DirectFaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluator that uses direct prompting to assess faithfulness and relevancy.
    """

    def __init__(self, eval_model: Any = None, embedding_model: Any = None, **kwargs):
        super().__init__(eval_model, **kwargs)
        self.embedding_model = embedding_model

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate faithfulness using direct model prompting.
        """
        results = []
        for sample in samples:
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            context = sample.get("context", "")
            
            eval_prompt = f"""
Given the question: "{question}"
And the answer: "{answer}"
And the context: "{context}"

Rate the faithfulness of the answer on a scale of 0-1, where:
- 1.0 = The answer is completely faithful to the context
- 0.0 = The answer contradicts or is not supported by the context

Also rate the answer relevancy on a scale of 0-1, where:
- 1.0 = The answer directly and completely answers the question
- 0.0 = The answer does not address the question

Respond with just two numbers separated by a space, like: 0.95 0.88
"""
            
            try:
                if self.model:
                    eval_response = self.model.generate(eval_prompt)
                    if isinstance(eval_response, list):
                        eval_response = eval_response[0]
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
            
            results.append({
                "sample_id": sample.get("id", ""),
                "question": question,
                "answer": answer,
                "context": context,
                "faithfulness": faithfulness_score,
                "answer_relevancy": relevancy_score
            })
        
        summary = self._summarize(results)
        return {"results": results, "summary": summary}

    def compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute average faithfulness and relevancy metrics.
        Note: This is a placeholder as faithfulness evaluation requires context.
        """
        # Since compute_metrics doesn't have context, return dummy values
        return {"average_faithfulness": 0.5, "average_relevancy": 0.5}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        """
        if not results:
            return {"average_faithfulness": 0.0, "average_relevancy": 0.0}
        
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
        
        return {
            "average_faithfulness": avg_faithfulness,
            "average_relevancy": avg_relevancy,
            "total_samples": len(results)
        }


class RagasFaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluator that uses RAGAS library for faithfulness and relevancy assessment.
    """

    def __init__(self, eval_model: Any = None, embedding_model: Any = None, **kwargs):
        super().__init__(eval_model, **kwargs)
        self.embedding_model = embedding_model

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate faithfulness using RAGAS.
        """
        from ..novelty.ragas_utils import evaluate_qa_faithfulness
        
        results = []
        for sample in samples:
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            contexts = sample.get("contexts", [])
            ground_truth = sample.get("ground_truth", None)

            # Accept both `contexts` (list) and `context` (single string)
            context_text = ""
            if isinstance(contexts, list) and contexts:
                context_text = contexts[0]
            elif isinstance(contexts, str):
                context_text = contexts
            else:
                context_text = sample.get("context", "")
            
            try:
                metrics = evaluate_qa_faithfulness(
                    question=question,
                    answer=answer,
                    context=context_text,
                    eval_model=self.model,
                    embedding_model=self.embedding_model
                )
                faithfulness_score = metrics.get("faithfulness", 0.0)
                relevancy_score = metrics.get("answer_relevancy", 0.0)
            except Exception as e:
                print(f"Error in RAGAS faithfulness evaluation: {e}")
                faithfulness_score = 0.0
                relevancy_score = 0.0
            
            results.append({
                "sample_id": sample.get("id", ""),
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "faithfulness": faithfulness_score,
                "answer_relevancy": relevancy_score
            })
        
        summary = self._summarize(results)
        return {"results": results, "summary": summary}

    def compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute average faithfulness and relevancy metrics.
        Note: This is a placeholder as RAGAS evaluation requires full context.
        """
        return {"average_faithfulness": 0.5, "average_relevancy": 0.5}

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        """
        if not results:
            return {"average_faithfulness": 0.0, "average_relevancy": 0.0}
        
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
        
        return {
            "average_faithfulness": avg_faithfulness,
            "average_relevancy": avg_relevancy,
            "total_samples": len(results)
        }