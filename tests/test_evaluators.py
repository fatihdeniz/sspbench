"""
Tests for the evaluator hierarchy and the new SalienceEvaluator.

These tests use a mock LLM so they run without GPU / model weights.
"""

import pytest
from sspbench.evaluators.base import BaseEvaluator
from sspbench.evaluators.llm_evaluator import LLMEvaluator
from sspbench.evaluators.salience import SalienceEvaluator
from sspbench.evaluators.scope import ScopeEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockModel:
    """Lightweight stand-in for an LLM that returns a canned response."""

    def __init__(self, response: str = ""):
        self.response = response

    def generate(self, *args, **kwargs):
        return self.response


class _ConcreteLLMEvaluator(LLMEvaluator):
    """Minimal concrete sub-class used to test LLMEvaluator itself."""

    @property
    def name(self):
        return "test_llm"

    def evaluate_single(self, sample):
        prompt = sample.get("prompt", "echo")
        resp = self._generate(prompt)
        return {"response": resp}


# ---------------------------------------------------------------------------
# BaseEvaluator contract
# ---------------------------------------------------------------------------

class TestBaseEvaluator:
    def test_cannot_instantiate(self):
        """BaseEvaluator is abstract — instantiation must fail."""
        with pytest.raises(TypeError):
            BaseEvaluator()

    def test_subclass_missing_methods(self):
        """A subclass that doesn't implement all abstract methods must fail."""

        class Incomplete(BaseEvaluator):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# LLMEvaluator
# ---------------------------------------------------------------------------

class TestLLMEvaluator:
    def test_requires_model(self):
        with pytest.raises(ValueError, match="requires an evaluation model"):
            _ConcreteLLMEvaluator(model=None)

    def test_evaluate_single(self):
        model = _MockModel("hello world")
        ev = _ConcreteLLMEvaluator(model=model)
        result = ev.evaluate_single({"prompt": "hi"})
        assert "response" in result

    def test_evaluate_batch(self):
        model = _MockModel("ok")
        ev = _ConcreteLLMEvaluator(model=model)
        out = ev.evaluate([{"prompt": "a"}, {"prompt": "b"}])
        assert "results" in out
        assert "summary" in out
        assert len(out["results"]) == 2

    def test_callable(self):
        """Evaluator instances should be callable (delegates to evaluate)."""
        model = _MockModel("ok")
        ev = _ConcreteLLMEvaluator(model=model)
        out = ev([{"prompt": "x"}])
        assert len(out["results"]) == 1


# ---------------------------------------------------------------------------
# SalienceEvaluator
# ---------------------------------------------------------------------------

class TestSalienceEvaluator:
    @pytest.fixture()
    def evaluator(self):
        model = _MockModel("Answer: 4: High importance.\nExplanation: The question matters a lot.")
        return SalienceEvaluator(model=model, min_score=3)

    def test_name(self, evaluator):
        assert evaluator.name == "salience"

    def test_evaluate_single(self, evaluator):
        result = evaluator.evaluate_single({"question": "What is the capital of France?"})
        assert result["salience_score"] == 4
        assert result["is_salient"] is True
        assert "explanation" in result["salience_explanation"].lower() or len(result["salience_explanation"]) > 0

    def test_evaluate_batch_filtering(self, evaluator):
        samples = [
            {"question": "Who discovered penicillin?"},
            {"question": "What color is the sky?"},
        ]
        out = evaluator.evaluate(samples)
        assert out["summary"]["total_samples"] == 2
        # Both get score 4 from the mock → both salient
        assert all(r["is_salient"] for r in out["results"])

    def test_low_salience_filtered(self):
        model = _MockModel("Answer: 1: No importance.\nExplanation: Trivial.")
        ev = SalienceEvaluator(model=model, min_score=3)
        result = ev.evaluate_single({"question": "What is the 4th book of Keynes?"})
        assert result["salience_score"] == 1
        assert result["is_salient"] is False

    def test_parse_fallback(self):
        """If the LLM returns just a digit, the parser should still extract it."""
        model = _MockModel("3")
        ev = SalienceEvaluator(model=model)
        result = ev.evaluate_single({"question": "test"})
        assert result["salience_score"] == 3


# ---------------------------------------------------------------------------
# ScopeEvaluator (smoke test after refactor)
# ---------------------------------------------------------------------------

class TestScopeEvaluator:
    def test_name(self):
        model = _MockModel('{"in_scope": true, "reason": "factual question"}')
        ev = ScopeEvaluator(model=model, prompt_template="{question} {answer}")
        assert ev.name == "scope"

    def test_evaluate_single(self):
        model = _MockModel('{"in_scope": true, "reason": "factual"}')
        ev = ScopeEvaluator(model=model, prompt_template="Q: {question}\nA: {answer}")
        result = ev.evaluate_single({"question": "Who?", "answer": "Someone"})
        assert result["in_scope"] is True
