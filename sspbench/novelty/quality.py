from ..utils.llm_utils import gen_from_prompt
from .config import FACTUALITY_QA_QUALITYCHECK_SYSTEM_PROMPT
from .json_utils import parse_json_response

def evaluate_question_quality(question: str, answer: str, eval_model) -> dict:
    """Evaluate question quality using LLM judge.
    
    Args:
        question: The question to evaluate
        answer: The gold answer
        eval_model: The evaluation model
        
    Returns:
        Dictionary with is_suitable, scores, total_score, and reasoning
    """
    prompt = f"""Please evaluate the following question-answer pair for hallucination benchmark suitability according to the criteria provided:

Question:
{question}

Gold Answer:
{answer}
"""

    response = gen_from_prompt(
        eval_model,
        prompt,
        temperature=0.0,
        max_tokens=1000,
        system_prompt=FACTUALITY_QA_QUALITYCHECK_SYSTEM_PROMPT
    )

    # Parse JSON response with fallback
    fallback = {
        "is_suitable": False,
        "scores": {},
        "total_score": 0,
        "reasoning": "Failed to parse quality evaluation response."
    }
    
    parsed = parse_json_response(response, fallback)
    
    # Ensure we have a valid dict
    if not isinstance(parsed, dict):
        return fallback
    
    return {
        "is_suitable": parsed.get("is_suitable", False),
        "scores": parsed.get("scores", {}),
        "total_score": parsed.get("total_score", 0),
        "reasoning": parsed.get("reasoning", "No reasoning provided.")
    }
