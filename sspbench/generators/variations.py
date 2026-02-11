"""
Variation utilities for the Novelty Engine.
Functions for applying semantic-preserving variations to questions.
"""

import copy
from ..utils.llm_utils import gen_from_prompt


def apply_variations_to_dataset(questions, model):
    """
    Apply semantic-preserving variations to a dataset of questions.

    Args:
        questions: List of question dictionaries
        model: Model to use for generating variations

    Returns:
        List of questions with variations
    """
    varied_questions = []
    for q in questions:
        # Original question
        q['variation_type'] = 'original'
        varied_questions.append(q.copy())

        # Typo variation
        typo_q = q.copy()
        typo_q['question'] = apply_typos(q['question'])
        typo_q['id'] = f"{q['id']}_typo"
        typo_q['variation_type'] = 'typo'
        varied_questions.append(typo_q)

        # Contextual variation
        context_q = q.copy()
        context_q['question'] = apply_contextualization(q['question'], model)
        context_q['id'] = f"{q['id']}_context"
        context_q['variation_type'] = 'context'
        varied_questions.append(context_q)

    print(f"Applied variations: {len(questions)} original → {len(varied_questions)} total questions")
    return varied_questions


def apply_typos(text):
    """
    Apply simple typos to text.

    Args:
        text: Input text

    Returns:
        Text with typos applied
    """
    # Simple replacement - only replace first occurrence to avoid over-modification
    typos = {'the': 'teh', 'and': 'adn', 'is': 'si', 'to': 'ot', 'for': 'foe', 'that': 'taht'}
    for correct, wrong in typos.items():
        text = text.replace(correct, wrong, 1)  # Replace only first occurrence
    return text


def apply_contextualization(question, model):
    """
    Add context to a question to make it longer but keep the core question.

    Args:
        question: Original question
        model: Model to use for generation

    Returns:
        Question with added context
    """
    context = f"Add some background context to this question: {question}\nMake it longer but keep the core question."
    response = gen_from_prompt(model, context, temperature=0.3, max_tokens=200)
    return response.strip()