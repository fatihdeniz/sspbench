"""
Core functionality for the Novelty Engine.
Main pipeline and category/question generation functions.
"""

import os
import json
import copy
from collections import defaultdict

from tool_util import extract_json_v2
from .llm_utils import gen_from_prompt
from .wiki_utils import search_step
from .config import DEFAULT_JSON_MESSAGE
from .ragas_utils import generate_qa_with_ragas, is_ragas_available, evaluate_qa_faithfulness


def _generate_categories_random(theme, agent_model, history, iteration, outfile_prefix='att1'):
    """
    Generate initial categories for a theme.

    Args:
        theme: Theme for question generation
        agent_model: Model to use for generation
        history: Previous iteration results
        iteration: Current iteration number
        outfile_prefix: Prefix for output files

    Returns:
        List of generated categories
    """
    context = f"""
Generate 5 diverse categories for knowledge-intensive questions on the theme: {theme}.
Each category should be a Wikipedia-style category.

Output format: JSON list of dictionaries with keys: id, category, additional_requirement
```json
[
{{"id": "1", "category": "Physics", "additional_requirement": "focus on fundamental concepts"}},
...
]
```
"""
    response = gen_from_prompt(agent_model, context, temperature=0.7, max_tokens=1000)
    categories = extract_json_v2(response, None)
    return categories[0] if categories else []


def _refine_categories_random(theme, agent_model, history, iteration, outfile_prefix='att1'):
    """
    Refine categories based on history.

    Args:
        theme: Theme for question generation
        agent_model: Model to use for generation
        history: Previous iteration results
        iteration: Current iteration number
        outfile_prefix: Prefix for output files

    Returns:
        List of refined categories
    """
    # For simplicity, generate new ones
    return _generate_categories_random(theme, agent_model, history, iteration, outfile_prefix)


def _ask_question_from_wiki(category, agent_model, history, iteration, outfile_prefix='att1'):
    """
    Generate questions from Wikipedia for a category.

    Args:
        category: Category dictionary with 'category' and 'additional_requirement' keys
        agent_model: Model to use for generation
        history: Previous iteration results
        iteration: Current iteration number
        outfile_prefix: Prefix for output files

    Returns:
        List of generated questions
    """
    # Search Wikipedia
    obs, entity, wiki_url = search_step(category['category'])

    if not obs:
        return []

    context = f"""
Based on this Wikipedia content, generate 3 knowledge-intensive questions.
Content: {' '.join(obs[:5])}

Category: {category['category']}
Additional requirement: {category.get('additional_requirement', '')}

Output format: JSON list of dictionaries with keys: id, question, answer, category, difficulty
```json
[
{{"id": "1", "question": "What is X?", "answer": "Y", "category": "{category['category']}", "difficulty": "2"}},
...
]
```
"""
    response = gen_from_prompt(agent_model, context, temperature=0.5, max_tokens=1500)
    questions = extract_json_v2(response, None)
    return questions[0] if questions else []


def gen_qa_pairs_augmented(paragraph, agent_info, additional_req, use_ragas=False, eval_model=None):
    """
    Generate Q&A pairs from a paragraph using RAGAS or LLM-based generation.

    Args:
        paragraph: Text paragraph
        agent_info: Model for generation
        additional_req: Additional requirements
        use_ragas: Whether to use RAGAS for generation (default: False)
        eval_model: Model for RAGAS faithfulness evaluation (optional)

    Returns:
        List of Q&A pairs with optional RAGAS quality metrics
    """
    # Try RAGAS if requested and available
    if use_ragas and is_ragas_available():
        try:
            qa_pairs = generate_qa_with_ragas(paragraph, agent_info, num_questions=3)
            
            # Optionally evaluate faithfulness
            if eval_model:
                for qa in qa_pairs:
                    metrics = evaluate_qa_faithfulness(
                        qa['question'], qa['answer'], paragraph, eval_model
                    )
                    qa['faithfulness'] = metrics['faithfulness']
                    qa['answerability'] = metrics['answerability']
            
            return qa_pairs
        except Exception as e:
            print(f"RAGAS generation failed, falling back to LLM: {e}")
    
    # Fallback to original LLM-based generation
    context = """Conditioned on the wikipedia paragraph, you will generate a few question and answer pairs.
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase.
Make sure that the question you selected is answerable by the given wikipedia paragraph, and make the answer concise. It's recommended to use the exact text from the paragraph as answers.
Make sure that the questions are also answerable by an expert **without the wikipedia paragraph**. For example, dont ask questions that are too specific to the paragraph, like "what are the three locations mentioned in the paragraph?". Or "who's the most famous soldier, according to the paragraph?".

Output format: JSON list of dictionaries with keys: id, question, answer, difficulty
```json
[
{"id": "1", "question": "What is X?", "answer": "Y", "difficulty": "2"},
...
]
```
"""
    context += f"\nParagraph: {paragraph}\nAdditional requirements: {additional_req}"

    response = gen_from_prompt(agent_info, context, temperature=0.0, max_tokens=2000)
    extracted_json = extract_json_v2(response, None)
    return extracted_json[0] if extracted_json else []


def generate_long_questions(line_, agent_info, outfile_prefix, generate_qa_func=gen_qa_pairs_augmented,
                    historical_psg=None, use_ragas=False, eval_model=None):
    """
    Generate questions for a category using Wikipedia content.

    Args:
        line_: Category dictionary
        agent_info: Model for generation
        outfile_prefix: Prefix for output files
        generate_qa_func: Function to generate Q&A pairs
        historical_psg: Historical passages
        use_ragas: Whether to use RAGAS for generation
        eval_model: Model for RAGAS evaluation (optional)

    Returns:
        List of generated questions with metadata
    """
    if historical_psg is None:
        historical_psg = []

    # Search Wikipedia for the category
    obs, entity, wiki_url = search_step(line_['category'])

    if not obs:
        return []

    # Filter and limit observations
    obs = [p for p in obs if len(p.split(" ")) > 2 and len(p.split(".")) > 1]
    obs = obs[:10]  # Limit to first 10 paragraphs

    full_lst = []
    for idx, paragraph in enumerate(obs):
        try:
            # Check if generate_qa_func signature supports RAGAS parameters
            import inspect
            sig = inspect.signature(generate_qa_func)
            if 'use_ragas' in sig.parameters:
                json_questions = generate_qa_func(
                    paragraph, agent_info, line_.get('additional_requirement', ''),
                    use_ragas=use_ragas, eval_model=eval_model
                )
            else:
                json_questions = generate_qa_func(paragraph, agent_info, line_.get('additional_requirement', ''))
        except Exception as e:
            print(f"Error generating questions: {e}")
            continue

        for json_question in json_questions:
            line = copy.deepcopy(line_)
            line['question'] = json_question['question']
            line['gold_answer'] = json_question['answer']
            line['difficulty'] = json_question.get('difficulty', '1')
            line['wiki_entity'] = entity
            line['wiki_url'] = wiki_url
            line['paragraph_idx'] = idx
            
            # Add RAGAS metrics if available
            if 'faithfulness' in json_question:
                line['faithfulness'] = json_question['faithfulness']
            if 'answerability' in json_question:
                line['answerability'] = json_question['answerability']
            if 'ragas_type' in json_question:
                line['ragas_type'] = json_question['ragas_type']
            
            full_lst.append(line)

    return full_lst


def generate_full_qa(theme, agent_info, history, iters, outfile_prefix='att1',
                    historical_psg=None, category_gen_func=None, generate_qa_func=None,
                    acc_target="0.1--0.4"):
    """
    Main function to generate questions for a theme.

    Args:
        theme: Theme for question generation
        agent_info: Model for generation
        history: Previous iteration results
        iters: Current iteration number
        outfile_prefix: Prefix for output files
        historical_psg: Historical passages
        category_gen_func: Function to generate/refine categories
        generate_qa_func: Function to generate questions
        acc_target: Accuracy target

    Returns:
        Updated historical passages
    """
    if historical_psg is None:
        historical_psg = []

    if category_gen_func is None:
        category_gen_func = _refine_categories_targetacc_augmented

    if generate_qa_func is None:
        generate_qa_func = generate_long_questions

    # Generate/refine categories
    category_json = category_gen_func(theme, agent_info, history, iters,
                                     outfile_prefix=outfile_prefix + '.brainstorm',
                                     acc_target=acc_target)

    # Save categories
    with open(f"{outfile_prefix}.categories_augmented.json", "w") as f:
        json.dump(category_json, f, indent=2)

    full_questions = []
    for line_ in category_json[:5]:  # Limit to 5 categories for demo
        questions = generate_qa_func(line_, agent_info, outfile_prefix + f"_{line_['id']}",
                                    historical_psg=historical_psg)
        full_questions.extend(questions)

    # Save questions
    with open(f"{outfile_prefix}.KI_questions.json", "w") as f:
        json.dump(full_questions, f, indent=2)

    # Save thoughts
    context = f"Generated {len(full_questions)} questions for theme '{theme}' in iteration {iters}"
    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)

    return historical_psg


# Import the complex category generation functions from tools
try:
    from tools.AutoBencher.autobencher_v2 import (
        _refine_categories_targetacc_augmented,
        _generate_categories_targetacc_augmented,
        _refine_categories
    )
except ImportError:
    # Fallback implementations
    def _refine_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
        return _generate_categories_random(theme, agent_info, history, iters, outfile_prefix)

    def _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
        return _generate_categories_random(theme, agent_info, history, iters, outfile_prefix)

    def _refine_categories(theme, context, agent_info, history, iters, candidate_lst, outfile_prefix='att1'):
        return _generate_categories_random(theme, agent_info, history, iters, outfile_prefix)