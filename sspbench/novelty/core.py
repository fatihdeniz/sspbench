"""
Core functionality for the Novelty Engine.
Main pipeline and category/question generation functions.
"""

import os
import json
import copy
import random
from collections import defaultdict

from ..utils.llm_utils import gen_from_prompt
from .wiki_utils import search_step, search_related_pages
from .config import DEFAULT_JSON_MESSAGE, MAX_JSON_RETRY_ATTEMPTS
from .ragas_utils import generate_qa_with_ragas, is_ragas_available, evaluate_qa_faithfulness
from .json_utils import extract_json_v2
from ..evaluators import DuplicateEvaluator


def _generate_categories_random(theme, agent_model, history, iteration, outfile_prefix='att1', num_categories=5):
    """
    Generate initial categories for a theme.

    Args:
        theme: Theme for question generation
        agent_model: Model to use for generation
        history: Previous iteration results
        iteration: Current iteration number
        outfile_prefix: Prefix for output files
        num_categories: Number of categories to generate

    Returns:
        List of generated categories
    """
    context = f"""
Generate {num_categories} diverse categories for knowledge-intensive questions on the theme: {theme}.
Each category should be a Wikipedia-style category.

Output format: JSON list of dictionaries with keys: id, category, additional_requirement
```json
[
{{"id": "1", "category": "Physics", "additional_requirement": "focus on fundamental concepts"}},
...
]
```

Output the JSON now:
"""
    response = gen_from_prompt(agent_model, context, temperature=0.7, max_tokens=1000)
    categories = extract_json_v2(response, None)
    return categories


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
Generate 3 knowledge-intensive factual questions about the topic below.
Each question must be SELF-CONTAINED — it will be shown to people without any source text.
NEVER reference "the context", "the text", "the passage", or any source document.
Focus on main, well-known facts — not obscure details.

Reference material: {' '.join(obs[:5])}

Category: {category['category']}
Additional requirement: {category.get('additional_requirement', '')}

Output format: JSON list of dictionaries with keys: id, question, answer, category, difficulty
```json
[
{{"id": "1", "question": "What is X?", "answer": "Y", "category": "{category['category']}", "difficulty": "2"}},
...
]
```

Output the JSON now:
"""
    response = gen_from_prompt(agent_model, context, temperature=0.5, max_tokens=1500)
    questions = extract_json_v2(response, None)
    return questions[0] if questions else []


def gen_qa_pairs_augmented(paragraph, agent_info, additional_req, use_ragas=False, eval_model=None, embedding_model=None):
    """
    Generate Q&A pairs from a paragraph using RAGAS or LLM-based generation.

    Args:
        paragraph: Text paragraph
        agent_info: Model for generation
        additional_req: Additional requirements
        use_ragas: Whether to use RAGAS for generation (default: False)
        eval_model: Model for RAGAS faithfulness evaluation (optional)
        embedding_model: Embedding model for RAGAS (optional)

    Returns:
        List of Q&A pairs with optional RAGAS quality metrics
    """
    # Try RAGAS if requested and available
    if use_ragas and is_ragas_available():
        try:
            qa_pairs = generate_qa_with_ragas(paragraph, agent_info, embedding_model, num_questions=3, eval_model=eval_model)
            
            print(f"✓ RAGAS generated {len(qa_pairs)} questions")
            
            # Add context to each QA pair
            for qa in qa_pairs:
                qa['context'] = paragraph
            
            # Optionally evaluate faithfulness
            if eval_model:
                for qa in qa_pairs:
                    metrics = evaluate_qa_faithfulness(
                        qa['question'], qa['answer'], paragraph, eval_model, embedding_model
                    )
                    qa['faithfulness'] = metrics['faithfulness']
                    qa['answer_relevancy'] = metrics['answer_relevancy']
            
            # Deduplicate: LLM picks the clearest question from each near-duplicate group
            if eval_model and len(qa_pairs) > 1:
                dedup = DuplicateEvaluator(eval_model)
                qa_pairs = dedup.select(qa_pairs)

            return qa_pairs
        except Exception as e:
            print(f"✗ RAGAS generation failed, falling back to LLM: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback to original LLM-based generation
    context = """Conditioned on the wikipedia paragraph, you will generate 3 question and answer pairs.
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase.
Make sure that the question you selected is answerable by the given wikipedia paragraph, and make the answer concise. It's recommended to use the exact text from the paragraph as answers.
Make sure that the questions are also answerable by an expert **without the wikipedia paragraph**. For example, dont ask questions that are too specific to the paragraph, like "what are the three locations mentioned in the paragraph?". Or "who's the most famous soldier, according to the paragraph?".
IMPORTANT: The question must NOT contain the answer. For example, do NOT ask "When did X happen in 1893?" when the answer is "1893". The purpose is to TEST whether the model knows the answer — it must be possible to answer incorrectly.

Output format: JSON list of dictionaries with keys: id, question, answer
```json
[
{"id": "1", "question": "What is X?", "answer": "Y"},
...
]
```
"""
    context += f"\nParagraph: {paragraph}\nAdditional requirements: {additional_req}\n\nOutput the JSON now:"

    response = gen_from_prompt(agent_info, context, temperature=0.0, max_tokens=2000)
    extracted_json = extract_json_v2(response, None)
    qa_pairs = extracted_json[0] if extracted_json else []
    
    # Add context to each QA pair for fallback generation too
    for qa in qa_pairs:
        qa['context'] = paragraph

    if eval_model and len(qa_pairs) > 1:
        dedup = DuplicateEvaluator(eval_model)
        qa_pairs = dedup.select(qa_pairs)

    return qa_pairs


def generate_long_questions(line_, agent_info, outfile_prefix, generate_qa_func=gen_qa_pairs_augmented,
                    historical_psg=None, use_ragas=False, eval_model=None, embedding_model=None):
    """
    Generate questions for a category using Wikipedia content.

    Args:
        line_: Category dictionary
        agent_info: Model for generation
        outfile_prefix: Prefix for output files
        generate_qa_func: Function to generate Q&A pairs
        historical_psg: Historical passages
        use_ragas: Whether to use RAGAS for generation
        eval_model: Model for RAGAS faithfulness evaluation (optional)
        embedding_model: Embedding model for RAGAS (optional)

    Returns:
        List of generated questions with metadata
    """
    if historical_psg is None:
        historical_psg = []

    # Search Wikipedia for the category
    obs, entity, wiki_url = search_step(line_['category'])

    if not obs:
        print(f"⚠️  No Wikipedia content found for '{line_['category']}'")
        return []

    # Filter and limit observations
    obs = [p for p in obs if len(p.split(" ")) > 2 and len(p.split(".")) > 1]
    obs = obs[:5]
    
    if not obs:
        print(f"⚠️  All Wikipedia content filtered out for '{line_['category']}'")
        return []

    combined_paragraph = "\n\n".join(obs)
    print(f"   Wikipedia content: {len(combined_paragraph)} chars from {len(obs)} paragraphs")
    
    full_lst = []
    try:
        import inspect
        sig = inspect.signature(generate_qa_func)
        if 'use_ragas' in sig.parameters:
            json_questions = generate_qa_func(
                combined_paragraph, agent_info, line_.get('additional_requirement', ''),
                use_ragas=use_ragas, eval_model=eval_model, embedding_model=embedding_model
            )
        else:
            json_questions = generate_qa_func(combined_paragraph, agent_info, line_.get('additional_requirement', ''))
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

    for json_question in json_questions:
        line = copy.deepcopy(line_)
        line['question'] = json_question['question']
        line['gold_answer'] = json_question['answer']
        line['wiki_entity'] = entity
        line['wiki_url'] = wiki_url
        # line['paragraph_idx'] = 0  # Single combined context
        
        # Add context if available
        if 'context' in json_question:
            line['context'] = json_question['context']
        
        # Add RAGAS metrics if available
        if 'faithfulness' in json_question:
            line['faithfulness'] = json_question['faithfulness']
        if 'answer_relevancy' in json_question:
            line['answer_relevancy'] = json_question['answer_relevancy']
        if 'answerability' in json_question:
            line['answerability'] = json_question['answerability']
        if 'ragas_type' in json_question:
            line['ragas_type'] = json_question['ragas_type']
        
        # Add quality check metadata if available
        if 'quality' in json_question:
            line['quality'] = json_question['quality']
        
        full_lst.append(line)

    return full_lst


def generate_full_qa(theme, agent_info, history, iters, outfile_prefix='att1',
                    historical_psg=None, category_gen_func=None, generate_qa_func=None,
                    acc_target="0.1--0.4", max_categories: int = 5):
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
                                     acc_target=acc_target, num_categories=max_categories)

    # Save categories
    with open(f"{outfile_prefix}.categories_augmented.json", "w") as f:
        json.dump(category_json, f, indent=2)

    full_questions = []
    for line_ in category_json[:max_categories]:
        print(f"\n🔍 Generating questions for category {line_['id']}: {line_['category']}")
        questions = generate_qa_func(line_, agent_info, outfile_prefix + f"_{line_['id']}",
                                    historical_psg=historical_psg)
        print(f"   Generated {len(questions)} questions for this category")
        full_questions.extend(questions)

    # Save questions
    with open(f"{outfile_prefix}.KI_questions.json", "w") as f:
        json.dump(full_questions, f, indent=2)

    # Save thoughts
    context = f"Generated {len(full_questions)} questions for theme '{theme}' in iteration {iters}"
    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)

    return historical_psg


# Category generation functions

def _refine_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5", num_categories=5):
    category_json = _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    # given the json_lst, refine the categories to achieve the target accuracy.
    # Shuffle each seed's Wikipedia results before capping so we sample
    # across the full breadth of related pages, not just the top-ranked
    # (most narrowly related) ones.  Then deduplicate and shuffle the
    # combined pool to eliminate positional bias in the refine prompt.
    full_cat_lst = []
    seen = set()
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        # random.shuffle(cat_lst)
        for cat in cat_lst:
            if cat not in seen:
                seen.add(cat)
                full_cat_lst.append(cat)
    random.shuffle(full_cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are likely to achieve the target accuracy of {ACC_TARGET}.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) likely to obtain the target accuracy of {ACC_TARGET}, you can judge this based on the accuracy statistics from previous iterations. and (3) salient and cover important topics.
IMPORTANT: The selected categories MUST be diverse and cover different sub-domains of THEME. Do NOT select multiple categories that are sub-topics of each other or belong to the same narrow area. Spread your selections across as many distinct branches of THEME as possible.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "focus on major, well-known facts and events, avoid obscure or niche details". That way, the questions will focus on important, widely-known information.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
```
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top {NUM_CATEGORIES} categories from the list, that are most likely to achieve the target accuracy level, while still being relevant and salient. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    context = context.replace("{NUM_CATEGORIES}", str(num_categories))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')

def _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))
    agent_model = agent_info
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that achieve the target accuracy of {ACC_TARGET}.
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "focus on major, well-known facts and events, avoid obscure or niche details". That way, the questions will focus on important, widely-known information.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage. Coming up with new categories that can are likely to achieve the target accuracy level.
2. For example, If you find the model now lacks categories of 0.3 -- 0.5 accuracy, you should come up with more categories that would yield accuracy in that range, by either reducing the difficulty of questions that achieve lower accuracy (via subcategory or via additional requirement), or increasing the difficulty of questions that achieve higher accuracy.
3. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters)
    
    # Add explicit instruction to output JSON immediately
    context += "\n\nBased on the criteria above, output the categories in JSON format now (no explanations, just the JSON block):\n"

    for attempt in range(MAX_JSON_RETRY_ATTEMPTS):
        try:
            response = gen_from_prompt(agent_model, context, temperature=0.0, max_tokens=8000, system_prompt=DEFAULT_JSON_MESSAGE)

            with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
                out_handle.write(context)
                out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                out_handle.write(response)

            extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
            return extracted_json
        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  Attempt {attempt + 1}/{MAX_JSON_RETRY_ATTEMPTS}: JSON parsing failed - {str(e)[:100]}")
            if attempt == MAX_JSON_RETRY_ATTEMPTS - 1:  # Last attempt
                print(f"✗ Failed to parse JSON after {MAX_JSON_RETRY_ATTEMPTS} attempts, returning empty list")
                return []
            # Add stronger prompt for next attempt
            context += "\n\nIMPORTANT: Output ONLY the JSON block, starting with ```json"

def _refine_categories(theme, context, agent_info, history, iters, candidate_lst, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))
    agent_model = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1." + "Here are the category candidates to select from (delimited by ||): " + " || ".join(candidate_lst) + "\n"
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters) + "Here are the category candidates to select from (delimited by ||): " + "||".join(candidate_lst) + "\n"
    
    context += "\nBased on the criteria above, output the selected categories in JSON format now (no explanations, just the JSON block):\n"

    # Dynamic token budget matching brainstorm (each category ≈ 80-100 tokens)
    # Cap at 32768 to stay within Azure OpenAI completion-token limits
    refine_max_tokens = min(32768, max(4000, len(candidate_lst) * 200))
    
    for attempt in range(MAX_JSON_RETRY_ATTEMPTS):
        try:
            response = gen_from_prompt(agent_model, context, temperature=0.0, max_tokens=refine_max_tokens, system_prompt=DEFAULT_JSON_MESSAGE)

            with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
                out_handle.write(context)
                out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                out_handle.write(response)

            extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
            return extracted_json
        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  Attempt {attempt + 1}/{MAX_JSON_RETRY_ATTEMPTS}: JSON parsing failed - {str(e)[:100]}")
            if attempt == MAX_JSON_RETRY_ATTEMPTS - 1:  # Last attempt
                print(f"✗ Failed to parse JSON after {MAX_JSON_RETRY_ATTEMPTS} attempts, returning empty list")
                return []
            # Add stronger prompt for next attempt
            context += "\n\nIMPORTANT: Output ONLY the JSON block, starting with ```json"