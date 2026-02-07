"""
Evaluation utilities for the Novelty Engine.
Functions for testing models and evaluating results.
"""

import json
import os
import tqdm
from collections import defaultdict

from tool_util import extract_json_v2
from .llm_utils import gen_from_prompt

try:
    from ..evaluators import RagasFaithfulnessEvaluator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS evaluators not available, skipping faithfulness evaluation")


def test_taker_inference(test_model_info, problem_json, outfile, bsz=1, temperature=0.01, max_length=50):
    """
    Run inference on a test model for a set of problems.

    Args:
        test_model_info: Model to test
        problem_json: List of problem dictionaries
        outfile: Output file path
        bsz: Batch size
        temperature: Sampling temperature
        max_length: Maximum response length

    Returns:
        List of results
    """
    print(f'Writing to {outfile}')
    out_handle = open(outfile, 'w')
    full_result_lst = []
    batch_lst, line_lst = [], []

    for line in tqdm.tqdm(problem_json):
        line['prompt'] = "Output just with the final answer to the question.\nQuestion:" + line['question'] + "\n" + "Answer:"
        line_lst.append(line)
        batch_lst.append(line['prompt'])
        if len(batch_lst) < bsz:
            continue  # batch not full yet

        responses = gen_from_prompt(model=test_model_info, prompt=batch_lst, temperature=temperature, max_tokens=max_length)

        for line, xx in zip(line_lst, responses):
            line['test_taker_response'] = xx
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
        batch_lst, line_lst = [], []

    if len(batch_lst) > 0:
        responses = gen_from_prompt(model=test_model_info, prompt=batch_lst, temperature=temperature, max_tokens=max_length)
        for line, xx in zip(line_lst, responses):
            line['test_taker_response'] = xx
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
    out_handle.close()
    return full_result_lst


def _generate_lm_answers(question_inputs, test_model_info, outfile_prefix='att1'):
    """
    Generate answers from a language model for given questions.

    Args:
        question_inputs: List of questions or question dictionaries
        test_model_info: Model to use for generation
        outfile_prefix: Prefix for output files

    Returns:
        List of results with answers
    """
    if isinstance(question_inputs, list) or isinstance(question_inputs, dict):
        question_inputs_str = json.dumps(question_inputs, indent=2)
    else:
        assert False

    if isinstance(question_inputs, list) and isinstance(question_inputs[0], list):
        json_dict = question_inputs[0]
    elif isinstance(question_inputs, list):
        json_dict = question_inputs
    else:
        print('question_inputs should be a list.')
        assert False

    full_result_lst = test_taker_inference(test_model_info, json_dict,
                                           outfile=f"{outfile_prefix}.test_taker_inference.json")

    return full_result_lst


def get_acc_lst(json_dict, gold_key="python_answer"):
    """
    Get accuracy list by category.

    Args:
        json_dict: Results dictionary
        gold_key: Key for gold answers

    Returns:
        List of accuracies by category
    """
    category2correct_count = defaultdict(list)
    for line in json_dict:
        category2correct_count[line['category']].append(line['is_correct'])
    acc_lst = []
    for category in category2correct_count:
        acc = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        acc_lst.append(acc)
    return acc_lst


def get_summary_of_results(json_dict, gold_key="python_answer", verbose=False):
    """
    Generate a summary of evaluation results.

    Args:
        json_dict: Results dictionary
        gold_key: Key for gold answers
        verbose: Whether to include detailed output

    Returns:
        String summary of results
    """
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = 'In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n'

    for line in json_dict:
        line['category2'] = f"{line['category']} || {line['wiki_entity']} [{line['additional_requirement']}]" if 'additional_requirement' in line else line['category']
        category2correct_count[line['category2']].append(line['is_correct'])
        category2question[(line['category2'], line['is_correct'])].append(line)

    for category in category2correct_count:
        acc_temp = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        str_summary += f"category: {category}, accuracy: {round(acc_temp, 3)} " \
                       f"|| {sum([1 if x == 'true' else 0 for x in category2correct_count[category]])} out of {len(category2correct_count[category])}" + "\n"

        if verbose:
            str_summary += "# Questions answered correctly:\n"
            for qq in category2question[(category, 'true')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "# Questions answered incorrectly:\n"
            for qq in category2question[(category, 'false')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "\n + ------------------------------------ + \n"

    return str_summary


def summarize_over_history(history_json_dict, gold_key="python_answer", verbose=True):
    """
    Summarize results over multiple iterations.

    Args:
        history_json_dict: List of result dictionaries from different iterations
        gold_key: Key for gold answers
        verbose: Whether to include detailed output

    Returns:
        String summary
    """
    # augment each line of the dictionary with the iteration number.
    for idx, json_dict in enumerate(history_json_dict):
        for line in json_dict:
            line['iteration'] = idx
    # concatenate the dictionaries.
    json_dict = [line for json_dict in history_json_dict for line in json_dict]
    # a summary of the results.
    str_summary = get_summary_of_results(json_dict, gold_key=gold_key, verbose=verbose)
    return str_summary


def summarize_over_history_multi_model(history_json_dict, primary_model_name, gold_key="gold_answer", verbose=True):
    """
    Summarize results for multiple models over history.

    Args:
        history_json_dict: History dictionary with model results
        primary_model_name: Name of primary model to focus on
        gold_key: Key for gold answers
        verbose: Whether to include detailed output

    Returns:
        String summary
    """
    single_model_history = []
    for iter_dict in history_json_dict:
        if primary_model_name in iter_dict:
            single_model_history.append(iter_dict[primary_model_name])
        else:
            single_model_history.append([])
    return summarize_over_history(single_model_history, gold_key=gold_key, verbose=verbose)


def solve_and_compare_questions(test_taker_info, agent_info, question_json, gold_answer, outfile_prefix, gold_ans_key='gold_answer', eval_model=None, embedding_model=None):
    """
    Solve questions with a test model and compare against gold answers.

    Args:
        test_taker_info: Model to test
        agent_info: Model for evaluation
        question_json: Questions to answer
        gold_answer: Gold answers
        outfile_prefix: Prefix for output files
        gold_ans_key: Key for gold answers
        eval_model: Model for RAGAS evaluation (optional)
        embedding_model: Embedding model for RAGAS (optional)

    Returns:
        Results dictionary
    """
    test_taker_output = _generate_lm_answers(question_json, test_taker_info, outfile_prefix=outfile_prefix)
    summary_prev_iteration, history_json = fast_compare_answers(gold_answer, test_taker_output,
                                                                agent_info, outfile_prefix=outfile_prefix,
                                                                gold_ans_key=gold_ans_key,
                                                                eval_model=eval_model,
                                                                embedding_model=embedding_model)
    return history_json


def fast_compare_answers(gold_output, test_taker_output, agent_model_info, outfile_prefix='att1', gold_ans_key='gold_answer', eval_model=None, embedding_model=None):
    """
    Compare predicted answers against gold answers.

    Args:
        gold_output: Gold answers
        test_taker_output: Predicted answers
        agent_model_info: Model for comparison
        outfile_prefix: Prefix for output files
        gold_ans_key: Key for gold answers
        eval_model: Model for RAGAS evaluation (optional)
        embedding_model: Embedding model for RAGAS (optional)

    Returns:
        Tuple of (summary_string, results_dict)
    """
    print("Checking the answers generated by the test taker...")
    print(len(gold_output), len(test_taker_output))
    assert len(gold_output) == len(test_taker_output)

    context_str = """Your goal is to compare the prediction with the gold answer, and judge the correctness of the prediction.
We'd still consider the prediction to be correct if
1. the prediction is semantically the same as the gold answer: formating or different way of reference shouldn't affect correctness. For example, if the gold answer is Jan 21, and the test taker output is 01/21, we would still consider the prediction to be correct. For example, United States and USA refer to the same entity.
2. the prediction refers a broader entity that contains the gold answer. For example, if the gold answer is Beijing, and the test taker output is Asia, we will then consider correctness based on the question.
3. If the question is slightly ambiguous, such that there are multiple correct answers: For example, if the question asks for reasons why something happens, and it could be caused by multiple reasons, we will consider the prediction to be correct if the prediction contains one of the correct answers.

You should output a short and succinct reasoning for the your correctness prediction. Then, you should output delimiter "##" and output "true" if the prediction is correct, and "false" if the prediction is incorrect.
Example Format:
Question: What is 1+1?
pred=2 || gold=2.0
reason: identical numbers ## true
"""

    out_handle = open(f"{outfile_prefix}.compare_answers.jsonl", 'w')
    final_lst = []
    correct_count2 = 0

    for idx, (line_gold, line_pred) in tqdm.tqdm(enumerate(zip(gold_output, test_taker_output))):
        line = {'id': str(idx + 1), 'question': line_gold['question'], 'gold_answer': line_gold[gold_ans_key],
                "test_taker_answer": line_pred['test_taker_response']}

        # add other fields in line_gold to line.
        for k, v in line_gold.items():
            if k not in line:
                line[k] = v

        pred = line_pred['test_taker_response'].strip()
        gold = line_gold[gold_ans_key].strip()
        q_str = f"Question {idx+1}: {line_gold['question']}\npred={pred} || gold={gold}\nreason:"
        context = context_str + q_str

        response = gen_from_prompt(model=agent_model_info, prompt=context, temperature=0.0, max_tokens=3000)

        line['reasons'] = response.strip()
        
        # More robust parsing of correctness judgment
        response_lower = response.lower().strip()
        if '## true' in response_lower or response_lower.endswith('true'):
            line['is_correct'] = 'true'
        elif '## false' in response_lower or response_lower.endswith('false'):
            line['is_correct'] = 'false'
        elif 'true' in response_lower.split('##')[-1].strip():
            line['is_correct'] = 'true'
        elif 'false' in response_lower.split('##')[-1].strip():
            line['is_correct'] = 'false'
        else:
            # Default to false if parsing fails
            print(f"Warning: Could not parse correctness from response: {response}")
            line['is_correct'] = 'false'
        test_taker_line = test_taker_output[idx]
        line['question'] = test_taker_line['question']

        if 'category' in test_taker_line:
            line['category'] = test_taker_line['category']
        else:
            line['category'] = 'None'

        if 'difficulty' in test_taker_line:
            line['difficulty'] = test_taker_line['difficulty']

        if line["is_correct"] == 'true':
            correct_count2 += 1

        print(json.dumps(line), file=out_handle)
        final_lst.append(line)

    json_dict = final_lst
    accuracy = correct_count2 / len(json_dict)
    print("Accuracy: ", accuracy)
    assert len(json_dict) == len(test_taker_output)
    out_handle.close()

    # Add RAGAS evaluation if available
    if RAGAS_AVAILABLE and eval_model is not None and embedding_model is not None:
        print("Running RAGAS evaluation for faithfulness and answer relevance...")
        try:
            ragas_evaluator = RagasFaithfulnessEvaluator(
                eval_model=eval_model,
                embedding_model=embedding_model
            )
            
            # Prepare samples for RAGAS evaluation
            ragas_samples = []
            for item in json_dict:
                # Find the corresponding gold answer entry to get context
                gold_item = None
                for gold in gold_output:
                    if gold.get('question') == item.get('question'):
                        gold_item = gold
                        break
                
                if gold_item and 'context' in gold_item:
                    ragas_samples.append({
                        'id': item['id'],
                        'question': item['question'],
                        'answer': item['test_taker_answer'],
                        'context': gold_item['context'],
                        'ground_truth': item['gold_answer']  # Add ground truth for better evaluation
                    })
            
            if ragas_samples:
                print(f"Evaluating {len(ragas_samples)} samples with RAGAS...")
                ragas_results = ragas_evaluator.evaluate(ragas_samples)
                
                # Add RAGAS scores to the results
                for i, item in enumerate(json_dict):
                    if i < len(ragas_results['results']):
                        ragas_item = ragas_results['results'][i]
                        item['faithfulness'] = ragas_item.get('faithfulness', None)
                        item['answer_relevancy'] = ragas_item.get('answer_relevancy', None)
                
                print("✓ RAGAS evaluation completed and scores added to results")
            else:
                print("⚠️  No samples prepared for RAGAS evaluation")
                
        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
    else:
        print("⚠️  RAGAS evaluation skipped (models not available)")

    with open(f"{outfile_prefix}.compare_answers.json", 'w') as out_handle:
        json.dump(json_dict, out_handle, indent=2)

    str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
    return str_summary, json_dict