"""
Main execution functions for the Novelty Engine.
High-level functions to run the novelty engine pipeline.
"""

import os
import json
import copy
from types import SimpleNamespace

from .core import generate_full_qa, _refine_categories_targetacc_augmented, generate_long_questions
from .evaluation import solve_and_compare_questions, get_summary_of_results, get_acc_lst
from .variations import apply_variations_to_dataset
from .llm_utils import create_model_from_config


def run_novelty_engine(agent_model, test_model, eval_model, theme="general knowledge",
                    max_iterations=3, acc_target="0.1--0.4", engine="novelty", use_ragas=False, embedding_model=None):
    """
    Run the novelty engine for dynamic benchmark generation.

    Args:
        agent_model: Model for generating questions and evaluating
        test_model: Model to be tested
        eval_model: Model for evaluation
        theme: Theme for question generation
        max_iterations: Maximum number of iterations
        acc_target: Accuracy target for category refinement
        engine: Engine name for file organization
        use_ragas: Whether to use RAGAS for question generation (default: False)
        embedding_model: Embedding model for RAGAS (optional, defaults to SentenceTransformer)

    Returns:
        History of results across iterations
    """
    # Get the project root directory (parent of sspbench package)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(package_dir)
    data_dir = os.path.join(project_root, "data", engine)

    history_dict = []
    historical_psg = []
    
    # Log RAGAS usage
    if use_ragas:
        from .ragas_utils import is_ragas_available
        if is_ragas_available():
            print("🔬 RAGAS integration enabled for question generation")
            if embedding_model is None:
                from .ragas_utils import SentenceTransformerEmbeddings
                embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
                print("Using default SentenceTransformer embedding model")
        else:
            print("⚠️  RAGAS requested but not available. Install with: pip install ragas langchain-core")
            print("    Falling back to standard LLM generation")

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")

        outfile_prefix = os.path.join(data_dir, f"{theme.replace(' ', '_')}_iter_{iteration}")

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(outfile_prefix), exist_ok=True)

        # Summarize previous iterations
        if history_dict:
            summarized_content = get_summary_of_results(
                [item for sublist in history_dict for item in sublist],
                gold_key='gold_answer', verbose=False
            )
            history = [summarized_content]
        else:
            history = ["Initial iteration"]
        print(f"Iteration {iteration}, SUMMARY: {history[0][:100]}...")

        def qa_generator_with_ragas(line_, agent_info, prefix, historical_psg=None):
            return generate_long_questions(
                line_, agent_info, prefix, 
                historical_psg=historical_psg,
                use_ragas=use_ragas,
                eval_model=eval_model if use_ragas else None,
                embedding_model=embedding_model if use_ragas else None
            )

        # Generate questions using existing pipeline
        historical_psg = generate_full_qa(
            theme, agent_model, history, iteration,
            outfile_prefix=outfile_prefix,
            historical_psg=historical_psg,
            category_gen_func=_refine_categories_targetacc_augmented,
            generate_qa_func=qa_generator_with_ragas,
            acc_target=acc_target
        )

        # Load generated questions
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            json_category = json.load(f)
        if len(json_category) == 1:  # remove outer list if needed
            json_category = json_category[0]

        # Apply variations to create more diverse questions (optional)
        # json_category = apply_variations_to_dataset(json_category, agent_model)

        gold_answer_json = copy.deepcopy(json_category)

        # Evaluate the test model
        json_dict = solve_and_compare_questions(
            test_model, eval_model, json_category, gold_answer_json,
            outfile_prefix, 'gold_answer'
        )

        history_dict.append(json_dict)

        # Print results
        verbose_description = get_summary_of_results(json_dict, verbose=False)
        print(verbose_description)

        # Check stopping condition
        acc_lst = get_acc_lst(json_dict)
        avg_acc = sum(acc_lst) / len(acc_lst) if acc_lst else 0
        print(f"Average accuracy: {avg_acc}")

        if avg_acc < 0.1:  # Very low accuracy, good for challenging benchmark
            print("Achieved challenging benchmark (low accuracy)")
        elif avg_acc > 0.8:  # Too easy
            print("Benchmark too easy, will adjust in next iteration")

    return history_dict


def run_autobencher_with_variations(theme="general knowledge", max_iterations=3,
                                   agent_model=None, test_model=None, use_variations=True):
    """
    Run autobencher with question variations.

    Args:
        theme: Theme for question generation
        max_iterations: Maximum number of iterations
        agent_model: Model for generation
        test_model: Model to test
        use_variations: Whether to apply variations

    Returns:
        History of results
    """
    # Get the project root directory (parent of sspbench package)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(package_dir)
    data_dir = os.path.join(project_root, "data", "novelty")

    history = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")

        # Generate categories
        outfile_prefix = os.path.join(data_dir, f'iter_{iteration}')
        os.makedirs(data_dir, exist_ok=True)

        categories = _generate_categories_random(theme, agent_model, history, iteration,
                                                outfile_prefix=outfile_prefix)
        print(f"Generated {len(categories)} categories")

        # Generate base prompts
        all_questions = []
        for cat in categories[:3]:  # Limit for demo
            cat_outfile = os.path.join(data_dir, f'iter_{iteration}_cat_{cat["id"]}')
            questions = _ask_question_from_wiki(cat, agent_model, history, iteration,
                                               outfile_prefix=cat_outfile)
            all_questions.extend(questions)

        print(f"Generated {len(all_questions)} base questions")

        # Apply variations
        if use_variations:
            all_questions = apply_variations_to_dataset(all_questions, agent_model)
            print(f"Applied variations: {len(all_questions)} total questions")

        # Verify and evaluate
        verified_questions = [q for q in all_questions if len(q.get('question', '')) > 10]

        eval_outfile = os.path.join(data_dir, f'iter_{iteration}_eval')
        results = solve_and_compare_questions(
            test_model, agent_model, verified_questions,
            [{'question': q['question'], 'gold_answer': q['answer']} for q in verified_questions],
            outfile_prefix=eval_outfile
        )

        summary = get_summary_of_results(results, gold_key='gold_answer', verbose=False)
        print(summary)

        history.append(results)

        acc_lst = get_acc_lst(results)
        avg_acc = sum(acc_lst) / len(acc_lst) if acc_lst else 0
        print(f"Average accuracy: {avg_acc}")

    return history


def save_benchmarks(questions, filename):
    """
    Save generated questions to benchmarks folder.

    Args:
        questions: List of question dictionaries
        filename: Output filename
    """
    # Get the project root directory (parent of sspbench package)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(package_dir)
    benchmarks_dir = os.path.join(project_root, 'benchmarks', 'autobencher')

    os.makedirs(benchmarks_dir, exist_ok=True)
    filepath = os.path.join(benchmarks_dir, f'{filename}.jsonl')
    with open(filepath, 'w') as f:
        for q in questions:
            json.dump(q, f)
            f.write('\n')
    print(f"Saved {len(questions)} questions to {filepath}")


# def run_autobencher_with_saving(theme="general knowledge", max_iterations=3,
#                                agent_model=None, test_model=None, use_variations=True):
#     """
#     Run autobencher with saving functionality.

#     Args:
#         theme: Theme for question generation
#         max_iterations: Maximum number of iterations
#         agent_model: Model for generation
#         test_model: Model to test
#         use_variations: Whether to apply variations

#     Returns:
#         History of results
#     """
#     history = []

#     for iteration in range(1, max_iterations + 1):
#         print(f"\n=== Iteration {iteration} ===")

#         categories = _generate_categories_random(theme, agent_model, history, iteration,
#                                                 outfile_prefix=f'iter_{iteration}')
#         print(f"Generated {len(categories)} categories")

#         all_questions = []
#         for cat in categories[:3]:
#             questions = _ask_question_from_wiki(cat, agent_model, history, iteration,
#                                                outfile_prefix=f'iter_{iteration}_cat_{cat["id"]}')
#             all_questions.extend(questions)

#         print(f"Generated {len(all_questions)} base questions")

#         if use_variations:
#             all_questions = apply_variations_to_dataset(all_questions, agent_model)
#             print(f"Applied variations: {len(all_questions)} total questions")

#         # Save the generated questions
#         save_benchmarks(all_questions, f'{theme}_iter_{iteration}')

#         verified_questions = [q for q in all_questions if len(q.get('question', '')) > 10]

#         results = solve_and_compare_questions(
#             test_model, agent_model, verified_questions,
#             [{'question': q['question'], 'gold_answer': q['answer']} for q in verified_questions],
#             outfile_prefix=f'iter_{iteration}_eval'
#         )

#         summary = get_summary_of_results(results, gold_key='gold_answer', verbose=False)
#         print(summary)

#         history.append(results)

#         acc_lst = get_acc_lst(results)
#         avg_acc = sum(acc_lst) / len(acc_lst) if acc_lst else 0
#         print(f"Average accuracy: {avg_acc}")

#     return history