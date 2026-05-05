"""
Main execution functions for the Novelty Engine.

Factual QA Benchmark Generation Pipeline
=========================================
The pipeline generates challenging factual question-answer benchmarks
through an iterative, Wikipedia-grounded process:

1. **Summarize history** — Aggregate results from prior iterations so the
   LLM can adjust category selection to approach the target accuracy.
2. **Brainstorm categories** — Ask the LLM to propose diverse knowledge
   categories (e.g. "Ancient Philosophers", "Second World War") under
   the given theme, targeting a specified accuracy range.
3. **Search Wikipedia for subcategories** — For each brainstormed
   category, query the Wikipedia API to discover related page titles
   (subcategories / candidate topics).  Deduplicate, shuffle, and cap
   at MAX_REFINE_CANDIDATES.
4. **Refine & select categories** — Feed the candidate list back to the
   LLM and ask it to pick the top-N categories most likely to hit the
   target accuracy, while maximizing diversity.
5. **Fetch Wikipedia content** — For each selected category, search
   Wikipedia for the article, extract and clean paragraphs.
6. **Generate QA pairs** — Prompt the LLM (or RAGAS) to produce
   question-answer pairs grounded in the Wikipedia paragraphs.  Apply
   deduplication across generated questions.
7. **Answer-leak filter** — Remove questions whose text contains the
   gold answer (trivially answerable).
8. **Scope filter** — An evaluation LLM judges whether each question
   falls within the intended theme/scope; out-of-scope questions are
   dropped.
9. **Salience filter** — An evaluation LLM scores each remaining
   question on salience (importance / notability); low-salience
   questions are dropped.
10. **Save annotated questions** — Write all questions (including
    filtered ones) with their scope and salience annotations to
    ``*.KI_questions_annotated.json``.
11. **Test-taker evaluation** — The test model answers the surviving
    questions; answers are compared against gold answers by the
    evaluation LLM.
12. **Record & iterate** — Store results, compute accuracy, and feed
    the summary back into step 1 for the next iteration.
"""

import os
import json
import copy
from types import SimpleNamespace

from .core import generate_full_qa, _refine_categories_targetacc_augmented, generate_long_questions
from .evaluation import solve_and_compare_questions, get_summary_of_results, get_acc_lst
from ..utils.llm_utils import create_model_from_config
from .config import FACTUALITY_QA_SCOPE_JUDGE_PROMPT
from ..evaluators import ScopeEvaluator, SalienceEvaluator, AnswerLeakageEvaluator


def run_novelty_engine(agent_model, test_models, eval_model, theme="general knowledge",
                    max_iterations=3, acc_target="0.1--0.4", engine="novelty", use_ragas=False, embedding_model=None, output_dir=None,
                    start_iteration=1):
    """
    Run the novelty engine for dynamic benchmark generation.

    Args:
        agent_model: Model for generating questions and evaluating
        test_models: Dict of {model_name: model} for the steering panel.
            Each model is evaluated independently on every generated item.
            Legacy callers may pass a single model instance, which is
            wrapped automatically.
        eval_model: Model for evaluation
        theme: Theme for question generation
        max_iterations: Maximum number of iterations
        acc_target: Accuracy target for category refinement
        engine: Engine name for file organization
        use_ragas: Whether to use RAGAS for question generation (default: False)
        embedding_model: Embedding model for RAGAS (optional, defaults to SentenceTransformer)
        output_dir: Custom output directory (optional, defaults to data/<engine>)
        start_iteration: Iteration number to start from (default: 1).
            When > 1, loads compare_answers.json from prior iterations
            to reconstruct the history summary for category refinement.

    Returns:
        History of results across iterations
    """
    # Backward compatibility: wrap a single model in a dict
    if not isinstance(test_models, dict):
        test_models = {"test_model": test_models}
    # Get the project root directory (parent of sspbench package)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(package_dir)
    
    if output_dir:
        data_dir = output_dir
    else:
        data_dir = os.path.join(project_root, "data", "generated", engine)

    history_dict = []
    historical_psg = []

    # ── Restore history from prior iterations when resuming ──────────
    if start_iteration > 1:
        theme_slug = theme.replace(' ', '_')
        print(f"Resuming from iteration {start_iteration} — loading history from iterations 1..{start_iteration - 1}")
        for prev_iter in range(1, start_iteration):
            prev_prefix = os.path.join(data_dir, f"{theme_slug}_iter_{prev_iter}")
            ca_path = f"{prev_prefix}.compare_answers.json"
            if os.path.isfile(ca_path):
                with open(ca_path, "r") as fh:
                    prev_results = json.load(fh)
                history_dict.append(prev_results)
                print(f"  ✓ Loaded iteration {prev_iter}: {len(prev_results)} results from {os.path.basename(ca_path)}")
            else:
                print(f"  ⚠  Missing {os.path.basename(ca_path)} — skipping iteration {prev_iter}")
                history_dict.append([])
        print(f"History restored: {sum(len(h) for h in history_dict)} total results from {len(history_dict)} iterations")

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

    scope_evaluator = ScopeEvaluator(eval_model, prompt_template=FACTUALITY_QA_SCOPE_JUDGE_PROMPT) if eval_model else None
    salience_evaluator = SalienceEvaluator(eval_model) if eval_model else None
    leak_evaluator = AnswerLeakageEvaluator()

    for iteration in range(start_iteration, max_iterations + 1):
        print(f"\n=== Iteration {iteration} ===")

        outfile_prefix = os.path.join(data_dir, f"{theme.replace(' ', '_')}_iter_{iteration}")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(outfile_prefix), exist_ok=True)

        # Step 1: summarize previous iterations
        if history_dict:
            summarized_content = get_summary_of_results(
                [item for sublist in history_dict for item in sublist],
                gold_key='gold_answer', verbose=False
            )
            history = [summarized_content]
        else:
            history = ["Initial iteration"]
        print(f"Iteration {iteration}, SUMMARY: {history[0][:200]}...")

        def qa_generator_with_ragas(line_, agent_info, prefix, historical_psg=None):
            return generate_long_questions(
                line_, agent_info, prefix, 
                historical_psg=historical_psg,
                use_ragas=use_ragas,
                eval_model=eval_model if use_ragas else None,
                embedding_model=embedding_model if use_ragas else None
            )

        # Step 2: generate questions using existing pipeline
        historical_psg = generate_full_qa(
            theme, agent_model, history, iteration,
            outfile_prefix=outfile_prefix,
            historical_psg=historical_psg,
            category_gen_func=_refine_categories_targetacc_augmented,
            generate_qa_func=qa_generator_with_ragas,
            acc_target=acc_target,
            max_categories=10
        )

        # Step 3: load generated questions
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            json_category = json.load(f)
        if len(json_category) == 1 and isinstance(json_category[0], list):
            json_category = json_category[0]

        original_question_count = len(json_category)

        if json_category and isinstance(json_category[0], dict):
            json_category = leak_evaluator.filter(json_category)
        else:
            json_category = []

        if len(json_category) < original_question_count:
            print(f"  Quality filters: {original_question_count} → {len(json_category)} questions ")

        # Step 4: Annotate scope using the evaluation model if available
        all_annotated_questions = copy.deepcopy(json_category)

        if scope_evaluator:
            try:
                scope_results = scope_evaluator.evaluate(json_category)["results"]
                filtered_questions = []
                for question_dict, scope_info in zip(json_category, scope_results):
                    question_dict["in_scope"] = scope_info.get("in_scope")
                    question_dict["in_scope_reason"] = scope_info.get("in_scope_reason", "")
                    question_dict["scope_raw_response"] = scope_info.get("raw_response", "")
                    if question_dict["in_scope"]:
                        filtered_questions.append(question_dict)

                all_annotated_questions = copy.deepcopy(json_category)
                removed = len(json_category) - len(filtered_questions)
                if removed:
                    print(f"Filtered out {removed} out-of-scope question(s)")
                json_category = filtered_questions

                print("Scope summary:", {
                    "total_before_scope": original_question_count,
                    "in_scope": len(filtered_questions),
                    "filtered_out": removed,
                })
            except Exception as scope_error:
                print(f"⚠️  Scope evaluation failed: {scope_error}")

        # Step 4b: Annotate salience using the evaluation model if available
        if salience_evaluator and json_category:
            try:
                before_salience = len(json_category)
                salience_output = salience_evaluator.evaluate(json_category)
                salience_results = salience_output["results"]

                filtered_questions = []
                for question_dict, sal_info in zip(json_category, salience_results):
                    question_dict["salience_score"] = sal_info.get("salience_score", 0)
                    question_dict["salience_explanation"] = sal_info.get("salience_explanation", "")
                    question_dict["is_salient"] = sal_info.get("is_salient", False)
                    if sal_info.get("is_salient"):
                        filtered_questions.append(question_dict)

                sal_by_question = {q["question"]: q for q in json_category}
                for q in all_annotated_questions:
                    sal_q = sal_by_question.get(q["question"])
                    if sal_q:
                        for k in ("salience_score", "salience_explanation", "is_salient"):
                            q[k] = sal_q[k]

                removed = before_salience - len(filtered_questions)
                if removed:
                    print(f"Filtered out {removed} low-salience question(s)")
                json_category = filtered_questions

                print("Salience summary:", salience_output.get("summary", {}))
            except Exception as salience_error:
                print(f"⚠️  Salience evaluation failed: {salience_error}")

        with open(f"{outfile_prefix}.KI_questions_annotated.json", "w") as f:
            json.dump(all_annotated_questions, f, indent=2)

        gold_answer_json = copy.deepcopy(json_category)

        if not gold_answer_json:
            print("No questions remain after scope/salience filtering; skipping iteration")
            history_dict.append([])
            continue

        # Step 5: Evaluate each steering panel model
        iteration_results = {}
        for model_name, test_model in test_models.items():
            print(f"\n--- Evaluating: {model_name} ---")
            model_prefix = f"{outfile_prefix}.{model_name.replace('/', '_')}"
            model_json_dict = solve_and_compare_questions(
                test_model, eval_model,
                copy.deepcopy(json_category), copy.deepcopy(gold_answer_json),
                model_prefix, 'gold_answer',
                eval_model=eval_model if use_ragas else None,
                embedding_model=embedding_model if use_ragas else None
            )
            iteration_results[model_name] = model_json_dict

            acc_lst = get_acc_lst(model_json_dict)
            avg_acc = sum(acc_lst) / len(acc_lst) if acc_lst else 0
            print(f"  {model_name} accuracy: {avg_acc:.3f}")

        # Merge results: each question gets per-model correctness
        merged = _merge_steering_results(gold_answer_json, iteration_results)

        # Write merged results
        import json as _json
        with open(f"{outfile_prefix}.compare_answers.json", "w") as f:
            _json.dump(merged, f, indent=2)

        history_dict.append(merged)

        # Print aggregate summary
        verbose_description = get_summary_of_results(merged, verbose=False)
        print(verbose_description)

        n_models = len(test_models)
        avg_accs = []
        for model_name, model_results in iteration_results.items():
            acc_lst = get_acc_lst(model_results)
            avg_accs.append(sum(acc_lst) / len(acc_lst) if acc_lst else 0)
        overall_avg = sum(avg_accs) / len(avg_accs) if avg_accs else 0
        print(f"Average accuracy across {n_models} model(s): {overall_avg:.3f}")

    return history_dict


def _merge_steering_results(gold_questions, iteration_results):
    """Merge per-model evaluation results into a single list.

    Each question gets additional fields:
      - model_results: {model_name: {is_correct, test_taker_answer}}
      - difficulty: fraction of models that answered incorrectly
      - separability: variance of binary correctness across models
    """
    model_names = list(iteration_results.keys())
    merged = []
    for q_idx, gold_q in enumerate(gold_questions):
        item = copy.deepcopy(gold_q)
        model_res = {}
        correct_count = 0
        for mname in model_names:
            mresults = iteration_results[mname]
            if q_idx < len(mresults):
                mr = mresults[q_idx]
                is_correct = mr.get("is_correct", "false")
                model_res[mname] = {
                    "is_correct": is_correct,
                    "test_taker_answer": mr.get("test_taker_answer", ""),
                }
                if is_correct == "true":
                    correct_count += 1
            else:
                model_res[mname] = {"is_correct": "false", "test_taker_answer": ""}

        n = len(model_names)
        p = correct_count / n if n > 0 else 0
        item["model_results"] = model_res
        item["difficulty"] = 1.0 - p
        item["separability"] = p * (1.0 - p)
        # For backward compatibility with summary functions
        item["is_correct"] = "true" if correct_count > n / 2 else "false"
        merged.append(item)
    return merged