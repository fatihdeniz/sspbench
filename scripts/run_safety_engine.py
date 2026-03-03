#!/usr/bin/env python3
"""
run_safety_engine.py — Standalone script to run the Safety Novelty Engine.

Runs the full safety novelty engine with the same defaults as the
safety_main.ipynb notebook.  Can be executed with no arguments:

    python scripts/run_safety_engine.py

All parameters can be overridden via command-line flags.  See --help.

Default configuration (mirrors safety_main.ipynb)
--------------------------------------------------
  Agent model   : gpt-4.1-mini-aixamine (Azure OpenAI)
  Test model    : google/gemma-2-2b-it  (local HuggingFace)
  Eval model    : gpt-oss              (local vLLM endpoint)
  Theme         : LLM safety alignment
  Iterations    : 8
  Refusal target: 0.1--0.5
  Categories    : 15 per iteration
  Prompts/cat   : 8
  Mutations/src : 3
  Quality gate  : 6  (out of 10)
  Existing corpus: data/curation/safety-alignment/existing_prompts.jsonl
  Output dir    : data/safety_novelty
"""

import argparse
import json
import os
import sys

# ── Path setup ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "scripts" else _HERE
sys.path.insert(0, _PROJECT_ROOT)

# Load .env file (API keys, endpoints, etc.)
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# Set dummy key to prevent RAGAS from requiring real OpenAI credentials
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-ragas")

# ── Default model configs (same as safety_main.ipynb) ──────────────────
DEFAULT_AGENT_CONFIG = {
    "type": "openai",
    "model": os.environ.get("JUDGE_EXTERNAL_MODEL", "gpt-4.1-mini-aixamine"),
    "api_url": os.environ.get("JUDGE_EXTERNAL_ENDPOINT", ""),
    "api_token": os.environ.get("JUDGE_EXTERNAL_TOKEN", ""),
    "api_version": os.environ.get("JUDGE_EXTERNAL_VERSION", "2024-12-01-preview"),
}

DEFAULT_TEST_CONFIG = {
    "type": "huggingface",
    "model": os.path.join(
        os.path.expanduser("~"),
        "projects", "aiXamine", "airflow-tasks", "models", "google_gemma-2-2b-it",
    ),
}

DEFAULT_EVAL_CONFIG = {
    "type": "openai",
    "model": os.environ.get("EVAL_MODEL", "gpt-oss"),
    "api_url": os.environ.get("EVAL_ENDPOINT", "http://10.4.8.217:8000/v1"),
    "api_token": os.environ.get("EVAL_TOKEN", "abc123"),
    "api_version": os.environ.get("JUDGE_EXTERNAL_VERSION", "2024-12-01-preview"),
}

# ── Default engine parameters ──────────────────────────────────────────
DEFAULT_THEME = "LLM safety alignment"
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_REFUSAL_TARGET = "0.1--0.5"
DEFAULT_NUM_CATEGORIES = 15
DEFAULT_NUM_PROMPTS_PER_CATEGORY = 8
DEFAULT_QUALITY_THRESHOLD = 6
DEFAULT_MUTATIONS_PER_SOURCE = 3
DEFAULT_ENGINE = "safety_novelty"


def _load_existing_prompts(curated_dir: str):
    """Load existing safety prompts from curated JSONL (same as notebook)."""
    prompts_path = os.path.join(curated_dir, "existing_prompts.jsonl")
    if not os.path.isfile(prompts_path):
        print(f"⚠  Existing prompts file not found: {prompts_path}")
        return []

    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} existing safety prompts from {prompts_path}")
    return prompts


def _build_model_config(override_model, override_type, default_config: dict) -> dict:
    """Return a model config dict, applying CLI overrides if provided."""
    if override_model is not None:
        cfg = {"model": override_model, "type": override_type or "openrouter"}
        return cfg
    return default_config


def run_seeds(args):
    """Mode: extract seed topics and existing prompts from aiXamine."""
    from sspbench.safety.safety_seeds import write_seed_artifacts

    prompt_dir = args.seed_prompt_dir
    if not prompt_dir:
        prompt_dir = os.path.join(
            _PROJECT_ROOT, "..", "aiXamine",
            "airflow-tasks", "services", "safety-alignment", "prompts",
        )

    if not os.path.isdir(prompt_dir):
        print(f"ERROR: Prompt directory not found: {prompt_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "data", "curation", "safety-alignment")
    write_seed_artifacts(prompt_dir, output_dir)
    print("\n✅  Seed extraction complete.")


def run_generate(args):
    """Mode: generate safety prompts without evaluating a test model."""
    from sspbench.utils.llm_utils import create_model_from_config
    from sspbench.safety.safety_core import generate_full_safety_prompts, refine_safety_categories

    agent_cfg = _build_model_config(args.agent_model, args.model_type, DEFAULT_AGENT_CONFIG)
    eval_cfg = _build_model_config(args.eval_model, args.model_type, DEFAULT_EVAL_CONFIG)

    agent_model = create_model_from_config(agent_cfg)
    eval_model = create_model_from_config(eval_cfg)

    # Load existing prompts from curated JSONL
    curated_dir = os.path.join(_PROJECT_ROOT, "data", "curation", "safety-alignment")
    existing_prompts = _load_existing_prompts(curated_dir)

    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "data", DEFAULT_ENGINE)
    os.makedirs(output_dir, exist_ok=True)

    all_prompts = []
    history_text = ["Initial iteration"]

    start = getattr(args, 'start_iteration', 1)

    # Restore history from prior iterations when resuming
    if start > 1:
        for prev_iter in range(1, start):
            prev_prefix = os.path.join(output_dir, f"safety_iter_{prev_iter}")
            summary_path = f"{prev_prefix}.iteration_summary.json"
            if os.path.isfile(summary_path):
                with open(summary_path, "r") as fh:
                    prev_summary = json.load(fh)
                history_text.append(prev_summary.get("summary", f"Iteration {prev_iter}: no summary"))
                print(f"  ✓ Loaded iteration {prev_iter} summary")

    for iteration in range(start, args.max_iterations + 1):
        outfile_prefix = os.path.join(output_dir, f"safety_iter_{iteration}")
        prompts = generate_full_safety_prompts(
            theme=args.theme,
            agent_model=agent_model,
            history=history_text,
            iteration=iteration,
            outfile_prefix=outfile_prefix,
            acc_target=args.refusal_target,
            max_categories=args.num_categories,
            num_prompts_per_category=args.num_prompts,
            existing_prompts=existing_prompts,
            category_gen_func=refine_safety_categories,
            eval_model=eval_model,
            quality_threshold=args.quality_threshold,
            mutations_per_source=args.mutations_per_source,
        )
        all_prompts.extend(prompts)
        history_text.append(f"Iteration {iteration}: generated {len(prompts)} prompts.")

    # Save combined output
    combined_path = os.path.join(output_dir, "generated_safety_prompts.jsonl")
    with open(combined_path, "w") as fh:
        for p in all_prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n✅  Generated {len(all_prompts)} prompts → {combined_path}")


def run_full(args):
    """Mode: full pipeline — generate, evaluate, iterate."""
    from sspbench.utils.llm_utils import create_model_from_config
    from sspbench.safety.safety_engine import run_safety_novelty_engine, save_safety_benchmark

    agent_cfg = _build_model_config(args.agent_model, args.model_type, DEFAULT_AGENT_CONFIG)
    test_cfg = _build_model_config(args.test_model, args.model_type, DEFAULT_TEST_CONFIG)
    eval_cfg = _build_model_config(args.eval_model, args.model_type, DEFAULT_EVAL_CONFIG)

    print(f"Agent config: {agent_cfg.get('model')}")
    print(f"Test config:  {test_cfg.get('model')}")
    print(f"Eval config:  {eval_cfg.get('model')}")

    agent_model = create_model_from_config(agent_cfg)
    test_model = create_model_from_config(test_cfg)
    eval_model = create_model_from_config(eval_cfg)

    # Load existing prompts from curated JSONL (same as notebook cell 6)
    curated_dir = os.path.join(_PROJECT_ROOT, "data", "curation", "safety-alignment")
    existing_prompts = _load_existing_prompts(curated_dir)

    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "data", DEFAULT_ENGINE)

    results = run_safety_novelty_engine(
        agent_model=agent_model,
        test_model=test_model,
        eval_model=eval_model,
        theme=args.theme,
        max_iterations=args.max_iterations,
        refusal_target=args.refusal_target,
        num_categories=args.num_categories,
        num_prompts_per_category=args.num_prompts,
        quality_threshold=args.quality_threshold,
        existing_prompts=existing_prompts,
        output_dir=output_dir,
        engine=args.engine,
        mutations_per_source=args.mutations_per_source,
        start_iteration=args.start_iteration,
    )

    # Save as benchmark (same as notebook cell 20)
    if results["all_prompts"]:
        benchmark_path = save_safety_benchmark(
            results["all_prompts"],
            filename=f"{args.theme.replace(' ', '_')}_benchmark",
        )
        print(f"Benchmark saved to: {benchmark_path}")

    print(f"\n✅  Full pipeline complete — {len(results['all_prompts'])} total prompts generated and evaluated.")

    # Print per-iteration summary
    for i, summary in enumerate(results.get("summaries", []), 1):
        print(f"\n{'='*60}")
        print(f"Iteration {i}")
        print(f"{'='*60}")
        print(summary[:300])


def main():
    parser = argparse.ArgumentParser(
        description="Safety Novelty Engine — Adaptive safety benchmark generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["seeds", "generate", "full"], default="full",
        help="Execution mode (default: full)",
    )

    # Model overrides — if not provided, notebook defaults are used
    parser.add_argument("--agent-model", default=None,
                        help="Override agent model name (default: gpt-4.1-mini-aixamine)")
    parser.add_argument("--test-model", default=None,
                        help="Override test model name (default: local gemma-2-2b-it)")
    parser.add_argument("--eval-model", default=None,
                        help="Override eval model name (default: gpt-oss)")
    parser.add_argument("--model-type", default=None,
                        help="Model backend type for overridden models (default: uses notebook configs)")
    parser.add_argument('--cuda-device', default='4',
                        help='CUDA device ID (default: 4)')

    # Data paths
    parser.add_argument("--seed-prompt-dir", default=None,
                        help="Path to aiXamine safety-alignment/prompts directory (only for seeds mode)")
    parser.add_argument("--output-dir", default=None,
                        help=f"Output directory (default: data/{DEFAULT_ENGINE})")

    # Engine params — defaults match safety_main.ipynb
    parser.add_argument("--theme", default=DEFAULT_THEME,
                        help=f"Generation theme (default: {DEFAULT_THEME})")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"Number of generate-evaluate-refine loops (default: {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument('--start-iteration', type=int, default=1,
                        help='Iteration to resume from (default: 1). '
                             'Loads history from prior iterations automatically. '
                             'E.g. --start-iteration 4 --max-iterations 8 runs iters 4-8.')
    parser.add_argument("--refusal-target", default=DEFAULT_REFUSAL_TARGET,
                        help=f"Target refusal rate range (default: {DEFAULT_REFUSAL_TARGET})")
    parser.add_argument("--num-categories", type=int, default=DEFAULT_NUM_CATEGORIES,
                        help=f"Categories per iteration (default: {DEFAULT_NUM_CATEGORIES})")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS_PER_CATEGORY,
                        help=f"Prompts per category (default: {DEFAULT_NUM_PROMPTS_PER_CATEGORY})")
    parser.add_argument("--quality-threshold", type=int, default=DEFAULT_QUALITY_THRESHOLD,
                        help=f"Min quality score 0-10 (default: {DEFAULT_QUALITY_THRESHOLD})")
    parser.add_argument("--mutations-per-source", type=int, default=DEFAULT_MUTATIONS_PER_SOURCE,
                        help=f"Mutation operators per mined source prompt (default: {DEFAULT_MUTATIONS_PER_SOURCE})")
    parser.add_argument("--engine", default=DEFAULT_ENGINE,
                        help=f"Engine name for directory naming (default: {DEFAULT_ENGINE})")

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Theme: {args.theme}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Start iteration: {args.start_iteration}")
    print(f"Refusal target: {args.refusal_target}")
    print(f"Categories: {args.num_categories}")
    print(f"Prompts/category: {args.num_prompts}")
    print(f"Mutations/source: {args.mutations_per_source}")
    print(f"Quality threshold: {args.quality_threshold}")
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print(f"CUDA device set to: {args.cuda_device}")
    
    if args.mode == "seeds":
        run_seeds(args)
    elif args.mode == "generate":
        run_generate(args)
    elif args.mode == "full":
        run_full(args)


if __name__ == "__main__":
    main()
