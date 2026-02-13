#!/usr/bin/env python3
"""
run_safety_engine.py — Standalone script to run the Safety Novelty Engine.

This script mirrors the factuality engine runner but targets safety-alignment
benchmark generation.

Usage examples
--------------

1.  Seed extraction only (no LLM needed)::

        python run_safety_engine.py --mode seeds \
            --seed-prompt-dir /path/to/aiXamine/.../safety-alignment/prompts \
            --output-dir ./output/safety

2.  Full pipeline (requires LLM access)::

        python run_safety_engine.py --mode full \
            --agent-model openai/gpt-4o \
            --test-model meta-llama/Llama-3.1-8B-Instruct \
            --eval-model openai/gpt-4o \
            --seed-prompt-dir /path/to/aiXamine/.../safety-alignment/prompts \
            --max-iterations 3 \
            --output-dir ./output/safety

3.  Generate only (no model evaluation)::

        python run_safety_engine.py --mode generate \
            --agent-model openai/gpt-4o \
            --eval-model openai/gpt-4o \
            --max-iterations 1 \
            --output-dir ./output/safety
"""

import argparse
import json
import os
import sys

# Ensure the project root is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "scripts" else _HERE
sys.path.insert(0, _PROJECT_ROOT)


def run_seeds(args):
    """Mode: extract seed topics and existing prompts from aiXamine."""
    from sspbench.safety.seed_topics import write_seed_artifacts

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
    from sspbench.safety.core import generate_full_safety_prompts, refine_safety_categories
    from sspbench.safety.seed_topics import build_existing_prompts

    agent_model = create_model_from_config({"model": args.agent_model, "type": args.model_type})
    eval_model = None
    if args.eval_model:
        eval_model = create_model_from_config({"model": args.eval_model, "type": args.model_type})

    existing_prompts = []
    if args.seed_prompt_dir and os.path.isdir(args.seed_prompt_dir):
        existing_prompts = build_existing_prompts(args.seed_prompt_dir)
        print(f"Loaded {len(existing_prompts)} existing prompts for diversity checking")

    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "data", "safety_novelty")
    os.makedirs(output_dir, exist_ok=True)

    all_prompts = []
    history_text = ["Initial iteration"]

    for iteration in range(1, args.max_iterations + 1):
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
    from sspbench.safety.main import run_safety_novelty_engine

    agent_model = create_model_from_config({"model": args.agent_model, "type": args.model_type})
    test_model = create_model_from_config({"model": args.test_model, "type": args.model_type})
    eval_model = create_model_from_config({"model": args.eval_model, "type": args.model_type})

    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "data", "safety_novelty")

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
        seed_prompt_dir=args.seed_prompt_dir,
        output_dir=output_dir,
        mutations_per_source=args.mutations_per_source,
    )

    print(f"\n✅  Full pipeline complete — {len(results['all_prompts'])} total prompts generated and evaluated.")


def main():
    parser = argparse.ArgumentParser(
        description="Safety Novelty Engine — Adaptive safety benchmark generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["seeds", "generate", "full"], default="seeds",
                        help="Execution mode: seeds (extract only), generate (no eval), full (end-to-end)")

    # Model config
    parser.add_argument("--agent-model", default=None, help="Agent model (red-teamer / generator)")
    parser.add_argument("--test-model", default=None, help="Target model to evaluate")
    parser.add_argument("--eval-model", default=None, help="Evaluation judge model")
    parser.add_argument("--model-type", default="openrouter", help="Model backend type (default: openrouter)")

    # Data paths
    parser.add_argument("--seed-prompt-dir", default=None,
                        help="Path to aiXamine safety-alignment/prompts directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    # Engine params
    parser.add_argument("--theme", default="LLM safety alignment", help="Generation theme")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--refusal-target", default="0.8--1.0", help="Target refusal rate range")
    parser.add_argument("--num-categories", type=int, default=10, help="Categories per iteration")
    parser.add_argument("--num-prompts", type=int, default=5, help="Prompts per category")
    parser.add_argument("--quality-threshold", type=int, default=6, help="Min quality score (0-10)")
    parser.add_argument("--mutations-per-source", type=int, default=2,
                        help="Number of mutation operators to apply per mined source prompt")

    args = parser.parse_args()

    if args.mode == "seeds":
        run_seeds(args)
    elif args.mode == "generate":
        if not args.agent_model:
            parser.error("--agent-model is required for generate mode")
        run_generate(args)
    elif args.mode == "full":
        if not all([args.agent_model, args.test_model, args.eval_model]):
            parser.error("--agent-model, --test-model, and --eval-model are all required for full mode")
        run_full(args)


if __name__ == "__main__":
    main()
