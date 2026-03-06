#!/usr/bin/env python3
"""
run_safety_variations.py — Exhaustive safety-alignment variation generator.

For every source prompt in the existing corpus, applies **all** mutation
operators via LLM:
  - 6 LLM-rewrite operators  (persona, hypothetical, entailment,
    topic-transplant, escalation-harder, escalation-easier)
  - 81 system-prompt-override templates (each LLM-rephrased)
  - Compositional blends (configurable count)

Outputs a canonical JSONL file keyed by ``source_id``.  Use
``sample_safety_variations.py`` to draw balanced benchmark rounds from it.

Usage
-----
    # Full run on all source prompts
    python scripts/run_safety_variations.py

    # Limit source to 200 prompts (for testing)
    python scripts/run_safety_variations.py --max-source-prompts 200

    # Resume from a previous partial run
    python scripts/run_safety_variations.py --resume

    # Add quality annotations (slower, needs eval model)
    python scripts/run_safety_variations.py --quality-check
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "scripts" else _HERE
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-ragas")


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_EXISTING_PROMPTS = os.path.join(
    _PROJECT_ROOT, "data", "curation", "safety-alignment", "existing_prompts.jsonl",
)
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "safety_variations")
DEFAULT_MAX_SOURCE = 0          # 0 = all
DEFAULT_BLEND_COUNT = 500       # number of compositional blends
DEFAULT_QUALITY_THRESHOLD = 6

DEFAULT_AGENT_CONFIG = {
    "type": "openai",
    "model": os.environ.get("JUDGE_EXTERNAL_MODEL", "gpt-4.1-mini-aixamine"),
    "api_url": os.environ.get("JUDGE_EXTERNAL_ENDPOINT", ""),
    "api_token": os.environ.get("JUDGE_EXTERNAL_TOKEN", ""),
    "api_version": os.environ.get("JUDGE_EXTERNAL_VERSION", "2024-12-01-preview"),
}

DEFAULT_EVAL_CONFIG = {
    "type": "openai",
    "model": os.environ.get("EVAL_MODEL", "gpt-oss"),
    "api_url": os.environ.get("EVAL_ENDPOINT", "http://10.4.8.217:8000/v1"),
    "api_token": os.environ.get("EVAL_TOKEN", "abc123"),
    "api_version": os.environ.get("JUDGE_EXTERNAL_VERSION", "2024-12-01-preview"),
}

# The 6 LLM-rewrite operators (everything except system_prompt_override)
LLM_OPERATORS = [
    "persona_injection",
    "hypothetical_frame",
    "entailment_shift",
    "topic_transplant",
    "escalation_harder",
    "escalation_easier",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_prompts(path: str) -> List[Dict[str, Any]]:
    """Load existing prompts from a JSONL file."""
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def load_completed_sources(output_path: str) -> set:
    """Scan existing output to find source_ids already fully processed."""
    done = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    done.add(row.get("source_id"))
                except json.JSONDecodeError:
                    pass
    return done


def write_variation(fh, variation: Dict[str, Any]):
    """Append a single variation row to the open file handle."""
    fh.write(json.dumps(variation, ensure_ascii=False) + "\n")
    fh.flush()


def _make_variation(
    src: Dict[str, Any],
    mutation_name: str,
    variant_idx: int,
    mutated: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalise a mutation result into the canonical variation schema."""
    source_id = src.get("id")
    return {
        "source_id": source_id,
        "source_prompt": src.get("prompt", ""),
        "dataset": src.get("dataset", ""),
        "categories": src.get("categories", []),
        "raw_categories": src.get("raw_categories", []),
        "variation_id": f"{source_id}_{mutation_name}_{variant_idx}",
        "mutation": mutation_name,
        "prompt": mutated.get("prompt", ""),
        "original_prompt": mutated.get("original_prompt", src.get("prompt", "")),
        "subtlety": mutated.get("subtlety", ""),
        "generation_method": f"variation_{mutation_name}",
        # system_prompt_override extras
        "override_template_id": mutated.get("override_template_id"),
        "override_category": mutated.get("override_category"),
        # topic_transplant extras
        "topic_target": mutated.get("topic_target"),
    }


def _make_blend_variation(
    prompt_a: Dict[str, Any],
    prompt_b: Dict[str, Any],
    blend_idx: int,
    blended: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalise a blend result into the canonical variation schema."""
    return {
        "source_id": f"{prompt_a.get('id')}+{prompt_b.get('id')}",
        "source_prompt": prompt_a.get("prompt", ""),
        "source_prompt_b": prompt_b.get("prompt", ""),
        "dataset": f"{prompt_a.get('dataset', '')}+{prompt_b.get('dataset', '')}",
        "categories": list(set(
            prompt_a.get("categories", []) + prompt_b.get("categories", [])
        )),
        "raw_categories": list(set(
            prompt_a.get("raw_categories", []) + prompt_b.get("raw_categories", [])
        )),
        "variation_id": f"blend_{blend_idx}",
        "mutation": "compositional_blend",
        "prompt": blended.get("prompt", ""),
        "original_prompt": blended.get("source_a", ""),
        "subtlety": "compositional",
        "generation_method": "variation_compositional_blend",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Core generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_variations_for_prompt(
    src: Dict[str, Any],
    agent_model,
    override_template_count: int,
) -> List[Dict[str, Any]]:
    """Generate all mutation variations for a single source prompt.

    Returns a list of normalised variation dicts.
    """
    from sspbench.generators.safety_mutations import (
        MUTATION_OPERATORS,
        mutate_system_prompt_override,
        MAX_PROMPT_CHARS,
    )

    variations = []
    variant_idx = 0

    # ── 6 LLM-rewrite operators ──────────────────────────────────────────
    for op_name in LLM_OPERATORS:
        try:
            op_func = MUTATION_OPERATORS[op_name]
            result = op_func(src, agent_model)
            if len(result.get("prompt", "")) <= MAX_PROMPT_CHARS:
                var = _make_variation(src, op_name, variant_idx, result)
                variations.append(var)
                variant_idx += 1
        except Exception as exc:
            print(f"    ⚠ {op_name} failed on {src.get('id')}: {str(exc)[:80]}")

    # ── All system-prompt-override templates ──────────────────────────────
    for tmpl_idx in range(override_template_count):
        try:
            result = mutate_system_prompt_override(
                src, agent_model, template_index=tmpl_idx,
            )
            if len(result.get("prompt", "")) <= MAX_PROMPT_CHARS:
                var = _make_variation(src, "system_prompt_override", variant_idx, result)
                variations.append(var)
                variant_idx += 1
        except Exception as exc:
            print(f"    ⚠ spo#{tmpl_idx} failed on {src.get('id')}: {str(exc)[:80]}")

    return variations


def generate_blends(
    source_prompts: List[Dict[str, Any]],
    agent_model,
    blend_count: int,
) -> List[Dict[str, Any]]:
    """Generate compositional blends from random prompt pairs."""
    from sspbench.generators.safety_mutations import (
        mutate_compositional_blend,
        MAX_PROMPT_CHARS,
    )

    shuffled = source_prompts.copy()
    random.shuffle(shuffled)
    n_pairs = min(blend_count, len(shuffled) // 2)

    variations = []
    for i in range(n_pairs):
        a = shuffled[i * 2]
        b = shuffled[i * 2 + 1]
        try:
            blended = mutate_compositional_blend(a, b, agent_model)
            if len(blended.get("prompt", "")) <= MAX_PROMPT_CHARS:
                var = _make_blend_variation(a, b, i, blended)
                variations.append(var)
        except Exception as exc:
            print(f"    ⚠ blend#{i} failed: {str(exc)[:80]}")

        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{n_pairs}] blends generated")

    return variations


# ═══════════════════════════════════════════════════════════════════════════════
#  Quality annotation (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def run_quality_checks(
    output_path: str,
    eval_model,
    quality_threshold: int,
) -> str:
    """Read the variations JSONL, annotate with quality scores, write filtered file.

    Returns path to the filtered file.
    """
    from sspbench.safety.safety_core import check_safety_scope, check_safety_quality

    # Load all variations
    all_vars = []
    with open(output_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                all_vars.append(json.loads(line))

    print(f"\n  Running scope check on {len(all_vars)} variations ...")
    in_scope = check_safety_scope(all_vars, eval_model)

    print(f"  Running quality check on {len(in_scope)} in-scope variations ...")
    high_quality = check_safety_quality(in_scope, eval_model, quality_threshold)

    # Write annotated full set (all_vars now have quality annotations in-place)
    annotated_path = output_path.replace(".jsonl", "_annotated.jsonl")
    with open(annotated_path, "w", encoding="utf-8") as fh:
        for v in all_vars:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")

    # Write filtered set
    filtered_path = output_path.replace(".jsonl", "_filtered.jsonl")
    with open(filtered_path, "w", encoding="utf-8") as fh:
        for v in high_quality:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"  Annotated: {len(all_vars)} → {annotated_path}")
    print(f"  Filtered:  {len(high_quality)} → {filtered_path}")
    return filtered_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive safety-alignment variation generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--existing-prompts", default=DEFAULT_EXISTING_PROMPTS,
        help="Path to existing_prompts.jsonl",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--max-source-prompts", type=int, default=DEFAULT_MAX_SOURCE,
        help="Max source prompts to process (0 = all)",
    )
    parser.add_argument(
        "--blend-count", type=int, default=DEFAULT_BLEND_COUNT,
        help=f"Number of compositional blends (default: {DEFAULT_BLEND_COUNT})",
    )
    parser.add_argument(
        "--quality-check", action="store_true",
        help="Run scope + quality filtering with eval model after generation",
    )
    parser.add_argument(
        "--quality-threshold", type=int, default=DEFAULT_QUALITY_THRESHOLD,
        help=f"Min quality score 0-10 (default: {DEFAULT_QUALITY_THRESHOLD})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous partial run (skip already-processed source_ids)",
    )
    parser.add_argument(
        "--agent-model", default=None,
        help="Override agent model",
    )
    parser.add_argument(
        "--eval-model", default=None,
        help="Override eval model",
    )
    parser.add_argument(
        "--model-type", default=None,
        help="Model backend type for overrides",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("  Safety-Alignment Variation Generator (exhaustive)")
    print("=" * 60)

    # ── Load source prompts ──────────────────────────────────────────────
    if not os.path.isfile(args.existing_prompts):
        print(f"ERROR: File not found: {args.existing_prompts}")
        sys.exit(1)

    source = load_existing_prompts(args.existing_prompts)
    print(f"Loaded {len(source)} existing prompts")

    if args.max_source_prompts > 0 and args.max_source_prompts < len(source):
        random.shuffle(source)
        source = source[: args.max_source_prompts]
        print(f"Sampled {len(source)} prompts")

    # ── Models ───────────────────────────────────────────────────────────
    from sspbench.utils.llm_utils import create_model_from_config

    agent_cfg = DEFAULT_AGENT_CONFIG.copy()
    if args.agent_model:
        agent_cfg = {"model": args.agent_model, "type": args.model_type or "openrouter"}
    agent_model = create_model_from_config(agent_cfg)

    eval_model = None
    if args.quality_check:
        eval_cfg = DEFAULT_EVAL_CONFIG.copy()
        if args.eval_model:
            eval_cfg = {"model": args.eval_model, "type": args.model_type or "openrouter"}
        eval_model = create_model_from_config(eval_cfg)

    # ── Count override templates ─────────────────────────────────────────
    from sspbench.generators.safety_mutations import _load_override_templates
    override_templates = _load_override_templates()
    n_templates = len(override_templates)

    # ── Print config ─────────────────────────────────────────────────────
    per_prompt = len(LLM_OPERATORS) + n_templates
    print(f"\nConfig:")
    print(f"  Source prompts:       {len(source)}")
    print(f"  LLM operators:        {len(LLM_OPERATORS)} ({', '.join(LLM_OPERATORS)})")
    print(f"  Override templates:   {n_templates}")
    print(f"  Variations/prompt:    {per_prompt}")
    print(f"  Blend count:          {args.blend_count}")
    print(f"  Estimated total:      ~{len(source) * per_prompt + args.blend_count}")
    print(f"  Quality check:        {'ON' if args.quality_check else 'OFF'}")
    print(f"  Resume:               {'ON' if args.resume else 'OFF'}")
    print()

    # ── Output setup ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "safety_variations.jsonl")

    # ── Resume logic ─────────────────────────────────────────────────────
    done_ids: set = set()
    if args.resume:
        done_ids = load_completed_sources(output_path)
        print(f"Resume: {len(done_ids)} source_ids already processed, skipping")
        source = [s for s in source if s.get("id") not in done_ids]
        print(f"Remaining: {len(source)} source prompts to process")

    # ── Generate per-prompt variations ───────────────────────────────────
    mode = "a" if args.resume else "w"
    t0 = time.time()
    total_written = 0
    method_counts: Counter = Counter()

    with open(output_path, mode, encoding="utf-8") as fh:
        for i, src in enumerate(source):
            variations = generate_all_variations_for_prompt(
                src, agent_model, n_templates,
            )
            for v in variations:
                write_variation(fh, v)
                method_counts[v["mutation"]] += 1
            total_written += len(variations)

            if (i + 1) % 10 == 0 or i + 1 == len(source):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(source) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1}/{len(source)}] "
                    f"{total_written} variations | "
                    f"{rate:.1f} prompts/s | "
                    f"ETA {eta / 60:.0f}m"
                )

        # ── Compositional blends ─────────────────────────────────────────
        if args.blend_count > 0:
            # Reload full source for blending (including already-done ones)
            full_source = load_existing_prompts(args.existing_prompts)
            if args.max_source_prompts > 0:
                full_source = full_source[: args.max_source_prompts]

            print(f"\n  Generating {args.blend_count} compositional blends ...")
            blends = generate_blends(full_source, agent_model, args.blend_count)
            for b in blends:
                write_variation(fh, b)
                method_counts["compositional_blend"] += 1
            total_written += len(blends)

    t_gen = time.time() - t0
    print(f"\nGeneration complete: {total_written} variations in {t_gen / 60:.1f}m")
    print("  By mutation:")
    for m, cnt in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"    {m}: {cnt}")

    # ── Quality checks (optional) ────────────────────────────────────────
    if args.quality_check and eval_model:
        run_quality_checks(output_path, eval_model, args.quality_threshold)

    # ── Stats ────────────────────────────────────────────────────────────
    stats = {
        "source_prompts": len(source) + len(done_ids),
        "total_variations": total_written + (len(done_ids) * per_prompt if args.resume else 0),
        "blend_count": args.blend_count,
        "llm_operators": LLM_OPERATORS,
        "override_templates": n_templates,
        "variations_per_prompt": per_prompt,
        "method_counts": dict(method_counts),
        "generation_time_mins": round(t_gen / 60, 1),
        "quality_check": args.quality_check,
    }
    stats_path = os.path.join(args.output_dir, "safety_variations_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nOutput:  {output_path}")
    print(f"Stats:   {stats_path}")
    print(f"\n{'=' * 60}")
    print(f"  Done — {total_written} safety-alignment variations")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
