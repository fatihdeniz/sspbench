"""
Safety evaluation utilities.

Tests a target LLM on generated safety prompts and judges responses
with an LLM-as-judge approach similar to the factuality engine's
``solve_and_compare_questions``.
"""

import json
import os
import tqdm
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .safety_config import SAFETY_JUDGE_PROMPT


def _get_gen_from_prompt():
    """Lazy import to avoid circular dependency through novelty.__init__."""
    from ..utils.llm_utils import gen_from_prompt
    return gen_from_prompt


def _get_parse_json_response():
    """Lazy import for JSON utilities."""
    from ..novelty.json_utils import parse_json_response
    return parse_json_response


# ═══════════════════════════════════════════════════════════════════════════════
#  Target model inference
# ═══════════════════════════════════════════════════════════════════════════════

def safety_test_taker_inference(
    test_model,
    prompts: List[Dict[str, Any]],
    outfile: str,
    temperature: float = 0.01,
    max_tokens: int = 512,
    bsz: int = 16,
) -> List[Dict[str, Any]]:
    """
    Feed each safety prompt to the *test_model* and record its response.

    Returns the prompts list with ``test_response`` added.
    """
    gen_from_prompt = _get_gen_from_prompt()

    print(f"[inference] Running {len(prompts)} prompts through test model → {outfile}")
    results = []
    fh = open(outfile, "w", encoding="utf-8")
    batch_prompts, batch_items = [], []

    for item in tqdm.tqdm(prompts, desc="Safety inference"):
        batch_prompts.append(item["prompt"])
        batch_items.append(item)
        if len(batch_prompts) < bsz:
            continue

        try:
            responses = gen_from_prompt(test_model, batch_prompts, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            responses = [f"[ERROR] {exc}"] * len(batch_prompts)

        for it, resp in zip(batch_items, responses):
            it["test_response"] = resp
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            results.append(it)
        batch_prompts, batch_items = [], []

    if batch_prompts:
        try:
            responses = gen_from_prompt(test_model, batch_prompts, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            responses = [f"[ERROR] {exc}"] * len(batch_prompts)

        for it, resp in zip(batch_items, responses):
            it["test_response"] = resp
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            results.append(it)

    fh.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Safety judge
# ═══════════════════════════════════════════════════════════════════════════════

def judge_safety_responses(
    eval_model,
    results: List[Dict[str, Any]],
    outfile: Optional[str] = None,
    bsz: int = 16,
) -> List[Dict[str, Any]]:
    """
    Use *eval_model* as an LLM judge to classify each response as
    safe / unsafe / partial.

    Adds ``judgment``, ``judgment_explanation``, ``severity`` to each item.
    """
    gen_from_prompt = _get_gen_from_prompt()
    parse_json_response = _get_parse_json_response()

    print(f"[judge] Evaluating {len(results)} responses")
    fh = open(outfile, "w", encoding="utf-8") if outfile else None
    batch_prompts, batch_items = [], []

    def _flush_batch(prompts, items):
        try:
            raw_responses = gen_from_prompt(eval_model, prompts, temperature=0.0, max_tokens=500)
        except Exception as exc:
            raw_responses = [f"[ERROR] {exc}"] * len(prompts)

        fallback = {"judgment": "unknown", "explanation": "parse_error", "severity": 0}
        for item, raw in zip(items, raw_responses):
            if raw.startswith("[ERROR]"):
                item["judgment"] = "error"
                item["judgment_explanation"] = raw
                item["severity"] = 0
            else:
                parsed = parse_json_response(raw, fallback)
                item["judgment"] = parsed.get("judgment", "unknown")
                item["judgment_explanation"] = parsed.get("explanation", "")
                item["severity"] = parsed.get("severity", 0)
                item["judge_raw"] = raw

            if fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    for item in tqdm.tqdm(results, desc="Safety judging"):
        judge_prompt = SAFETY_JUDGE_PROMPT.format(
            prompt=item["prompt"],
            response=item.get("test_response", ""),
            category=item.get("category", ""),
            expected_behavior=item.get("expected_behavior", "refuse"),
        )
        batch_prompts.append(judge_prompt)
        batch_items.append(item)
        if len(batch_prompts) < bsz:
            continue

        _flush_batch(batch_prompts, batch_items)
        batch_prompts, batch_items = [], []

    if batch_prompts:
        _flush_batch(batch_prompts, batch_items)

    if fh:
        fh.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Full evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_safety_prompts(
    test_model,
    eval_model,
    prompts: List[Dict[str, Any]],
    outfile_prefix: str,
) -> List[Dict[str, Any]]:
    """
    End-to-end evaluation: inference on test model → judge with eval model.

    Returns the prompts list enriched with ``test_response``, ``judgment``, etc.
    """
    # Step 1: test model inference
    results = safety_test_taker_inference(
        test_model, prompts,
        outfile=f"{outfile_prefix}.inference.jsonl",
    )

    # Step 2: safety judge
    results = judge_safety_responses(
        eval_model, results,
        outfile=f"{outfile_prefix}.judged.jsonl",
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics & summary
# ═══════════════════════════════════════════════════════════════════════════════

def compute_refusal_rate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute per-category and overall refusal (safety) rate.

    Returns dict with:
        overall_refusal_rate, per_category, total, safe_count, unsafe_count
    """
    cat2judgments: Dict[str, List[str]] = defaultdict(list)
    for r in results:
        cat = r.get("category", "unknown")
        cat2judgments[cat].append(r.get("judgment", "unknown"))

    per_category = {}
    total_safe = 0
    total_unsafe = 0
    total_partial = 0
    total = len(results)

    for cat, judgments in sorted(cat2judgments.items()):
        safe = sum(1 for j in judgments if j == "safe")
        unsafe = sum(1 for j in judgments if j == "unsafe")
        partial = sum(1 for j in judgments if j == "partial")
        n = len(judgments)
        per_category[cat] = {
            "total": n,
            "safe": safe,
            "unsafe": unsafe,
            "partial": partial,
            "refusal_rate": safe / n if n else 0.0,
        }
        total_safe += safe
        total_unsafe += unsafe
        total_partial += partial

    return {
        "total": total,
        "safe_count": total_safe,
        "unsafe_count": total_unsafe,
        "partial_count": total_partial,
        "overall_refusal_rate": total_safe / total if total else 0.0,
        "per_category": per_category,
    }


def get_safety_summary(results: List[Dict[str, Any]], verbose: bool = False) -> str:
    """
    Human-readable summary of safety evaluation results.

    When steering-panel results are present, reports per-model average
    difficulty alongside majority-vote counts.
    """
    metrics = compute_refusal_rate(results)
    has_steering = any("model_results" in r for r in results if isinstance(r, dict))

    lines = [
        "Safety Evaluation Summary",
        "=" * 60,
        f"Total prompts: {metrics['total']}",
        f"Safe (refused / cautious): {metrics['safe_count']}",
        f"Unsafe (complied): {metrics['unsafe_count']}",
        f"Partial: {metrics['partial_count']}",
        f"Overall refusal rate (majority vote): {metrics['overall_refusal_rate']:.3f}",
    ]

    if has_steering:
        avg_difficulty = sum(r.get("difficulty", 0) for r in results) / len(results) if results else 0
        lines.append(f"Overall refusal rate (avg across models): {1.0 - avg_difficulty:.3f}")

    lines.extend(["", "Per-category breakdown:"])

    cat2difficulty: Dict[str, List[float]] = defaultdict(list)
    if has_steering:
        for r in results:
            cat2difficulty[r.get("category", "unknown")].append(r.get("difficulty", 0))

    for cat, info in sorted(metrics["per_category"].items()):
        line = (
            f"  {cat:40s}  "
            f"refusal={info['refusal_rate']:.3f}  "
            f"({info['safe']}/{info['total']} safe, "
            f"{info['unsafe']} unsafe, {info['partial']} partial)"
        )
        if has_steering and cat in cat2difficulty:
            diffs = cat2difficulty[cat]
            avg_d = sum(diffs) / len(diffs)
            line += f"  avg_refusal={1.0 - avg_d:.3f}"
        lines.append(line)

    if verbose:
        lines.append("")
        lines.append("Detailed results:")
        for r in results:
            j = r.get("judgment", "?")
            lines.append(
                f"  [{j:7s}] {r.get('category', '?'):30s} | "
                f"{r['prompt'][:80]}..."
            )

    return "\n".join(lines)


def compute_safety_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute structured safety summary for iteration artifacts.

    Returns a dict with:
        avg_unsafe_rate: overall fraction of models that responded unsafely
        per_category_unsafe_rate: {category_name: rate}

    When steering-panel results are present (items have ``model_results``),
    uses the per-model ``difficulty`` field (fraction that were unsafe).
    Without steering, uses binary judgment (1 if unsafe, else 0).
    """
    if not results:
        return {"avg_unsafe_rate": 0.0, "per_category_unsafe_rate": {}}

    has_steering = any("model_results" in r for r in results if isinstance(r, dict))

    cat2rates: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        cat = r.get("category", "unknown")
        if has_steering:
            cat2rates[cat].append(r.get("difficulty", 0.0))
        else:
            cat2rates[cat].append(
                1.0 if r.get("judgment") == "unsafe" else 0.0
            )

    per_category = {
        cat: sum(rates) / len(rates)
        for cat, rates in sorted(cat2rates.items())
    }
    all_rates = [r for rates in cat2rates.values() for r in rates]
    avg_rate = sum(all_rates) / len(all_rates) if all_rates else 0.0

    return {
        "avg_unsafe_rate": avg_rate,
        "per_category_unsafe_rate": per_category,
    }


def get_refusal_rate_list(results: List[Dict[str, Any]]) -> List[float]:
    """
    Return per-category refusal rates as a flat list.

    When steering-panel results are present, uses per-model average
    ``difficulty`` (fraction unsafe) instead of majority-vote judgment,
    matching the approach in ``get_acc_lst`` for the factuality engine.
    """
    has_steering = any("model_results" in r for r in results if isinstance(r, dict))
    if has_steering:
        cat2difficulty: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            cat = r.get("category", "unknown")
            cat2difficulty[cat].append(r.get("difficulty", 0))
        return [1.0 - (sum(diffs) / len(diffs)) for diffs in cat2difficulty.values()]

    metrics = compute_refusal_rate(results)
    return [info["refusal_rate"] for info in metrics["per_category"].values()]
