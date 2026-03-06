#!/usr/bin/env python3
"""
sample_safety_variations.py — Sample a benchmark round from the canonical
safety variations file produced by ``run_safety_variations.py``.

Each round samples a balanced subset, tracking what was used in prior
rounds so there is no overlap.

Usage
-----
    # Round 1: sample 2000 variations
    python scripts/sample_safety_variations.py --round 1 --size 2000

    # Round 2: automatically excludes round 1 selections
    python scripts/sample_safety_variations.py --round 2 --size 2000

    # Sample only system-prompt overrides
    python scripts/sample_safety_variations.py --round 1 --size 500 \\
        --mutations system_prompt_override

    # Balanced across mutations, 300 per mutation type
    python scripts/sample_safety_variations.py --round 1 --per-mutation 300
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "scripts" else _HERE
sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_VARIATIONS_DIR = os.path.join(_PROJECT_ROOT, "data", "safety_variations")
DEFAULT_VARIATIONS_FILE = os.path.join(DEFAULT_VARIATIONS_DIR, "safety_variations.jsonl")
MANIFEST_FILE = "sampling_manifest.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  Manifest (tracks which variation_ids have been used in prior rounds)
# ═══════════════════════════════════════════════════════════════════════════════

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load the sampling manifest, or create an empty one."""
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as fh:
            return json.load(fh)
    return {"rounds": {}}


def save_manifest(manifest: Dict[str, Any], manifest_path: str):
    """Persist the manifest to disk."""
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def get_used_ids(manifest: Dict[str, Any], exclude_round: Optional[str] = None) -> Set[str]:
    """Collect all variation_ids used in prior rounds."""
    used = set()
    for rnd, info in manifest.get("rounds", {}).items():
        if rnd != exclude_round:
            used.update(info.get("variation_ids", []))
    return used


# ═══════════════════════════════════════════════════════════════════════════════
#  Sampling strategies
# ═══════════════════════════════════════════════════════════════════════════════

def sample_balanced(
    variations: List[Dict[str, Any]],
    size: int,
    used_ids: Set[str],
    target_mutations: Optional[List[str]] = None,
    per_mutation: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Sample *size* variations balanced across mutation types.

    Parameters
    ----------
    variations : list
        The full canonical variations pool.
    size : int
        Total number of variations to sample (ignored if per_mutation set).
    used_ids : set
        variation_ids to exclude (already used in prior rounds).
    target_mutations : list, optional
        Only sample from these mutation types.
    per_mutation : int, optional
        If set, sample exactly this many per mutation type (overrides size).
    """
    # Filter out already-used
    available = [v for v in variations if v["variation_id"] not in used_ids]
    print(f"  {len(variations)} total, {len(available)} available (excluding {len(used_ids)} used)")

    # Group by mutation
    by_mutation: Dict[str, List[Dict]] = defaultdict(list)
    for v in available:
        m = v["mutation"]
        if target_mutations and m not in target_mutations:
            continue
        by_mutation[m].append(v)

    if not by_mutation:
        print("  WARNING: No available variations match the criteria")
        return []

    mutations = sorted(by_mutation.keys())
    print(f"  Mutation types: {', '.join(f'{m}({len(by_mutation[m])})' for m in mutations)}")

    selected: List[Dict[str, Any]] = []

    if per_mutation is not None:
        # Fixed count per mutation
        for m in mutations:
            pool = by_mutation[m]
            random.shuffle(pool)
            n = min(per_mutation, len(pool))
            selected.extend(pool[:n])
            if n < per_mutation:
                print(f"    ⚠ {m}: only {n} available (requested {per_mutation})")
    else:
        # Balanced round-robin up to size
        per_m = max(1, size // len(mutations))
        remainder = size - per_m * len(mutations)

        for m in mutations:
            pool = by_mutation[m]
            random.shuffle(pool)
            n = min(per_m, len(pool))
            selected.extend(pool[:n])

        # Fill remainder from the largest pools
        if remainder > 0:
            for m in sorted(mutations, key=lambda m: len(by_mutation[m]), reverse=True):
                pool = by_mutation[m]
                already_taken = per_m
                remaining_pool = pool[already_taken:]
                to_take = min(remainder, len(remaining_pool))
                selected.extend(remaining_pool[:to_take])
                remainder -= to_take
                if remainder <= 0:
                    break

    random.shuffle(selected)
    return selected


def sample_diverse_sources(
    variations: List[Dict[str, Any]],
    size: int,
    used_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Sample maximising source prompt diversity — at most one variation
    per source_id, spread across different mutations."""
    available = [v for v in variations if v["variation_id"] not in used_ids]

    # Group by source_id
    by_source: Dict[Any, List[Dict]] = defaultdict(list)
    for v in available:
        by_source[v["source_id"]].append(v)

    # Pick one random variation per source
    candidates = []
    for sid, vars_list in by_source.items():
        random.shuffle(vars_list)
        candidates.append(vars_list[0])

    random.shuffle(candidates)
    return candidates[:size]


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sample a benchmark round from safety variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--round", type=int, required=True,
        help="Round number (used for naming and history tracking)",
    )
    parser.add_argument(
        "--size", type=int, default=2000,
        help="Total samples to draw (default: 2000; ignored if --per-mutation set)",
    )
    parser.add_argument(
        "--per-mutation", type=int, default=None,
        help="Fixed count per mutation type (overrides --size)",
    )
    parser.add_argument(
        "--mutations", nargs="*", default=None,
        help="Only sample from these mutation types",
    )
    parser.add_argument(
        "--strategy", choices=["balanced", "diverse_sources"], default="balanced",
        help="Sampling strategy (default: balanced across mutations)",
    )
    parser.add_argument(
        "--variations-file", default=DEFAULT_VARIATIONS_FILE,
        help="Path to canonical safety_variations.jsonl",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_VARIATIONS_DIR,
        help="Output directory for round files",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: round number)",
    )

    args = parser.parse_args()
    seed = args.seed if args.seed is not None else args.round
    random.seed(seed)

    round_key = str(args.round)
    print(f"=" * 60)
    print(f"  Sampling Round {args.round}")
    print(f"=" * 60)

    # ── Load variations ──────────────────────────────────────────────────
    if not os.path.isfile(args.variations_file):
        print(f"ERROR: Variations file not found: {args.variations_file}")
        print("Run run_safety_variations.py first to generate the canonical file.")
        sys.exit(1)

    print("Loading variations ...")
    variations = []
    with open(args.variations_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                variations.append(json.loads(line))
    print(f"  Loaded {len(variations)} variations")

    # ── Load manifest ────────────────────────────────────────────────────
    manifest_path = os.path.join(args.output_dir, MANIFEST_FILE)
    manifest = load_manifest(manifest_path)
    used_ids = get_used_ids(manifest)
    print(f"  Prior rounds: {len(manifest.get('rounds', {}))} | Used IDs: {len(used_ids)}")

    # ── Sample ───────────────────────────────────────────────────────────
    if args.strategy == "diverse_sources":
        selected = sample_diverse_sources(variations, args.size, used_ids)
    else:
        selected = sample_balanced(
            variations, args.size, used_ids,
            target_mutations=args.mutations,
            per_mutation=args.per_mutation,
        )

    if not selected:
        print("No variations selected — pool may be exhausted.")
        sys.exit(0)

    print(f"\n  Selected {len(selected)} variations for round {args.round}")

    # Stats
    mutation_counts = defaultdict(int)
    source_ids = set()
    for v in selected:
        mutation_counts[v["mutation"]] += 1
        source_ids.add(v["source_id"])
    print(f"  Unique source prompts: {len(source_ids)}")
    print(f"  By mutation:")
    for m, cnt in sorted(mutation_counts.items(), key=lambda x: -x[1]):
        print(f"    {m}: {cnt}")

    # ── Write round output ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    round_path = os.path.join(args.output_dir, f"round_{args.round}.jsonl")
    with open(round_path, "w", encoding="utf-8") as fh:
        for v in selected:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"\n  Output: {round_path}")

    # ── Update manifest ──────────────────────────────────────────────────
    manifest["rounds"][round_key] = {
        "size": len(selected),
        "strategy": args.strategy,
        "mutations": args.mutations,
        "per_mutation": args.per_mutation,
        "seed": seed,
        "variation_ids": [v["variation_id"] for v in selected],
    }
    save_manifest(manifest, manifest_path)
    print(f"  Manifest updated: {manifest_path}")

    print(f"\n{'=' * 60}")
    print(f"  Round {args.round}: {len(selected)} variations sampled")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
