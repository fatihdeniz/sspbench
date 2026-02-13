"""
safety_seeds.py — Extract seed topics & existing prompts from aiXamine safety benchmarks.

Usage::

    python safety_seeds.py                                          # auto-detect paths
    python safety_seeds.py --prompt-dir /path/to/safety-alignment/prompts
    python safety_seeds.py --output-dir /tmp/seeds

This is the safety counterpart of ``hallucination_seeds.py``.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ── Unified safety taxonomy (matches sspbench.safety.config) ─────────────────
SAFETY_TAXONOMY = {
    "VH": "Violence & Physical Harm",
    "HV": "Hate & Violence",
    "V":  "Violence",
    "VC": "Violent Crimes",
    "SH": "Self-Harm & Suicide",
    "SC": "Sexual Content",
    "H":  "Hate Speech & Discrimination",
    "HR": "Harassment",
    "HL": "Harmful Language",
    "NVC":"Non-Violent Crimes & Illegal Activity",
    "CIA":"Criminal & Illegal Activity",
    "CS": "Controlled Substances",
    "CP": "Copyright & Intellectual Property Violations",
    "P":  "Privacy Violations & Personal Information",
    "IP": "Invasion of Privacy",
    "D":  "Deception & Fraud",
    "IW": "Influence & Manipulation",
    "MI": "Misinformation",
    "MU": "Malicious Use",
    "E":  "Ethical Concerns",
    "SA": "Sensitive & Controversial Topics",
    "SRC":"Socially Responsible Content",
    "CSE":"Child Safety & Exploitation",
    "SF": "System & Functional Safety",
    "OH": "Operational Hazards",
    "CA": "Cyber Attacks",
    "IR": "Inappropriate Refusal",
    "T":  "Toxicity",
    "I":  "Insult",
    "IA": "Identity Attack",
    "TH": "Threat",
    "ST": "Severe Toxicity",
}

PROMPT_FIELDS = ("prompt", "query", "question", "text")
CATEGORY_FIELDS = ("category", "type", "harm_code", "harm_type", "domain", "topic")


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load_records(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        items = data if isinstance(data, list) else data.get("data", data.get("items", []))
        for item in items:
            if isinstance(item, dict):
                yield item


def _extract_prompt(record: Dict[str, Any]) -> Optional[str]:
    for field in PROMPT_FIELDS:
        val = record.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_categories(record: Dict[str, Any]) -> Set[str]:
    cats: Set[str] = set()
    for key in CATEGORY_FIELDS:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            for part in val.split(","):
                part = part.strip()
                if part:
                    cats.add(part)
        elif isinstance(val, list):
            cats.update(str(v).strip() for v in val if str(v).strip())
    return cats


def _resolve_category(code: str) -> str:
    return SAFETY_TAXONOMY.get(code, code)


# ── Public API ───────────────────────────────────────────────────────────────

def build_seed_topics(prompt_dir: str) -> List[str]:
    """Extract deduplicated human-readable safety topics from prompt datasets."""
    topics: Set[str] = set()
    for filename in sorted(os.listdir(prompt_dir)):
        if not (filename.endswith(".jsonl") or filename.endswith(".json")):
            continue
        path = os.path.join(prompt_dir, filename)
        for record in _load_records(path):
            for code in _extract_categories(record):
                topics.add(_resolve_category(code))
    return sorted(topics)


def build_existing_prompts(prompt_dir: str) -> List[Dict[str, Any]]:
    """Read all safety prompts into a flat list with metadata."""
    prompts: List[Dict[str, Any]] = []
    idx = 0
    for filename in sorted(os.listdir(prompt_dir)):
        if not (filename.endswith(".jsonl") or filename.endswith(".json")):
            continue
        dataset_name = os.path.splitext(filename)[0]
        path = os.path.join(prompt_dir, filename)
        for record in _load_records(path):
            text = _extract_prompt(record)
            if not text:
                continue
            cats = _extract_categories(record)
            prompts.append({
                "id": record.get("id", idx),
                "dataset": dataset_name,
                "prompt": text,
                "categories": sorted({_resolve_category(c) for c in cats}),
                "raw_categories": sorted(cats),
            })
            idx += 1
    return prompts


def write_seed_topics(prompt_dir: str, output_dir: str, service: str = "safety-alignment") -> str:
    os.makedirs(output_dir, exist_ok=True)
    topics = build_seed_topics(prompt_dir)
    payload = {"service": service, "seed_topics": topics, "num_topics": len(topics)}
    path = os.path.join(output_dir, "seed_topics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[seed_topics] Wrote {len(topics)} topics to {path}")
    return path


def write_existing_prompts(prompt_dir: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    prompts = build_existing_prompts(prompt_dir)
    path = os.path.join(output_dir, "existing_safety_prompts.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[existing_prompts] Wrote {len(prompts)} prompts to {path}")
    return path


def write_category_stats(prompt_dir: str, output_dir: str) -> str:
    """Write per-dataset category statistics."""
    os.makedirs(output_dir, exist_ok=True)
    from collections import Counter, defaultdict

    dataset_stats: Dict[str, Dict[str, Any]] = {}
    overall_counts: Counter = Counter()

    for filename in sorted(os.listdir(prompt_dir)):
        if not (filename.endswith(".jsonl") or filename.endswith(".json")):
            continue
        dataset_name = os.path.splitext(filename)[0]
        path = os.path.join(prompt_dir, filename)
        ds_cats: Counter = Counter()
        n = 0
        for record in _load_records(path):
            n += 1
            for code in _extract_categories(record):
                resolved = _resolve_category(code)
                ds_cats[resolved] += 1
                overall_counts[resolved] += 1
        dataset_stats[dataset_name] = {
            "num_prompts": n,
            "categories": dict(ds_cats.most_common()),
        }

    stats = {
        "overall_category_counts": dict(overall_counts.most_common()),
        "num_datasets": len(dataset_stats),
        "datasets": dataset_stats,
    }
    path = os.path.join(output_dir, "category_stats.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[category_stats] Wrote stats for {len(dataset_stats)} datasets to {path}")
    return path


# ── Path resolution ──────────────────────────────────────────────────────────

def _resolve_paths(
    prompt_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[str, str]:
    # Try to find aiXamine safety-alignment prompts relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    if prompt_dir is None:
        candidate = os.path.join(
            project_root, "aiXamine",
            "airflow-tasks", "services", "safety-alignment", "prompts",
        )
        if os.path.isdir(candidate):
            prompt_dir = candidate
        else:
            # Might be one level up
            candidate2 = os.path.join(
                os.path.dirname(project_root), "aiXamine",
                "airflow-tasks", "services", "safety-alignment", "prompts",
            )
            prompt_dir = candidate2

    if output_dir is None:
        output_dir = os.path.join(project_root, "sspbench", "data", "curation", "safety-alignment")

    return prompt_dir, output_dir


# ── CLI entry point ──────────────────────────────────────────────────────────

def main(prompt_dir: Optional[str] = None, output_dir: Optional[str] = None):
    prompt_dir, output_dir = _resolve_paths(prompt_dir, output_dir)

    if not os.path.isdir(prompt_dir):
        print(f"ERROR: Prompt directory not found: {prompt_dir}", file=sys.stderr)
        print("Use --prompt-dir to specify the path to safety-alignment/prompts")
        sys.exit(1)

    print(f"Prompt directory: {prompt_dir}")
    print(f"Output directory: {output_dir}")
    print()

    write_seed_topics(prompt_dir, output_dir)
    write_existing_prompts(prompt_dir, output_dir)
    write_category_stats(prompt_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract safety seed topics from aiXamine benchmarks")
    parser.add_argument("--prompt-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    main(prompt_dir=args.prompt_dir, output_dir=args.output_dir)
