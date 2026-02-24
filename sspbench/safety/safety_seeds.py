"""
Seed topic extraction from existing aiXamine safety benchmarks.

Reads the JSONL prompt files in aiXamine/airflow-tasks/services/safety-alignment/prompts
and (optionally) the jailbreak prompts, and produces:

1. A deduplicated list of harm-category seed topics.
2. A JSONL file of existing prompts (for overlap / diversity checks).
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .safety_config import SAFETY_TAXONOMY, HARM_FAMILIES

# ── field names that may carry the prompt text ───────────────────────────────
PROMPT_FIELDS = ("prompt", "query", "question", "text")
CATEGORY_FIELDS = ("category", "type", "harm_code", "harm_type", "domain", "topic")


# ── low-level I/O ────────────────────────────────────────────────────────────

def _load_records(path: str) -> Iterable[Dict[str, Any]]:
    """Yield dicts from a .jsonl or .json file."""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(data, dict):
            for key in ("data", "items", "prompts"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            yield item
                    return


def _extract_prompt(record: Dict[str, Any]) -> Optional[str]:
    """Return the first non-empty prompt-like field."""
    for field in PROMPT_FIELDS:
        val = record.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_categories(record: Dict[str, Any]) -> Set[str]:
    """Return all category codes / labels found in the record."""
    cats: Set[str] = set()
    for key in CATEGORY_FIELDS:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            # Some datasets use comma-separated codes like "T,I,P"
            for part in val.split(","):
                part = part.strip()
                if part:
                    cats.add(part)
        elif isinstance(val, list):
            cats.update(str(v).strip() for v in val if str(v).strip())
    return cats


def _resolve_category_name(code: str) -> str:
    """Map a short code to its human-readable name.

    Returns ``None`` for codes that are not in :data:`SAFETY_TAXONOMY`
    and do not look like a real category name (i.e. short uppercase
    strings that are clearly codes rather than natural-language labels).
    """
    resolved = SAFETY_TAXONOMY.get(code)
    if resolved is not None:
        return resolved
    if len(code) <= 4 and code == code.upper():
        return None
    return code


# ── public API ───────────────────────────────────────────────────────────────

def build_seed_topics(prompt_dir: str, include_all: bool = True) -> Dict[str, Any]:
    """
    Scan *prompt_dir* for .jsonl / .json files and extract:
      - A list of unique harm-category seed topics (human-readable).
      - Per-dataset category frequency counts.
      - Total prompt counts per dataset.

    Parameters
    ----------
    prompt_dir : str
        Directory containing the benchmark prompt files.
    include_all : bool
        If True, include all datasets.  If False, restrict to a curated subset.

    Returns
    -------
    dict  with keys ``seed_topics``, ``category_counts``, ``dataset_stats``
    """
    seed_topics: Set[str] = set()
    category_counts: Counter = Counter()
    dataset_stats: Dict[str, Dict[str, Any]] = {}

    for filename in sorted(os.listdir(prompt_dir)):
        if not (filename.endswith(".jsonl") or filename.endswith(".json")):
            continue
        dataset_name = os.path.splitext(filename)[0]
        path = os.path.join(prompt_dir, filename)

        ds_categories: Set[str] = set()
        n_prompts = 0

        for record in _load_records(path):
            n_prompts += 1
            cats = _extract_categories(record)
            ds_categories.update(cats)
            category_counts.update(cats)

        resolved = set()
        unresolved = set()
        for code in ds_categories:
            name = _resolve_category_name(code)
            if name is None:
                unresolved.add(code)
                continue
            resolved.add(name)
            seed_topics.add(name)

        if unresolved:
            print(f"  [seed_topics] {dataset_name}: skipping unknown codes {sorted(unresolved)}")

        dataset_stats[dataset_name] = {
            "num_prompts": n_prompts,
            "raw_categories": sorted(ds_categories),
            "resolved_categories": sorted(resolved),
        }

    return {
        "seed_topics": sorted(seed_topics),
        "category_counts": dict(category_counts.most_common()),
        "dataset_stats": dataset_stats,
    }


def build_existing_prompts(prompt_dir: str) -> List[Dict[str, Any]]:
    """
    Read all prompts from *prompt_dir* into a flat list of dicts.

    Each entry has keys: ``dataset``, ``prompt``, ``categories``, ``id``.
    """
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
            resolved_cats = [_resolve_category_name(c) for c in cats]
            resolved_cats = [c for c in resolved_cats if c is not None]
            prompts.append({
                "id": record.get("id", idx),
                "dataset": dataset_name,
                "prompt": text,
                "categories": sorted(set(resolved_cats)),
                "raw_categories": sorted(cats),
            })
            idx += 1

    return prompts


def write_seed_artifacts(
    prompt_dir: str,
    output_dir: str,
    service: str = "safety-alignment",
) -> Tuple[str, str]:
    """
    Write seed_topics.json and existing_prompts.jsonl to *output_dir*.

    Returns the two output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── seed topics ──────────────────────────────────────────────────────
    topic_payload = build_seed_topics(prompt_dir)
    topic_payload["service"] = service
    topic_path = os.path.join(output_dir, "seed_topics.json")
    with open(topic_path, "w", encoding="utf-8") as fh:
        json.dump(topic_payload, fh, indent=2)

    # ── existing prompts ─────────────────────────────────────────────────
    prompts = build_existing_prompts(prompt_dir)
    prompts_path = os.path.join(output_dir, "existing_prompts.jsonl")
    with open(prompts_path, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[seed_topics] Wrote {len(topic_payload['seed_topics'])} seed topics to {topic_path}")
    print(f"[existing_prompts] Wrote {len(prompts)} prompts to {prompts_path}")
    return topic_path, prompts_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_default_dirs(
    prompt_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Return sensible defaults rooted at the project repo."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if prompt_dir is None:
        # Try aiXamine safety-alignment prompts
        candidate = os.path.join(
            repo_root, "..", "aiXamine",
            "airflow-tasks", "services", "safety-alignment", "prompts",
        )
        if os.path.isdir(candidate):
            prompt_dir = candidate
        else:
            prompt_dir = os.path.join(repo_root, "data", "services", "safety-alignment", "prompts")

    if output_dir is None:
        output_dir = os.path.join(repo_root, "data", "curation", "safety-alignment")

    return prompt_dir, output_dir


def main(
    prompt_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    service: str = "safety-alignment",
) -> None:
    prompt_dir, output_dir = _resolve_default_dirs(prompt_dir, output_dir)
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    write_seed_artifacts(prompt_dir, output_dir, service=service)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract safety seed topics from aiXamine benchmarks")
    parser.add_argument("--prompt-dir", default=None, help="Path to safety-alignment prompts directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for seed artifacts")
    parser.add_argument("--service", default="safety-alignment")
    args = parser.parse_args()
    main(prompt_dir=args.prompt_dir, output_dir=args.output_dir, service=args.service)
