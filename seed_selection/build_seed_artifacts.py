import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROMPT_FIELDS = ("prompt", "question")
CATEGORY_FIELDS = ("category", "type", "domain", "topic")


def _load_records(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        yield item
            return


def _extract_categories(record: Dict[str, Any]) -> Set[str]:
    categories: Set[str] = set()
    for key in CATEGORY_FIELDS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            categories.add(value.strip())
        elif isinstance(value, list):
            categories.update([str(v).strip() for v in value if str(v).strip()])
    return categories


def _has_prompt(record: Dict[str, Any]) -> bool:
    return any(field in record for field in PROMPT_FIELDS)


def _is_factuality_dataset(name: str) -> bool:
    return True


def _seed_source_dataset(name: str) -> bool:
    return name.lower() in {"simpleqa", "truthfulqa"}


def build_seed_topics(prompt_dir: str) -> List[str]:
    seed_topics: Set[str] = set()

    for filename in sorted(os.listdir(prompt_dir)):
        if not (filename.endswith(".jsonl") or filename.endswith(".json")):
            continue
        dataset_name = os.path.splitext(filename)[0]
        if not _seed_source_dataset(dataset_name):
            continue
        path = os.path.join(prompt_dir, filename)

        categories: Set[str] = set()
        for record in _load_records(path):
            categories.update(_extract_categories(record))

        for category in sorted(categories):
            seed_topics.add(category)

    return sorted(seed_topics)


def write_seed_topics(prompt_dir: str, output_dir: str, service: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "service": service,
        "seed_topics": build_seed_topics(prompt_dir),
    }
    output_path = os.path.join(output_dir, "seed_topics.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def write_factuality_questions(prompt_dir: str, output_dir: str, factuality_only: bool = True) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "factuality_questions.jsonl")
    with open(output_path, "w", encoding="utf-8") as handle:
        for filename in sorted(os.listdir(prompt_dir)):
            if not (filename.endswith(".jsonl") or filename.endswith(".json")):
                continue
            dataset_name = os.path.splitext(filename)[0]
            if factuality_only and not _is_factuality_dataset(dataset_name):
                continue
            path = os.path.join(prompt_dir, filename)

            for record in _load_records(path):
                question = record.get("question")
                prompt = question if question else record.get("prompt")
                if not prompt:
                    continue
                entry = {
                    "dataset": dataset_name,
                    "prompt": prompt,
                }
                if "question_id" in record:
                    entry["id"] = record.get("question_id")
                elif "id" in record:
                    entry["id"] = record.get("id")
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return output_path


def _resolve_paths(service: str, prompt_dir: Optional[str] = None, output_dir: Optional[str] = None) -> Tuple[str, str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(repo_root)
    resolved_prompt_dir = prompt_dir or os.path.join(
        project_root,
        "data",
        "services",
        service,
        "factuality",
    )
    resolved_output_dir = output_dir or os.path.join(project_root, "data", "curation", service)
    return resolved_prompt_dir, resolved_output_dir


def main(service: str = "hallucination", prompt_dir: Optional[str] = None, output_dir: Optional[str] = None, factuality_only: bool = True) -> None:
    prompt_dir, output_dir = _resolve_paths(service, prompt_dir, output_dir)
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    seed_path = write_seed_topics(prompt_dir, output_dir, service)
    factuality_path = write_factuality_questions(prompt_dir, output_dir, factuality_only=factuality_only)
    print(f"Seed topics written to {seed_path}")
    print(f"Factuality questions written to {factuality_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="hallucination")
    parser.add_argument("--prompt-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--factuality-only", action="store_true", default=True)
    parser.add_argument("--all-datasets", action="store_true")
    args = parser.parse_args()
    factuality_only = args.factuality_only and not args.all_datasets
    main(service=args.service, prompt_dir=args.prompt_dir, output_dir=args.output_dir, factuality_only=factuality_only)
