"""
Core functionality for the Over-Refusal Benchmark Engine.

Category generation, benign question generation (grounded in existing benign
examples via source mining and boundary mutation operators), and quality validation.

Mirrors ``sspbench.safety.safety_core`` but targets over-refusal testing
(finding benign boundary questions that models incorrectly refuse).

Pipeline comparison
-------------------
Safety engine:
    categories -> mine_source_prompts() -> build_source_context() -> generate_grounded_safety_prompts()
                       ^ retrieve harmful         ^ format context         ^ generate harmful prompts
                         existing prompts
               + apply_safety_mutations() (persona, hypothetical, etc.)

Over-Refusal engine (this file):
    categories -> mine_benign_examples() -> build_benign_context() -> generate_grounded_or_questions()
                       ^ retrieve benign          ^ format context        ^ generate benign questions
                         boundary examples
               + apply_boundary_mutations() (remove context, add sensitivity, etc.)

Key difference: We mine BENIGN examples and push toward boundary while maintaining benign score ≥ 8.0
"""

import os
import json
import copy
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Callable

from ..utils.llm_utils import gen_from_prompt
from ..novelty.json_utils import extract_json_v2, parse_json_response

from .or_config import (
    DEFAULT_JSON_MESSAGE,
    MAX_JSON_RETRY_ATTEMPTS,
    BENIGN_THRESHOLD,
    BOUNDARY_THRESHOLD,
    QUALITY_THRESHOLD,
    DEFAULT_REFUSAL_TARGET,
    OR_TOPICS,
    OR_LEGITIMACY,
    OR_BOUNDARIES,
    OR_CATEGORY_GENERATION_PROMPT,
    OR_CATEGORY_REFINEMENT_PROMPT,
    OR_CATEGORY_REFINEMENT_WITH_COVERAGE_PROMPT,
    OR_QUESTION_GENERATION_SYSTEM,
    OR_QUESTION_GENERATION_PROMPT,
    OR_BENIGN_VALIDATION_SYSTEM,
    OR_BENIGN_VALIDATION_PROMPT,
    OR_BOUNDARY_PROXIMITY_SYSTEM,
    OR_BOUNDARY_PROXIMITY_PROMPT,
)


# Valid taxonomy options
_VALID_TOPICS = set(OR_TOPICS.values())
_VALID_LEGITIMACY = set(OR_LEGITIMACY.values())
_VALID_BOUNDARIES = set(OR_BOUNDARIES.values())

# Keyword mapping for parent category inference
_KEYWORD_TO_TOPIC = {
    "hate": "Hate & Discrimination",
    "discrimin": "Hate & Discrimination",
    "violen": "Violence & Harm",
    "harm": "Violence & Harm",
    "sexual": "Sexual Content",
    "self-harm": "Self-Harm & Suicide",
    "suicide": "Self-Harm & Suicide",
    "illegal": "Illegal Activities",
    "privac": "Privacy & Surveillance",
    "surveil": "Privacy & Surveillance",
    "decep": "Deception & Misinformation",
    "misinform": "Deception & Misinformation",
    "unethic": "Unethical Behavior",
    "harass": "Harassment & Bullying",
    "bully": "Harassment & Bullying",
    "histor": "Sensitive Historical/Political",
    "politic": "Sensitive Historical/Political",
}


def _parse_refusal_target(refusal_target: str) -> tuple[float, float]:
    """Parse refusal target interval like '0.7--0.9' into (min, max) tuple."""
    try:
        parts = refusal_target.split("--")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, AttributeError):
        pass
    # Fallback to default
    return 0.7, 0.9


def _format_refusal_target_percent(refusal_target: str) -> str:
    """Format refusal target interval as percentage range, e.g., '70-90%'."""
    min_val, max_val = _parse_refusal_target(refusal_target)
    return f"{int(min_val * 100)}-{int(max_val * 100)}%"


def _normalize_category(cat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix parent_category, legitimacy_pattern, and boundary_mechanism
    on a category dict.
    """
    # Normalize parent_category (Topic dimension)
    parent = cat.get("parent_category", "")
    if parent not in _VALID_TOPICS:
        matched = None
        parent_lower = parent.lower()
        for keyword, topic in _KEYWORD_TO_TOPIC.items():
            if keyword in parent_lower:
                matched = topic
                break
        if matched is None:
            cat_lower = cat.get("category", "").lower()
            for keyword, topic in _KEYWORD_TO_TOPIC.items():
                if keyword in cat_lower:
                    matched = topic
                    break
        cat["parent_category"] = matched or "Violence & Harm"  # default

    # Ensure legitimacy_pattern and boundary_mechanism are present
    cat.setdefault("legitimacy_pattern", "L1 (Educational & Academic)")
    cat.setdefault("boundary_mechanism", "B6 (Professional Purpose)")
    cat.setdefault("additional_requirement", "")
    # Compute default from DEFAULT_REFUSAL_TARGET midpoint
    min_refusal, max_refusal = _parse_refusal_target(DEFAULT_REFUSAL_TARGET)
    default_refusal = (min_refusal + max_refusal) / 2
    cat.setdefault("expected_refusal_rate", default_refusal)

    return cat


# ===============================================================================
#  Benign Corpus Coverage Analysis
# ===============================================================================

def compute_benign_coverage(
    existing_examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute coverage of existing benign examples across taxonomy dimensions.

    Analogous to compute_source_coverage() in safety_core.py, but analyzes
    benign example distribution across legitimacy patterns and boundary mechanisms.

    Parameters
    ----------
    existing_examples : list
        List of benign examples with taxonomy annotations.

    Returns
    -------
    dict with keys:
        - legitimacy_counts: dict mapping legitimacy pattern -> count
        - boundary_counts: dict mapping boundary mechanism -> count
        - topic_counts: dict mapping topic -> count
        - total: total number of examples
    """
    legitimacy_counts: Dict[str, int] = defaultdict(int)
    boundary_counts: Dict[str, int] = defaultdict(int)
    topic_counts: Dict[str, int] = defaultdict(int)

    for ex in existing_examples:
        tax = ex.get("taxonomy", {})

        # Count by legitimacy pattern (WHY benign)
        # Support both code (e.g., 'L1') and name formats
        legitimacy = tax.get("legitimacy_name")
        if not legitimacy:
            legitimacy_code = tax.get("legitimacy", "")
            legitimacy = OR_LEGITIMACY.get(legitimacy_code, "Unknown")
        legitimacy_counts[legitimacy] += 1

        # Count by boundary mechanism (HOW safe)
        # Support both code (e.g., 'B1') and name formats
        boundary = tax.get("boundary_name")
        if not boundary:
            boundary_code = tax.get("boundary", "")
            boundary = OR_BOUNDARIES.get(boundary_code, "Unknown")
        boundary_counts[boundary] += 1

        # Count by topic (WHAT sensitive area)
        # Support both code (e.g., 'T1') and name formats
        topic = tax.get("topic_name")
        if not topic:
            topic_code = tax.get("topic", "")
            topic = OR_TOPICS.get(topic_code, "Unknown")
        topic_counts[topic] += 1

    return {
        "legitimacy_counts": dict(legitimacy_counts),
        "boundary_counts": dict(boundary_counts),
        "topic_counts": dict(topic_counts),
        "total": len(existing_examples),
    }


def format_benign_coverage(coverage: Dict[str, Any]) -> str:
    """Format benign coverage dict as a readable string for prompts."""
    if not coverage or coverage.get("total", 0) == 0:
        return "No existing benign examples available."

    lines = [f"Total benign examples: {coverage['total']}\n"]

    # Legitimacy patterns (most important for diversification)
    lines.append("Legitimacy Patterns (WHY benign):")
    for pattern, count in sorted(
        coverage.get("legitimacy_counts", {}).items(),
        key=lambda x: -x[1]
    ):
        lines.append(f"  - {pattern}: {count} examples")

    # Boundary mechanisms
    lines.append("\nBoundary Mechanisms (HOW safe):")
    for mechanism, count in sorted(
        coverage.get("boundary_counts", {}).items(),
        key=lambda x: -x[1]
    )[:8]:  # top 8
        lines.append(f"  - {mechanism}: {count} examples")

    # Topics
    lines.append("\nTopics (WHAT sensitive area):")
    for topic, count in sorted(
        coverage.get("topic_counts", {}).items(),
        key=lambda x: -x[1]
    )[:8]:  # top 8
        lines.append(f"  - {topic}: {count} examples")

    return "\n".join(lines)


# ===============================================================================
#  Wikipedia Context Mining
# ===============================================================================

_TOPIC_NAME_TO_CODE = {v: k for k, v in OR_TOPICS.items()}
_LEGITIMACY_NAME_TO_CODE = {v: k for k, v in OR_LEGITIMACY.items()}


def mine_wiki_context(
    category_dict: Dict[str, Any],
    wiki_articles: List[Dict[str, Any]],
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Select Wikipedia articles relevant to a category by taxonomy overlap.

    Matching strategy (scored, then ranked):
      - Topic code match (parent_category -> T-code): +10 per overlap
      - Legitimacy code match (legitimacy_pattern -> L-code): +5 per overlap
      - Keyword hits in title/summary: +1 per hit

    Returns a relevance-ordered subset capped at *max_results*.
    """
    if not wiki_articles:
        return []

    parent = category_dict.get("parent_category", "")
    cat_topic_code = _TOPIC_NAME_TO_CODE.get(parent)

    legitimacy_raw = category_dict.get("legitimacy_pattern", "")
    cat_legit_code = None
    for code, name in OR_LEGITIMACY.items():
        if code in legitimacy_raw or name in legitimacy_raw:
            cat_legit_code = code
            break

    cat_name = category_dict.get("category", "").lower()
    keywords = set()
    for word in cat_name.split():
        w = word.strip(",.;:'\"()[]{}").lower()
        if len(w) > 3:
            keywords.add(w)

    scored: List[tuple] = []
    for article in wiki_articles:
        score = 0.0
        topic_codes = set(article.get("topic_codes", []))
        legit_codes = set(article.get("legitimacy_codes", []))

        if cat_topic_code and cat_topic_code in topic_codes:
            score += 10.0
        if cat_legit_code and cat_legit_code in legit_codes:
            score += 5.0

        text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        kw_hits = sum(1 for kw in keywords if kw in text)
        score += kw_hits * 1.0

        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: -x[0])

    candidates = [s[1] for s in scored[: max_results * 3]]
    if len(candidates) <= max_results:
        return candidates
    top = candidates[: max_results // 2 + 1]
    rest = candidates[len(top):]
    rand_k = min(max_results - len(top), len(rest))
    return top + random.sample(rest, rand_k)


def build_wiki_context(
    wiki_articles: List[Dict[str, Any]],
    max_context_chars: int = 3000,
) -> str:
    """Format selected Wikipedia articles into a context block for the prompt.

    Uses summaries only (1-2 sentences each) for a compact representation,
    analogous to ``build_incident_context`` in the safety pipeline.
    """
    if not wiki_articles:
        return ""

    lines = []
    total = 0
    for i, article in enumerate(wiki_articles, 1):
        summary = article.get("summary", "")
        if not summary:
            continue
        title = article.get("title", "Unknown")
        line = f'  {i}. "{title}" -- {summary}'
        if total + len(line) > max_context_chars:
            break
        lines.append(line)
        total += len(line)

    if not lines:
        return ""

    header = (
        "=== WIKIPEDIA REFERENCE ARTICLES ===\n"
        "Use these real Wikipedia articles as grounding material. They demonstrate "
        "legitimate, encyclopedic coverage of sensitive topics. Generate questions "
        "that a curious reader of these articles might naturally ask.\n\n"
    )
    footer = "\n=== END WIKIPEDIA REFERENCE ARTICLES ===\n"
    return header + "\n".join(lines) + footer


# ===============================================================================
#  Category Generation / Refinement
# ===============================================================================

def _fuzzy_match_brainstorm(
    refined_name: str,
    broad: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> Optional[Dict[str, Any]]:
    """
    Find the closest brainstorm category for a refined category name.

    Mirrors the same function in safety_core.py.
    """
    best_score = 0.0
    best_match = None
    refined_lower = refined_name.lower()
    for b in broad:
        b_lower = b["category"].lower()
        score = SequenceMatcher(None, refined_lower, b_lower).ratio()
        if score > best_score:
            best_score = score
            best_match = b
    if best_score >= threshold:
        return best_match
    return None


def generate_or_categories(
    theme: str,
    agent_model,
    history: List[str],
    iteration: int,
    outfile_prefix: str = "att1",
    refusal_target: str = DEFAULT_REFUSAL_TARGET,
    num_categories: int = 10,
    existing_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate initial over-refusal categories for a theme / iteration.

    If a cached file exists at ``{outfile_prefix}.categories.json`` it is
    loaded directly (idempotent re-runs).

    Returns a list of category dicts with keys:
        id, category, parent_category, legitimacy_pattern, boundary_mechanism,
        additional_requirement, expected_refusal_rate, why_challenging
    """
    cache_path = f"{outfile_prefix}.categories.json"
    if os.path.exists(cache_path):
        print(f"[categories] Loading cached categories from {cache_path}")
        return json.load(open(cache_path, "r"))

    history_block = ""
    if iteration == 1:
        history_block = "This is iteration 1. Generate a broad initial set of benign boundary categories."
    else:
        history_block = (
            "Previous iteration results:\n" + "\n".join(history)
            + f"\n\nThis is iteration {iteration}. Refine the categories to approach the target refusal rate."
            + "\nDO NOT repeat categories from prior iterations."
        )

    prompt = OR_CATEGORY_GENERATION_PROMPT.format(
        num_categories=num_categories,
        theme=theme,
        refusal_target=refusal_target,
        refusal_target_percent=_format_refusal_target_percent(refusal_target),
        history_block=history_block,
    )

    # Similar token allocation as safety engine
    category_max_tokens = max(4000, num_categories * 200)

    for attempt in range(MAX_JSON_RETRY_ATTEMPTS):
        try:
            response = gen_from_prompt(
                agent_model, prompt,
                temperature=0.7, max_tokens=category_max_tokens,
                system_prompt=DEFAULT_JSON_MESSAGE,
            )
            with open(f"{outfile_prefix}.full_thoughts.txt", "w", encoding="utf-8") as fh:
                fh.write(prompt + "\n" + "=" * 80 + "\n" + response)

            categories = extract_json_v2(response, cache_path)
            for cat in categories:
                _normalize_category(cat)
            return categories

        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  Attempt {attempt + 1}/{MAX_JSON_RETRY_ATTEMPTS}: JSON parse failed - {str(exc)[:120]}")
            if attempt == MAX_JSON_RETRY_ATTEMPTS - 1:
                print("  Returning empty category list")
                return []
            prompt += "\n\nIMPORTANT: Output ONLY the JSON block, starting with ```json"

    return []


def refine_or_categories(
    theme: str,
    agent_model,
    history: List[str],
    iteration: int,
    outfile_prefix: str = "att1",
    refusal_target: str = DEFAULT_REFUSAL_TARGET,
    num_categories: int = 10,
    existing_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    First generate broad categories, then refine by selecting those most
    likely to produce tough benign boundary questions.

    Analogous to refine_safety_categories() but for benign content:
      - Safety: brainstorm -> harm family coverage -> LLM selection
      - Over-Refusal: brainstorm -> legitimacy/boundary coverage -> LLM selection

    If *existing_examples* is provided, uses coverage-aware refinement
    (the ``OR_CATEGORY_REFINEMENT_WITH_COVERAGE_PROMPT``). Otherwise
    falls back to basic refinement.
    """
    # Step 1: broad generation (over-generate)
    broad = generate_or_categories(
        theme, agent_model, history, iteration,
        outfile_prefix=outfile_prefix + ".brainstorm",
        refusal_target=refusal_target,
        num_categories=num_categories * 3,
    )

    if not broad:
        return []

    # Step 2: collect unique candidates (full objects, not just names)
    candidates = []
    seen = set()
    for cat in broad:
        name = cat["category"]
        if name not in seen:
            candidates.append(cat)
            seen.add(name)

    random.shuffle(candidates)

    # Save unique candidates for logging
    with open(f"{outfile_prefix}.unique_candidates.json", "w") as fh:
        json.dump(candidates, fh, indent=2)
    print(f"[categories] Collected {len(candidates)} unique candidates")

    # Compact representation for the refinement prompt — include all metadata
    # so the LLM can make informed selections without re-generating fields
    candidate_summaries = []
    for cat in candidates:
        candidate_summaries.append({
            "category": cat["category"],
            "parent_category": cat.get("parent_category", ""),
            "legitimacy_pattern": cat.get("legitimacy_pattern", ""),
            "boundary_mechanism": cat.get("boundary_mechanism", ""),
            "additional_requirement": cat.get("additional_requirement", ""),
            "expected_refusal_rate": cat.get("expected_refusal_rate", 0.8),
        })

    # Step 3: LLM-based refinement / selection
    history_block = ""
    if iteration > 1:
        history_block = "Previous iteration results:\n" + "\n".join(history)

    # Choose refinement prompt based on whether we have coverage data
    if existing_examples:
        coverage = compute_benign_coverage(existing_examples)
        benign_coverage_str = format_benign_coverage(coverage)

        # Save coverage for logging
        with open(f"{outfile_prefix}.benign_coverage.json", "w") as fh:
            json.dump(coverage, fh, indent=2)

        prompt = OR_CATEGORY_REFINEMENT_WITH_COVERAGE_PROMPT.format(
            num_categories=num_categories,
            theme=theme,
            refusal_target=refusal_target,
            refusal_target_percent=_format_refusal_target_percent(refusal_target),
            candidates=json.dumps(candidate_summaries, indent=2),
            benign_coverage=benign_coverage_str,
            history_block=history_block,
        )
    else:
        prompt = OR_CATEGORY_REFINEMENT_PROMPT.format(
            num_categories=num_categories,
            theme=theme,
            refusal_target=refusal_target,
            refusal_target_percent=_format_refusal_target_percent(refusal_target),
            candidates=json.dumps(candidate_summaries, indent=2),
            history_block=history_block,
        )

    cache_path = f"{outfile_prefix}.refined_categories.json"
    if os.path.exists(cache_path):
        print(f"[categories] Loading cached refined categories from {cache_path}")
        return json.load(open(cache_path, "r"))

    for attempt in range(MAX_JSON_RETRY_ATTEMPTS):
        try:
            refine_max_tokens = min(max(4000, num_categories * 300), 16384)
            response = gen_from_prompt(
                agent_model, prompt,
                temperature=0.3, max_tokens=refine_max_tokens,
                system_prompt=DEFAULT_JSON_MESSAGE,
            )
            with open(f"{outfile_prefix}.refine_thoughts.txt", "w", encoding="utf-8") as fh:
                fh.write(prompt + "\n" + "=" * 80 + "\n" + response)

            refined = extract_json_v2(response, cache_path)
            for cat in refined:
                _normalize_category(cat)

            # Match back to brainstorm to fill any missing fields
            brainstorm_lookup = {b["category"]: b for b in broad}
            for cat in refined:
                orig = brainstorm_lookup.get(cat["category"])
                if orig is None:
                    orig = _fuzzy_match_brainstorm(cat["category"], broad)
                if orig:
                    cat["brainstorm_match"] = orig["category"]
                    # Inherit brainstorm fields only when refinement left them empty
                    for field in ("parent_category", "legitimacy_pattern",
                                  "boundary_mechanism", "additional_requirement",
                                  "why_challenging"):
                        if not cat.get(field) and orig.get(field):
                            cat[field] = orig[field]

            return refined

        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  Refinement attempt {attempt + 1}/{MAX_JSON_RETRY_ATTEMPTS}: {str(exc)[:120]}")
            if attempt == MAX_JSON_RETRY_ATTEMPTS - 1:
                print("  Refinement failed - falling back to broad categories")
                return broad[:num_categories]
            prompt += "\n\nIMPORTANT: Output ONLY the JSON block."

    return broad[:num_categories]


# ===============================================================================
#  Benign Example Mining (analogous to mine_source_prompts in safety)
# ===============================================================================

def mine_benign_examples(
    category_dict: Dict[str, Any],
    existing_examples: List[Dict[str, Any]],
    max_results: int = 15,
    strategy: str = "hybrid",
) -> List[Dict[str, Any]]:
    """
    Mine relevant benign examples from existing corpus for a category.

    Analogous to mine_source_prompts() in safety mutations, but mines
    BENIGN examples instead of harmful prompts.

    Strategy:
        - "keyword": Match on category keywords in prompt text
        - "taxonomy": Match on legitimacy/boundary/topic taxonomy
        - "hybrid": Combine both approaches (default)

    Returns list of relevant benign examples (with metadata).
    """
    category_name = category_dict.get("category", "").lower()
    parent_category = category_dict.get("parent_category", "").lower()
    legitimacy = category_dict.get("legitimacy_pattern", "").lower()
    boundary = category_dict.get("boundary_mechanism", "").lower()

    # Extract keywords from category name
    keywords = set(category_name.split())
    keywords.update(parent_category.split())
    keywords -= {"&", "and", "the", "of", "in", "a", "an", "to", "for"}

    scored_examples = []

    for ex in existing_examples:
        score = 0.0
        prompt_text = ex.get("prompt", "").lower()
        tax = ex.get("taxonomy", {})

        # Keyword matching
        if strategy in ["keyword", "hybrid"]:
            keyword_matches = sum(1 for kw in keywords if kw in prompt_text)
            score += keyword_matches * 2.0

        # Taxonomy matching
        if strategy in ["taxonomy", "hybrid"]:
            ex_legitimacy = tax.get("legitimacy_name", "").lower()
            ex_boundary = tax.get("boundary_name", "").lower()
            ex_topic = tax.get("topic_name", "").lower()

            # Exact legitimacy match (most important)
            if legitimacy and legitimacy in ex_legitimacy:
                score += 10.0

            # Exact boundary match
            if boundary and boundary in ex_boundary:
                score += 5.0

            # Topic match
            if parent_category and parent_category in ex_topic:
                score += 3.0

        if score > 0:
            ex_copy = copy.deepcopy(ex)
            ex_copy["mining_score"] = score
            scored_examples.append(ex_copy)

    # Sort by score and return top results
    scored_examples.sort(key=lambda x: x["mining_score"], reverse=True)
    return scored_examples[:max_results]


def build_benign_context(
    benign_examples: List[Dict[str, Any]],
    max_examples: int = 10,
) -> str:
    """
    Build a context string from benign examples for grounded generation.

    Analogous to build_source_context() in safety mutations.
    """
    if not benign_examples:
        return "No relevant benign examples found."

    lines = []
    for i, ex in enumerate(benign_examples[:max_examples], 1):
        prompt = ex.get("prompt", "")
        source = ex.get("source_benchmark", "Unknown")
        tax = ex.get("taxonomy", {})
        legitimacy = tax.get("legitimacy_name", "")
        boundary = tax.get("boundary_name", "")

        lines.append(f"Example {i} ({source}):")
        lines.append(f"  Prompt: {prompt}")
        lines.append(f"  Legitimacy: {legitimacy}")
        lines.append(f"  Boundary: {boundary}")
        lines.append("")

    return "\n".join(lines)


# ===============================================================================
#  Question Generation (grounded in benign examples)
# ===============================================================================

def generate_grounded_or_questions(
    category_dict: Dict[str, Any],
    agent_model,
    source_context: str,
    wiki_context: str = "",
    num_prompts: int = 5,
    outfile_prefix: str = "att1",
    benign_threshold: float = BENIGN_THRESHOLD,
    refusal_target: str = DEFAULT_REFUSAL_TARGET,
) -> List[Dict[str, Any]]:
    """
    Generate benign boundary questions grounded in existing benign examples.

    Analogous to generate_grounded_safety_prompts() but for benign content.
    """
    prompt = OR_QUESTION_GENERATION_PROMPT.format(
        num_prompts=num_prompts,
        category=category_dict["category"],
        parent_category=category_dict.get("parent_category", ""),
        legitimacy_pattern=category_dict.get("legitimacy_pattern", ""),
        additional_requirement=category_dict.get("additional_requirement", ""),
        source_context=source_context,
        wiki_context=wiki_context,
        benign_threshold=benign_threshold,
        refusal_target=refusal_target,
    )

    for attempt in range(MAX_JSON_RETRY_ATTEMPTS):
        try:
            response = gen_from_prompt(
                agent_model, prompt,
                temperature=0.8, max_tokens=3000,
                system_prompt=OR_QUESTION_GENERATION_SYSTEM,
            )

            generated = extract_json_v2(response, None)
            for q in generated:
                q.setdefault("category", category_dict["category"])
                q.setdefault("parent_category", category_dict.get("parent_category", ""))
                q.setdefault("legitimacy_pattern", category_dict.get("legitimacy_pattern", ""))
                q.setdefault("boundary_mechanism", category_dict.get("boundary_mechanism", ""))
                q.setdefault("generation_method", "grounded")

            return generated

        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  Grounded generation attempt {attempt + 1}: {str(exc)[:120]}")
            if attempt == MAX_JSON_RETRY_ATTEMPTS - 1:
                return []
            prompt += "\n\nIMPORTANT: Output ONLY a valid JSON list."

    return []


def _generate_ungrounded_or_questions(
    category_dict: Dict[str, Any],
    agent_model,
    outfile_prefix: str,
    num_prompts: int = 5,
    existing_examples: Optional[List[Dict[str, Any]]] = None,
    wiki_context: str = "",
    benign_threshold: float = BENIGN_THRESHOLD,
    refusal_target: str = DEFAULT_REFUSAL_TARGET,
) -> List[Dict[str, Any]]:
    """
    Fallback: generate benign questions when source mining returned nothing.

    If *existing_examples* are available, a random sample is included as
    context so the LLM still sees real benign examples.
    """
    # Build context from random sample of existing examples
    context_block = ""
    if existing_examples:
        sample_size = min(10, len(existing_examples))
        sample = random.sample(existing_examples, sample_size)
        context_block = (
            "=== REFERENCE EXAMPLES (from existing benign benchmarks) ===\n"
            + build_benign_context(sample)
            + "\n=== END REFERENCE EXAMPLES ===\n\n"
            "Use these examples as references for benign boundary questions. "
            "Generate NEW questions for the category above — do NOT copy."
        )

    prompt = OR_QUESTION_GENERATION_PROMPT.format(
        num_prompts=num_prompts,
        category=category_dict["category"],
        parent_category=category_dict.get("parent_category", ""),
        legitimacy_pattern=category_dict.get("legitimacy_pattern", ""),
        additional_requirement=category_dict.get("additional_requirement", ""),
        source_context=context_block,
        wiki_context=wiki_context,
        benign_threshold=benign_threshold,
        refusal_target=refusal_target,
    )

    for attempt in range(MAX_JSON_RETRY_ATTEMPTS):
        try:
            response = gen_from_prompt(
                agent_model, prompt,
                temperature=0.8, max_tokens=3000,
                system_prompt=OR_QUESTION_GENERATION_SYSTEM,
            )
            generated = extract_json_v2(response, None)
            for q in generated:
                q.setdefault("category", category_dict["category"])
                q.setdefault("parent_category", category_dict.get("parent_category", ""))
                q.setdefault("legitimacy_pattern", category_dict.get("legitimacy_pattern", ""))
                q.setdefault("generation_method", "ungrounded")
            return generated
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  Ungrounded generation attempt {attempt + 1}: {str(exc)[:120]}")
            if attempt == MAX_JSON_RETRY_ATTEMPTS - 1:
                return []
            prompt += "\n\nIMPORTANT: Output ONLY a valid JSON list."
    return []


def generate_or_questions(
    category_dict: Dict[str, Any],
    agent_model,
    outfile_prefix: str,
    num_prompts: int = 5,
    existing_examples: Optional[List[Dict[str, Any]]] = None,
    wiki_articles: Optional[List[Dict[str, Any]]] = None,
    mutations_per_source: int = 2,
    max_source_examples: int = 15,
    benign_threshold: float = BENIGN_THRESHOLD,
    refusal_target: str = DEFAULT_REFUSAL_TARGET,
) -> List[Dict[str, Any]]:
    """
    Generate benign boundary questions for a single category using:

    1. **Source mining** - retrieve relevant benign examples
    2. **Wiki context mining** - retrieve relevant Wikipedia articles
    3. **Grounded generation** - generate new questions from source + wiki context
    4. **Mutation** - apply boundary mutations (handled by or_mutations.py)

    Analogous to generate_safety_prompts() but for benign content.

    Returns list of question dicts with keys:
        id, question, category, legitimacy_pattern, boundary_mechanism,
        expected_benign_score, expected_refusal_rate, generation_method
    """
    all_generated: List[Dict[str, Any]] = []

    # -- Step A: Mine benign examples -------------------------------------------
    mined = mine_benign_examples(
        category_dict, existing_examples or [],
        max_results=max_source_examples,
        strategy="hybrid",
    )
    print(f"   Mined {len(mined)} benign examples for '{category_dict['category']}'")

    # Save mined examples for logging
    with open(f"{outfile_prefix}.mined_examples.json", "w") as fh:
        json.dump(mined, fh, indent=2, default=str)

    # -- Step A2: Mine Wikipedia context ----------------------------------------
    mined_wiki = mine_wiki_context(category_dict, wiki_articles or [], max_results=5)
    wiki_ctx = build_wiki_context(mined_wiki)
    if mined_wiki:
        print(f"   Mined {len(mined_wiki)} Wikipedia articles")
        with open(f"{outfile_prefix}.mined_wiki.json", "w") as fh:
            json.dump(
                [{"title": a["title"], "topic_codes": a.get("topic_codes", [])}
                 for a in mined_wiki],
                fh, indent=2,
            )

    # -- Step B: Grounded generation -------------------------------------------
    if mined:
        source_context = build_benign_context(mined)
        grounded = generate_grounded_or_questions(
            category_dict, agent_model,
            source_context=source_context,
            wiki_context=wiki_ctx,
            num_prompts=num_prompts,
            outfile_prefix=outfile_prefix,
            benign_threshold=benign_threshold,
            refusal_target=refusal_target,
        )
        print(f"   Grounded generation: {len(grounded)} new questions")
        all_generated.extend(grounded)
    else:
        # No source material -> fall back to ungrounded generation
        print(f"   No source material found -> falling back to ungrounded generation")
        grounded = _generate_ungrounded_or_questions(
            category_dict, agent_model, outfile_prefix, num_prompts,
            existing_examples=existing_examples,
            wiki_context=wiki_ctx,
            benign_threshold=benign_threshold,
            refusal_target=refusal_target,
        )
        all_generated.extend(grounded)

    # -- Step C: Mutations (will be handled by or_mutations.py) ---------------
    # NOTE: Mutations will be added in or_mutations.py module
    # For now, we just do grounded generation

    # -- Assign sequential IDs -------------------------------------------------
    for i, q in enumerate(all_generated):
        q["id"] = f"{category_dict.get('id', '?')}_{i + 1}"

    # Save per-category generated questions for logging
    with open(f"{outfile_prefix}.generated_questions.json", "w") as fh:
        json.dump(all_generated, fh, indent=2)

    return all_generated


# ===============================================================================
#  Quality Validation
# ===============================================================================

def validate_benign_question(
    question: Dict[str, Any],
    eval_model,
    benign_threshold: float = BENIGN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Validate that a question is GENUINELY BENIGN.

    This is CRITICAL - we must ensure all questions are safe and legitimate.

    Returns the question dict with added validation metadata:
        - is_benign: bool
        - benign_score: float (0-10)
        - validation_concerns: str
    """
    prompt_text = question.get("question", "")

    judge_prompt = OR_BENIGN_VALIDATION_PROMPT.format(
        question=prompt_text,
        benign_threshold=benign_threshold
    )

    try:
        response = gen_from_prompt(
            eval_model, judge_prompt,
            temperature=0.0, max_tokens=500,
            system_prompt=OR_BENIGN_VALIDATION_SYSTEM,
        )
        if isinstance(response, list):
            response = response[0]

        fallback = {
            "is_benign": False,
            "benign_score": 0.0,
            "legitimacy_pattern": "",
            "context_markers": [],
            "concerns": "parse_error",
            "rationale": "",
        }
        parsed = parse_json_response(response, fallback)

        question["is_benign"] = bool(parsed.get("is_benign", False))
        question["benign_score"] = float(parsed.get("benign_score", 0.0))
        question["validation_legitimacy"] = parsed.get("legitimacy_pattern", "")
        question["validation_concerns"] = parsed.get("concerns", "")
        question["validation_rationale"] = parsed.get("rationale", "")

        # Check against threshold
        question["passes_benign_validation"] = (
            question["benign_score"] >= benign_threshold
        )

    except Exception as exc:
        question["is_benign"] = False
        question["benign_score"] = 0.0
        question["validation_concerns"] = f"error: {exc}"
        question["passes_benign_validation"] = False

    return question


def check_boundary_proximity(
    question: Dict[str, Any],
    eval_model,
) -> Dict[str, Any]:
    """
    Evaluate how close a benign question is to the safety boundary.

    Returns the question dict with added proximity metadata:
        - boundary_proximity: float (0-10)
        - sensitive_terms: list
        - expected_refusal_rate: float
    """
    prompt_text = question.get("question", "")

    judge_prompt = OR_BOUNDARY_PROXIMITY_PROMPT.format(question=prompt_text)

    try:
        response = gen_from_prompt(
            eval_model, judge_prompt,
            temperature=0.0, max_tokens=500,
            system_prompt=OR_BOUNDARY_PROXIMITY_SYSTEM,
        )
        if isinstance(response, list):
            response = response[0]

        fallback = {
            "boundary_proximity": 0.0,
            "sensitive_terms": [],
            "expected_refusal_rate": 0.0,
            "why_challenging": "",
            "distinguishing_features": "",
        }
        parsed = parse_json_response(response, fallback)

        question["boundary_proximity"] = float(parsed.get("boundary_proximity", 0.0))
        question["sensitive_terms"] = parsed.get("sensitive_terms", [])
        question["eval_expected_refusal"] = float(parsed.get("expected_refusal_rate", 0.0))
        question["why_challenging"] = parsed.get("why_challenging", "")
        question["distinguishing_features"] = parsed.get("distinguishing_features", "")

    except Exception as exc:
        question["boundary_proximity"] = 0.0
        question["validation_error"] = f"boundary_check_error: {exc}"

    return question


# ===============================================================================
#  Full Pipeline (single iteration)
# ===============================================================================

def generate_full_or_questions(
    theme: str,
    agent_model,
    history: List[str],
    iteration: int,
    outfile_prefix: str = "att1",
    refusal_target: str = DEFAULT_REFUSAL_TARGET,
    max_categories: int = 10,
    num_prompts_per_category: int = 5,
    existing_examples: Optional[List[Dict[str, Any]]] = None,
    wiki_articles: Optional[List[Dict[str, Any]]] = None,
    category_gen_func: Optional[Callable] = None,
    eval_model=None,
    benign_threshold: float = BENIGN_THRESHOLD,
    mutations_per_source: int = 2,
) -> List[Dict[str, Any]]:
    """
    Main generation pipeline for a single iteration (over-refusal analogue of
    generate_full_safety_prompts).

    Generates benign boundary questions. Filtering (benign validation,
    boundary proximity, quality) is handled by or_engine.py, mirroring
    how generate_full_safety_prompts returns raw prompts and safety_engine.py
    does the filtering.

    Returns the list of generated questions (pre-filter, with all metadata).
    """
    os.makedirs(os.path.dirname(outfile_prefix) or ".", exist_ok=True)

    # -- Step 1: categories (coverage-aware) -----------------------------------
    if category_gen_func is None:
        category_gen_func = refine_or_categories

    categories = category_gen_func(
        theme, agent_model, history, iteration,
        outfile_prefix=outfile_prefix,
        refusal_target=refusal_target,
        num_categories=max_categories,
        existing_examples=existing_examples,
    )

    with open(f"{outfile_prefix}.categories.json", "w") as fh:
        json.dump(categories, fh, indent=2)
    print(f"\n[iter {iteration}] Generated {len(categories)} categories")

    # Log benign corpus coverage
    if existing_examples:
        coverage = compute_benign_coverage(existing_examples)
        with open(f"{outfile_prefix}.benign_coverage.json", "w") as fh:
            json.dump(coverage, fh, indent=2)
        print(f"[iter {iteration}] Benign corpus coverage:")
        print(f"   Legitimacy patterns: {len(coverage['legitimacy_counts'])}")
        print(f"   Boundary mechanisms: {len(coverage['boundary_counts'])}")
        print(f"   Topics: {len(coverage['topic_counts'])}")

    if wiki_articles:
        print(f"[iter {iteration}] Wikipedia context: {len(wiki_articles)} articles available")

    # -- Step 2: generate questions (grounded + mutations) ---------------------
    all_questions: List[Dict[str, Any]] = []
    gen_stats = Counter()

    for cat in categories[:max_categories]:
        cat_id = cat.get("id", "?")
        print(f"\n  Category {cat_id}: {cat['category']}")

        questions = generate_or_questions(
            cat, agent_model,
            outfile_prefix=outfile_prefix + f"_cat{cat_id}",
            num_prompts=num_prompts_per_category,
            existing_examples=existing_examples,
            wiki_articles=wiki_articles,
            mutations_per_source=mutations_per_source,
            benign_threshold=benign_threshold,
            refusal_target=refusal_target,
        )
        for q in questions:
            gen_stats[q.get("generation_method", "unknown")] += 1
        print(f"   Total: {len(questions)} questions")
        all_questions.extend(questions)

    print(f"\n[iter {iteration}] Generated {len(all_questions)} total questions")
    print(f"  Generation methods: {dict(gen_stats)}")

    # Save raw generated questions (filtering in or_engine.py)
    with open(f"{outfile_prefix}.raw_questions.json", "w") as fh:
        json.dump(all_questions, fh, indent=2)

    return all_questions
