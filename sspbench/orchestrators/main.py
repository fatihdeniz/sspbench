"""Core orchestrator coordinating generator/evaluator pipelines."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..novelty.main import run_novelty_engine


class BenchmarkOrchestrator:
    """Thin orchestrator over available benchmark pipelines."""

    def __init__(
        self,
        pipelines: Optional[Dict[str, Callable[..., Any]]] = None,
        default_pipeline: str = "novelty",
    ) -> None:
        base_pipelines = {"novelty": run_novelty_engine}
        if pipelines:
            base_pipelines.update(pipelines)

        if default_pipeline not in base_pipelines:
            raise ValueError(f"Default pipeline '{default_pipeline}' is not registered")

        self._pipelines = base_pipelines
        self._default_pipeline = default_pipeline

    def available_pipelines(self) -> Iterable[str]:
        return tuple(self._pipelines.keys())

    def describe(self) -> Dict[str, Any]:
        return {"pipelines": self.available_pipelines()}

    def run(self, pipeline: Optional[str] = None, **kwargs: Any) -> Any:
        target = pipeline or self._default_pipeline
        runner = self._pipelines.get(target)
        if runner is None:
            raise ValueError(f"Unknown pipeline '{target}'. Available: {self.available_pipelines()}")
        seed_index_path = kwargs.pop("seed_index_path", None)
        if seed_index_path:
            seed_topics = _load_seed_topics_manifest(seed_index_path)
            results: Dict[str, Any] = {}
            for seed_topic in seed_topics:
                results[seed_topic] = runner(**kwargs, theme=seed_topic)
            return results
        return runner(**kwargs)


def _load_seed_topics_manifest(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    seed_topics = data.get("seed_topics") if isinstance(data, dict) else None
    if not seed_topics:
        raise ValueError(f"No seed_topics found in manifest: {path}")
    return list(seed_topics)
