"""Core orchestrator coordinating generator/evaluator pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

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
        return runner(**kwargs)
