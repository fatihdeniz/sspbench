"""Safety benchmark generation engine with adaptive testing.

The SafetyEngine wraps the safety novelty pipeline, providing a clean API for:
- Full pipeline (generate + evaluate)
- Generation-only mode
- Seed extraction mode
- Multi-theme support
- Resume from checkpoints
- Adaptive mutation selection

Examples:
    >>> from sspbench.engines import SafetyEngine
    >>>
    >>> # Simple usage - full pipeline
    >>> engine = SafetyEngine()
    >>> results = engine.run(theme="LLM safety alignment", max_iterations=5)
    >>>
    >>> # Generation only (no evaluation)
    >>> results = engine.run(theme="safety", mode="generate")
    >>>
    >>> # Resume from iteration 3
    >>> results = engine.run(start_iteration=3, max_iterations=8)
    >>>
    >>> # Multi-theme execution
    >>> results = engine.run(themes=["safety", "privacy", "bias"])
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine

logger = logging.getLogger(__name__)


class SafetyEngine(BaseEngine):
    """Safety benchmark generation engine.

    This engine implements adaptive safety-alignment benchmark generation
    through iterative red-teaming with:
    - Harm category generation
    - Source-grounded prompt generation
    - Mutation operators
    - Quality filtering
    - Adaptive feedback loops

    Attributes:
        config: BenchmarkConfig with all settings
        models: ModelManager for lazy model loading

    Examples:
        >>> # From command line
        >>> engine = SafetyEngine.from_cli()
        >>> results = engine.run()
        >>>
        >>> # Programmatic
        >>> from sspbench.config import BenchmarkConfig
        >>> config = BenchmarkConfig(
        ...     theme="LLM safety alignment",
        ...     max_iterations=5,
        ...     refusal_target="0.7--0.9"
        ... )
        >>> engine = SafetyEngine(config)
        >>> results = engine.run()
    """

    @classmethod
    def _get_parser(cls):
        """Get argument parser for safety engine."""
        from ..cli import get_safety_parser
        return get_safety_parser()

    def run(
        self,
        mode: str = "full",
        theme: Optional[str] = None,
        themes: Optional[List[str]] = None,
        seed_index_path: Optional[str] = None,
        max_iterations: Optional[int] = None,
        start_iteration: Optional[int] = None,
        refusal_target: Optional[str] = None,
        num_categories: Optional[int] = None,
        num_prompts: Optional[int] = None,
        quality_threshold: Optional[int] = None,
        mutations_per_source: Optional[int] = None,
        mutation_operators: Optional[List[str]] = None,
        seed_prompt_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        engine: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the safety benchmark generation pipeline.

        Args:
            mode: Execution mode - "full" (default), "generate", or "seeds"
            theme: Single theme (e.g., "LLM safety alignment")
            themes: List of themes (alternative to theme)
            seed_index_path: Path to JSON file with themes (alternative to theme/themes)
            max_iterations: Number of generate-evaluate-refine loops (default: 8)
            start_iteration: Resume from this iteration (default: 1)
            refusal_target: Target refusal rate range (default: "0.1--0.5")
            num_categories: Categories per iteration (default: 15)
            num_prompts: Prompts per category (default: 8)
            quality_threshold: Minimum quality score 0-10 (default: 6)
            mutations_per_source: Mutations per source prompt (default: 3)
            mutation_operators: List of mutation operator names (default: all)
            seed_prompt_dir: Path to existing prompts for seeds mode
            output_dir: Output directory (default: data/generated/safety_novelty)
            engine: Engine name for directory naming (default: safety_novelty)
            **kwargs: Additional parameters passed to underlying functions

        Returns:
            Dictionary with results based on mode:
            - full: {"history": [...], "summaries": [...], "metrics": [...], "all_prompts": [...]}
            - generate: {"all_prompts": [...], "output_dir": "..."}
            - seeds: {"output_dir": "...", "num_seeds": N}

        Examples:
            >>> # Full pipeline
            >>> engine = SafetyEngine()
            >>> results = engine.run(theme="safety", max_iterations=5)
            >>>
            >>> # Generate only
            >>> results = engine.run(mode="generate", theme="privacy")
            >>>
            >>> # Extract seeds
            >>> results = engine.run(mode="seeds", seed_prompt_dir="path/to/prompts")
            >>>
            >>> # Multi-theme
            >>> results = engine.run(themes=["safety", "privacy", "bias"])
            >>>
            >>> # Resume from checkpoint
            >>> results = engine.run(start_iteration=4, max_iterations=8)
        """
        # Use config defaults if not provided
        # Override "general knowledge" default with safety-appropriate theme
        if seed_index_path is None and themes is None:
            theme = theme or (
                "LLM safety alignment" if self.config.theme in ("general knowledge", None)
                else self.config.theme
            )
        max_iterations = max_iterations or self.config.max_iterations
        start_iteration = start_iteration or self.config.start_iteration
        refusal_target = refusal_target or getattr(self.config, 'refusal_target', '0.1--0.5')
        num_categories = num_categories or getattr(self.config, 'num_categories', 15)
        num_prompts = num_prompts or getattr(self.config, 'num_prompts_per_category', 8)
        quality_threshold = quality_threshold or getattr(self.config, 'quality_threshold', 6)
        mutations_per_source = mutations_per_source or getattr(self.config, 'mutations_per_source', 3)
        engine = engine or 'safety_novelty'
        output_dir = output_dir or self.config.output_dir

        # Validate theme selection (mutually exclusive)
        theme_args = [theme is not None, themes is not None, seed_index_path is not None]
        if sum(theme_args) > 1:
            raise ValueError(
                "Provide only one of: theme, themes, or seed_index_path"
            )

        # Handle seeds mode
        if mode == "seeds":
            return self._run_seeds_mode(seed_prompt_dir, output_dir)

        # Multi-theme execution
        if themes or seed_index_path:
            theme_list = themes or self._load_seed_topics(seed_index_path)
            return self._run_multiple_themes(
                mode=mode,
                themes=theme_list,
                max_iterations=max_iterations,
                start_iteration=start_iteration,
                refusal_target=refusal_target,
                num_categories=num_categories,
                num_prompts=num_prompts,
                quality_threshold=quality_threshold,
                mutations_per_source=mutations_per_source,
                mutation_operators=mutation_operators,
                seed_prompt_dir=seed_prompt_dir,
                output_dir=output_dir,
                engine=engine,
                **kwargs
            )

        # Single theme execution
        return self._run_single_theme(
            mode=mode,
            theme=theme,
            max_iterations=max_iterations,
            start_iteration=start_iteration,
            refusal_target=refusal_target,
            num_categories=num_categories,
            num_prompts=num_prompts,
            quality_threshold=quality_threshold,
            mutations_per_source=mutations_per_source,
            mutation_operators=mutation_operators,
            seed_prompt_dir=seed_prompt_dir,
            output_dir=output_dir,
            engine=engine,
            **kwargs
        )

    def _run_seeds_mode(
        self,
        seed_prompt_dir: Optional[str],
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Extract seed topics and existing prompts from aiXamine.

        Args:
            seed_prompt_dir: Path to aiXamine safety-alignment/prompts directory
            output_dir: Output directory for seed artifacts

        Returns:
            Dictionary with output_dir and num_seeds
        """
        from ..safety.safety_seeds import write_seed_artifacts

        # Get project root
        package_dir = Path(__file__).parent.parent
        project_root = package_dir.parent

        # Default prompt directory
        if not seed_prompt_dir:
            seed_prompt_dir = str(
                project_root.parent / "aiXamine" / "airflow-tasks" / "services" /
                "safety-alignment" / "prompts"
            )

        if not os.path.isdir(seed_prompt_dir):
            raise ValueError(f"Prompt directory not found: {seed_prompt_dir}")

        # Default output directory
        if not output_dir:
            output_dir = str(project_root / "data" / "seeds" / "safety")

        logger.info(f"Extracting seeds from: {seed_prompt_dir}")
        logger.info(f"Output directory: {output_dir}")

        write_seed_artifacts(seed_prompt_dir, output_dir)

        logger.info("✅ Seed extraction complete")

        return {
            "output_dir": output_dir,
            "num_seeds": "See output files for details"
        }

    def _run_single_theme(
        self,
        mode: str,
        theme: str,
        max_iterations: int,
        start_iteration: int,
        refusal_target: str,
        num_categories: int,
        num_prompts: int,
        quality_threshold: int,
        mutations_per_source: int,
        mutation_operators: Optional[List[str]],
        seed_prompt_dir: Optional[str],
        output_dir: Optional[str],
        engine: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run pipeline for a single theme.

        Args:
            mode: "full" or "generate"
            theme: Safety theme
            max_iterations: Number of iterations
            start_iteration: Starting iteration (for resume)
            refusal_target: Target refusal rate range
            num_categories: Categories per iteration
            num_prompts: Prompts per category
            quality_threshold: Minimum quality score
            mutations_per_source: Mutations per source prompt
            mutation_operators: List of mutation operator names
            seed_prompt_dir: Path to existing prompts
            output_dir: Output directory
            engine: Engine name for directory naming
            **kwargs: Additional parameters

        Returns:
            Results dictionary based on mode
        """
        logger.info(f"Running safety engine for theme: {theme}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Iterations: {start_iteration} to {max_iterations}")
        logger.info(f"Refusal target: {refusal_target}")

        # Resolve default seed_prompt_dir so the same path is used
        # for existing prompts AND incident context loading
        if not seed_prompt_dir:
            package_dir = Path(__file__).parent.parent
            project_root = package_dir.parent
            seed_prompt_dir = str(project_root / "data" / "seeds" / "safety")

        existing_prompts = self._load_existing_prompts(seed_prompt_dir)

        if mode == "generate":
            return self._run_generate_mode(
                theme=theme,
                max_iterations=max_iterations,
                start_iteration=start_iteration,
                refusal_target=refusal_target,
                num_categories=num_categories,
                num_prompts=num_prompts,
                quality_threshold=quality_threshold,
                mutations_per_source=mutations_per_source,
                mutation_operators=mutation_operators,
                existing_prompts=existing_prompts,
                seed_prompt_dir=seed_prompt_dir,
                output_dir=output_dir,
                engine=engine,
                **kwargs
            )
        else:  # mode == "full"
            return self._run_full_mode(
                theme=theme,
                max_iterations=max_iterations,
                start_iteration=start_iteration,
                refusal_target=refusal_target,
                num_categories=num_categories,
                num_prompts=num_prompts,
                quality_threshold=quality_threshold,
                mutations_per_source=mutations_per_source,
                mutation_operators=mutation_operators,
                existing_prompts=existing_prompts,
                seed_prompt_dir=seed_prompt_dir,
                output_dir=output_dir,
                engine=engine,
                **kwargs
            )

    def _run_generate_mode(
        self,
        theme: str,
        max_iterations: int,
        start_iteration: int,
        refusal_target: str,
        num_categories: int,
        num_prompts: int,
        quality_threshold: int,
        mutations_per_source: int,
        mutation_operators: Optional[List[str]],
        existing_prompts: List[Dict[str, Any]],
        seed_prompt_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        engine: str = "safety_novelty",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate safety prompts without evaluating a test model."""
        from ..safety.safety_core import generate_full_safety_prompts, refine_safety_categories
        from ..safety.safety_seeds import load_incident_context

        # Get models
        agent_model = self.models.agent_model
        eval_model = self.models.eval_model

        # Setup output directory
        package_dir = Path(__file__).parent.parent
        project_root = package_dir.parent

        if not output_dir:
            output_dir = str(project_root / "data" / "generated" / engine)
        os.makedirs(output_dir, exist_ok=True)

        # Load AIAAIC incident context
        incidents = []
        if seed_prompt_dir and os.path.isdir(seed_prompt_dir):
            incidents = load_incident_context(seed_prompt_dir)
            if incidents:
                logger.info(f"Loaded {len(incidents)} AIAAIC incidents as supplementary context")

        all_prompts = []
        history_text = ["Initial iteration"]

        # Restore history from prior iterations when resuming
        if start_iteration > 1:
            theme_slug = theme.replace(' ', '_')
            logger.info(f"Resuming from iteration {start_iteration}")
            for prev_iter in range(1, start_iteration):
                prev_prefix = os.path.join(output_dir, f"{theme_slug}_iter_{prev_iter}")
                summary_path = f"{prev_prefix}.iteration_summary.json"
                if os.path.isfile(summary_path):
                    with open(summary_path, "r") as fh:
                        prev_summary = json.load(fh)
                    if "summary" in prev_summary:
                        history_text.append(prev_summary["summary"])
                    else:
                        rate = prev_summary.get("avg_unsafe_rate", "N/A")
                        n = prev_summary.get("num_prompts", "?")
                        history_text.append(
                            f"Iteration {prev_iter}: {n} prompts, "
                            f"avg_unsafe_rate={rate}"
                        )
                    logger.info(f"✓ Loaded iteration {prev_iter} summary")

        # Generation loop
        for iteration in range(start_iteration, max_iterations + 1):
            logger.info(f"\nGeneration iteration {iteration}/{max_iterations}")

            theme_slug = theme.replace(' ', '_')
            outfile_prefix = os.path.join(output_dir, f"{theme_slug}_iter_{iteration}")

            prompts = generate_full_safety_prompts(
                theme=theme,
                agent_model=agent_model,
                history=history_text,
                iteration=iteration,
                outfile_prefix=outfile_prefix,
                acc_target=refusal_target,
                max_categories=num_categories,
                num_prompts_per_category=num_prompts,
                existing_prompts=existing_prompts,
                incidents=incidents,
                category_gen_func=refine_safety_categories,
                eval_model=eval_model,
                quality_threshold=quality_threshold,
                mutations_per_source=mutations_per_source,
                mutation_operators=mutation_operators,
            )

            all_prompts.extend(prompts)
            history_text.append(f"Iteration {iteration}: generated {len(prompts)} prompts.")

        # Save combined output
        combined_path = os.path.join(output_dir, "generated_safety_prompts.jsonl")
        with open(combined_path, "w") as fh:
            for p in all_prompts:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

        logger.info(f"✅ Generated {len(all_prompts)} prompts → {combined_path}")

        return {
            "all_prompts": all_prompts,
            "output_dir": output_dir,
            "num_prompts": len(all_prompts)
        }

    def _run_full_mode(
        self,
        theme: str,
        max_iterations: int,
        start_iteration: int,
        refusal_target: str,
        num_categories: int,
        num_prompts: int,
        quality_threshold: int,
        mutations_per_source: int,
        mutation_operators: Optional[List[str]],
        existing_prompts: List[Dict[str, Any]],
        seed_prompt_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        engine: str = "safety_novelty",
        **kwargs
    ) -> Dict[str, Any]:
        """Run full pipeline -- generate, evaluate, iterate."""
        from ..safety.safety_engine import run_safety_novelty_engine, save_safety_benchmark

        # Get models
        agent_model = self.models.agent_model
        steering_panel = self.models.steering_panel
        eval_model = self.models.eval_model

        logger.info(f"Agent model: {type(agent_model).__name__}")
        logger.info(f"Steering panel: {list(steering_panel.keys())}")
        logger.info(f"Eval model: {type(eval_model).__name__}")

        # Run the safety novelty engine
        results = run_safety_novelty_engine(
            agent_model=agent_model,
            test_models=steering_panel,
            eval_model=eval_model,
            theme=theme,
            max_iterations=max_iterations,
            refusal_target=refusal_target,
            num_categories=num_categories,
            num_prompts_per_category=num_prompts,
            quality_threshold=quality_threshold,
            seed_prompt_dir=seed_prompt_dir,
            existing_prompts=existing_prompts,
            output_dir=output_dir,
            engine=engine,
            mutations_per_source=mutations_per_source,
            mutation_operators=mutation_operators,
            start_iteration=start_iteration,
        )

        # Save as benchmark
        if results["all_prompts"]:
            benchmark_path = save_safety_benchmark(
                results["all_prompts"],
                filename=f"{theme.replace(' ', '_')}_benchmark",
            )
            logger.info(f"Benchmark saved to: {benchmark_path}")
            results["benchmark_path"] = benchmark_path

        logger.info(
            f"✅ Full pipeline complete — {len(results['all_prompts'])} "
            f"total prompts generated and evaluated"
        )

        return results

    def _run_multiple_themes(
        self,
        mode: str,
        themes: List[str],
        max_iterations: int,
        start_iteration: int,
        refusal_target: str,
        num_categories: int,
        num_prompts: int,
        quality_threshold: int,
        mutations_per_source: int,
        mutation_operators: Optional[List[str]],
        seed_prompt_dir: Optional[str],
        output_dir: Optional[str],
        engine: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run pipeline for multiple themes sequentially.

        Args:
            mode: "full" or "generate"
            themes: List of themes
            max_iterations: Number of iterations per theme
            start_iteration: Starting iteration
            refusal_target: Target refusal rate range
            num_categories: Categories per iteration
            num_prompts: Prompts per category
            quality_threshold: Minimum quality score
            mutations_per_source: Mutations per source prompt
            mutation_operators: List of mutation operator names
            seed_prompt_dir: Path to existing prompts
            output_dir: Output directory
            engine: Engine name
            **kwargs: Additional parameters

        Returns:
            Dictionary with results per theme
        """
        logger.info(f"Running safety engine for {len(themes)} themes: {themes}")

        all_results = {}

        for theme in themes:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing theme: {theme}")
            logger.info(f"{'='*70}\n")

            # Use theme-specific output directory
            theme_output_dir = output_dir
            if output_dir:
                theme_slug = theme.replace(' ', '_')
                theme_output_dir = os.path.join(output_dir, theme_slug)

            results = self._run_single_theme(
                mode=mode,
                theme=theme,
                max_iterations=max_iterations,
                start_iteration=start_iteration,
                refusal_target=refusal_target,
                num_categories=num_categories,
                num_prompts=num_prompts,
                quality_threshold=quality_threshold,
                mutations_per_source=mutations_per_source,
                mutation_operators=mutation_operators,
                seed_prompt_dir=seed_prompt_dir,
                output_dir=theme_output_dir,
                engine=engine,
                **kwargs
            )

            all_results[theme] = results

        logger.info(f"\n✅ Completed all {len(themes)} themes")

        return {
            "results_by_theme": all_results,
            "themes": themes,
            "num_themes": len(themes)
        }

    def _load_existing_prompts(
        self,
        seed_prompt_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load existing safety prompts from curated JSONL.

        Args:
            seed_prompt_dir: Path to directory containing existing_prompts.jsonl

        Returns:
            List of existing prompt dictionaries
        """
        # Get default path if not provided
        if not seed_prompt_dir:
            package_dir = Path(__file__).parent.parent
            project_root = package_dir.parent
            seed_prompt_dir = str(project_root / "data" / "seeds" / "safety")

        prompts_path = os.path.join(seed_prompt_dir, "existing_prompts.jsonl")

        if not os.path.isfile(prompts_path):
            logger.warning(f"Existing prompts file not found: {prompts_path}")
            return []

        prompts = []
        with open(prompts_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))

        logger.info(f"Loaded {len(prompts)} existing safety prompts from {prompts_path}")
        return prompts

    def _load_seed_topics(self, seed_index_path: str) -> List[str]:
        """Load themes from a JSON seed file.

        Args:
            seed_index_path: Path to JSON file with themes

        Returns:
            List of theme strings

        Raises:
            ValueError: If file not found or invalid format
        """
        if not os.path.isfile(seed_index_path):
            raise ValueError(f"Seed index file not found: {seed_index_path}")

        with open(seed_index_path, "r") as fh:
            data = json.load(fh)

        # Support multiple formats
        if isinstance(data, list):
            themes = data
        elif isinstance(data, dict):
            themes = data.get("themes", data.get("topics", []))
        else:
            raise ValueError(f"Unexpected seed file format: {type(data)}")

        if not themes:
            raise ValueError(f"No themes found in {seed_index_path}")

        logger.info(f"Loaded {len(themes)} themes from {seed_index_path}")
        return themes
