"""Novelty Engine - High-level API for factuality benchmark generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseEngine

logger = logging.getLogger(__name__)


class NoveltyEngine(BaseEngine):
    """Novelty engine for generating knowledge-intensive QA benchmarks.

    This engine generates challenging factual question-answer pairs through
    an iterative, Wikipedia-grounded process. It supports:
    - Single theme generation
    - Multiple themes (from list or seed file)
    - Resuming from specific iterations
    - RAGAS integration for quality control
    - Full parameter customization

    Examples:
        # Simple usage - single theme
        >>> from sspbench.engines import NoveltyEngine
        >>> engine = NoveltyEngine()
        >>> results = engine.run(theme="science", max_iterations=3)

        # Multiple themes - direct list
        >>> results = engine.run(themes=["science", "history", "geography"])

        # Multiple themes - from seed file
        >>> results = engine.run(seed_index_path="data/seeds/hallucination/seed_topics.json")

        # Resume from iteration 3
        >>> results = engine.run(
        ...     theme="science",
        ...     max_iterations=10,
        ...     start_iteration=3
        ... )

        # With explicit model configuration
        >>> engine = NoveltyEngine(
        ...     agent_model="gpt-4",
        ...     test_model="gemma-2-2b",
        ...     eval_model="gpt-oss"
        ... )
        >>> results = engine.run(theme="history")

        # From CLI
        >>> engine = NoveltyEngine.from_cli()
        >>> results = engine.run()
    """

    def __init__(
        self,
        config: Optional['BenchmarkConfig'] = None,
        agent_model: Optional[str] = None,
        test_model: Optional[str] = None,
        eval_model: Optional[str] = None,
    ):
        """Initialize novelty engine.

        Args:
            config: Full configuration object (overrides other args)
            agent_model: Model name/path for question generation
            test_model: Model name/path to test
            eval_model: Model name/path for evaluation
        """
        if config is None:
            from ..config import BenchmarkConfig
            config = BenchmarkConfig.from_env()

            # Override with explicit model args
            if agent_model:
                config.agent_model = agent_model
            if test_model:
                config.test_model = test_model
            if eval_model:
                config.eval_model = eval_model

        super().__init__(config)
        self._embedding_model = None
        self._setup_ragas()

    def _setup_ragas(self):
        """Setup RAGAS embedding model if enabled."""
        if self.config.use_ragas:
            try:
                from ..novelty.ragas_utils import (
                    SentenceTransformerEmbeddings,
                    is_ragas_available
                )
                if is_ragas_available():
                    self._embedding_model = SentenceTransformerEmbeddings(
                        self.config.embedding_model,
                        device=self.config.embedding_device,
                    )
                    logger.info(f"✓ RAGAS enabled: {self.config.embedding_model}")
                else:
                    logger.warning("RAGAS not available. Install with: pip install ragas langchain-core")
                    self.config.use_ragas = False
            except Exception as e:
                logger.warning(f"Failed to load RAGAS: {e}")
                self.config.use_ragas = False
        else:
            logger.info("RAGAS disabled")

    @property
    def embedding_model(self):
        """Get RAGAS embedding model if available."""
        return self._embedding_model

    @classmethod
    def _get_parser(cls):
        """Get argument parser for novelty engine."""
        from ..cli import get_novelty_parser
        return get_novelty_parser()

    def run(
        self,
        theme: Optional[str] = None,
        themes: Optional[List[str]] = None,
        seed_index_path: Optional[str] = None,
        max_iterations: Optional[int] = None,
        start_iteration: Optional[int] = None,
        acc_target: Optional[str] = None,
        use_ragas: Optional[bool] = None,
        output_dir: Optional[str] = None,
        engine: Optional[str] = None,
        **kwargs
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Run the novelty engine pipeline.

        Args:
            theme: Single theme (mutually exclusive with themes/seed_index_path)
            themes: List of themes to run (mutually exclusive with theme/seed_index_path)
            seed_index_path: Path to JSON with seed_topics list
            max_iterations: Number of iterations per theme
            start_iteration: Iteration to start from (for resuming, default: 1)
            acc_target: Target accuracy range (e.g., "0.1--0.4")
            use_ragas: Whether to use RAGAS for question generation
            output_dir: Custom output directory
            engine: Engine name for file organization
            **kwargs: Additional engine-specific parameters

        Returns:
            If single theme: Dict with 'history', 'theme', 'output_dir' keys
            If multiple themes: Dict[theme_name, results_dict]

        Raises:
            ValueError: If multiple theme selection methods provided

        Examples:
            >>> engine = NoveltyEngine()
            >>> # Single theme
            >>> results = engine.run(theme="science")
            >>> # Multiple themes
            >>> results = engine.run(themes=["science", "history"])
            >>> # Resume from iteration 3
            >>> results = engine.run(theme="science", start_iteration=3, max_iterations=10)
        """
        explicitly_provided = [theme is not None, themes is not None]
        if sum(explicitly_provided) > 1:
            raise ValueError(
                "Provide only one of: theme or themes"
            )
        if (theme is not None or themes is not None) and seed_index_path is not None:
            raise ValueError(
                "Provide only one of: theme, themes, or seed_index_path"
            )

        # Use config defaults if not provided
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        if start_iteration is None:
            start_iteration = self.config.start_iteration
        if acc_target is None:
            acc_target = self.config.acc_target
        if use_ragas is None:
            use_ragas = self.config.use_ragas
        if output_dir is None:
            output_dir = self.config.output_dir
        if engine is None:
            engine = self.config.engine
        if seed_index_path is None and theme is None and themes is None:
            if self.config.seed_index_path:
                seed_index_path = self.config.seed_index_path
            elif self.config.themes:
                themes = self.config.themes
            elif self.config.theme:
                theme = self.config.theme
            else:
                from pathlib import Path
                default_path = Path(__file__).parent.parent.parent / 'data' / 'seeds' / 'hallucination' / 'seed_topics.json'
                if default_path.exists():
                    seed_index_path = str(default_path)
                    logger.info(f"No theme specified, using default seed file: {seed_index_path}")

        # Determine mode: single theme, multiple themes, or seed file
        if themes or seed_index_path:
            # Multiple themes mode
            theme_list = themes or self._load_seed_topics(seed_index_path)
            return self._run_multiple_themes(
                themes=theme_list,
                max_iterations=max_iterations,
                start_iteration=start_iteration,
                acc_target=acc_target,
                use_ragas=use_ragas,
                output_dir=output_dir,
                engine=engine,
                **kwargs
            )
        else:
            # Single theme mode
            if theme is None:
                theme = self.config.theme
            return self._run_single_theme(
                theme=theme,
                max_iterations=max_iterations,
                start_iteration=start_iteration,
                acc_target=acc_target,
                use_ragas=use_ragas,
                output_dir=output_dir,
                engine=engine,
                **kwargs
            )

    def _run_single_theme(
        self,
        theme: str,
        max_iterations: int,
        start_iteration: int,
        acc_target: str,
        use_ragas: bool,
        output_dir: Optional[str],
        engine: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute novelty engine for a single theme.

        Args:
            theme: Theme for generation
            max_iterations: Number of iterations
            start_iteration: Iteration to start from
            acc_target: Target accuracy range
            use_ragas: Whether to use RAGAS
            output_dir: Output directory
            engine: Engine name
            **kwargs: Additional parameters

        Returns:
            Results dictionary with 'history', 'theme', 'output_dir' keys
        """
        from ..novelty.main import run_novelty_engine

        logger.info("="*60)
        logger.info("NOVELTY ENGINE")
        logger.info("="*60)
        logger.info(f"Theme: {theme}")
        logger.info(f"Iterations: {start_iteration}-{max_iterations}")
        logger.info(f"Target accuracy: {acc_target}")
        logger.info(f"RAGAS: {'enabled' if use_ragas else 'disabled'}")
        if output_dir:
            logger.info(f"Output: {output_dir}")
        logger.info("="*60)

        # Get models (lazily created on first access)
        agent_model = self.models.agent_model
        steering_panel = self.models.steering_panel
        eval_model = self.models.eval_model

        # Run core engine function
        history = run_novelty_engine(
            agent_model=agent_model,
            test_models=steering_panel,
            eval_model=eval_model,
            theme=theme,
            max_iterations=max_iterations,
            start_iteration=start_iteration,
            acc_target=acc_target,
            engine=engine,
            use_ragas=use_ragas,
            embedding_model=self.embedding_model,
            output_dir=output_dir,
        )

        self._history.append(history)

        logger.info("="*60)
        logger.info(f"✓ COMPLETED - {len(history)} iterations")
        logger.info("="*60)

        return {
            'history': history,
            'theme': theme,
            'output_dir': output_dir,
            'max_iterations': max_iterations,
            'start_iteration': start_iteration,
            'acc_target': acc_target,
        }

    def _run_multiple_themes(
        self,
        themes: List[str],
        max_iterations: int,
        start_iteration: int,
        acc_target: str,
        use_ragas: bool,
        output_dir: Optional[str],
        engine: str,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Execute novelty engine across multiple themes.

        Args:
            themes: List of themes
            max_iterations: Number of iterations per theme
            start_iteration: Iteration to start from
            acc_target: Target accuracy range
            use_ragas: Whether to use RAGAS
            output_dir: Output directory
            engine: Engine name
            **kwargs: Additional parameters

        Returns:
            Dictionary keyed by theme with results
        """
        logger.info("="*60)
        logger.info(f"NOVELTY ENGINE - MULTI-THEME MODE")
        logger.info("="*60)
        logger.info(f"Themes: {len(themes)}")
        for i, t in enumerate(themes, 1):
            logger.info(f"  {i}. {t}")
        logger.info(f"Iterations per theme: {start_iteration}-{max_iterations}")
        logger.info(f"Target accuracy: {acc_target}")
        logger.info("="*60)

        results = {}
        for i, theme in enumerate(themes, 1):
            logger.info(f"\n[{i}/{len(themes)}] Processing theme: {theme}")
            logger.info("-" * 60)

            try:
                results[theme] = self._run_single_theme(
                    theme=theme,
                    max_iterations=max_iterations,
                    start_iteration=start_iteration,
                    acc_target=acc_target,
                    use_ragas=use_ragas,
                    output_dir=output_dir,
                    engine=engine,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"✗ Theme '{theme}' failed: {e}")
                results[theme] = {'error': str(e)}

        logger.info("\n" + "="*60)
        logger.info(f"✓ MULTI-THEME COMPLETED")
        logger.info("="*60)
        successful = sum(1 for r in results.values() if 'error' not in r)
        logger.info(f"Successful: {successful}/{len(themes)} themes")

        return results

    def _load_seed_topics(self, path: str) -> List[str]:
        """Load seed topics from JSON manifest.

        Args:
            path: Path to JSON file with seed_topics list

        Returns:
            List of theme names

        Raises:
            ValueError: If seed_topics not found in file
            FileNotFoundError: If file doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Seed index file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        seed_topics = data.get("seed_topics")
        if not seed_topics:
            raise ValueError(f"No seed_topics found in {path}")

        logger.info(f"Loaded {len(seed_topics)} themes from {path}")
        return list(seed_topics)

    def __repr__(self) -> str:
        return f"NoveltyEngine(config={self.config}, ragas={self.config.use_ragas})"
