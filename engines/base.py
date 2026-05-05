"""Base engine class for all benchmark generation pipelines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """Abstract base class for benchmark generation engines.

    This class provides a common interface for all benchmark engines,
    handling configuration, model management, and pipeline execution.

    Subclasses must implement:
        - _get_parser(): Return ArgumentParser for CLI usage
        - run(): Execute the benchmark generation pipeline

    Attributes:
        config: BenchmarkConfig instance with all settings
        models: ModelManager for lazy model loading
    """

    def __init__(self, config: Optional['BenchmarkConfig'] = None):
        """Initialize engine with configuration.

        Args:
            config: BenchmarkConfig instance. If None, creates default config
                from environment variables.
        """
        if config is None:
            # Lazy import to avoid circular dependencies
            from ..config import BenchmarkConfig
            config = BenchmarkConfig.from_env()

        self.config = config
        self._models = None
        self._history = []

        # Set up logging
        self._setup_logging()

    @property
    def models(self):
        """Get ModelManager instance (lazy initialization)."""
        if self._models is None:
            from ..config import ModelManager
            self._models = ModelManager(self.config)
        return self._models

    def _setup_logging(self):
        """Configure logging based on config settings."""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add file handler if log_file specified
        if self.config.log_file:
            import os
            log_dir = os.path.dirname(self.config.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)

    @classmethod
    def from_cli(cls, args: Optional[Any] = None) -> 'BaseEngine':
        """Create engine from command-line arguments.

        Args:
            args: Parsed argparse Namespace. If None, parses sys.argv
                using the engine's parser.

        Returns:
            Configured engine instance

        Examples:
            >>> engine = NoveltyEngine.from_cli()  # Parses sys.argv
            >>> results = engine.run()

            >>> # Or with explicit args
            >>> parser = NoveltyEngine._get_parser()
            >>> args = parser.parse_args(['--theme', 'science'])
            >>> engine = NoveltyEngine.from_cli(args)
        """
        if args is None:
            parser = cls._get_parser()
            args = parser.parse_args()

        # Convert args to config
        from ..config import BenchmarkConfig
        config = BenchmarkConfig.from_args(args)

        return cls(config)

    @classmethod
    @abstractmethod
    def _get_parser(cls):
        """Return argparse.ArgumentParser for this engine.

        Each engine must provide its own parser with engine-specific
        arguments (built on top of the common parser).

        Returns:
            argparse.ArgumentParser instance
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the benchmark generation pipeline.

        Each engine implements its own run() method with appropriate
        parameters for its pipeline.

        Args:
            **kwargs: Engine-specific parameters

        Returns:
            Results dictionary or dict of dicts (for multi-theme)
        """
        pass

    def clear_cache(self):
        """Clear cached models to free memory."""
        if self._models:
            self._models.clear_cache()
            self._models = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
