"""Configuration dataclasses for benchmark engines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark generation engines.

    This dataclass holds all configuration settings for benchmark engines,
    including model configurations, engine parameters, and I/O settings.

    Attributes:
        # Model configurations
        agent_model: Model name/path for generation (LLM "red-teamer")
        agent_api_url: API endpoint URL for agent model
        agent_api_token: API token for agent model
        agent_api_version: API version for agent model
        agent_type: Model type ("openai", "huggingface", etc.)

        test_model: Model name/path to test
        test_model_type: Type of test model

        eval_model: Model name/path for evaluation (LLM judge)
        eval_api_url: API endpoint URL for eval model
        eval_api_token: API token for eval model
        eval_api_version: API version for eval model
        eval_type: Model type for evaluator

        # Theme configuration (mutually exclusive)
        theme: Single theme for generation
        themes: List of themes to run
        seed_index_path: Path to JSON file with seed_topics list

        # Engine parameters
        max_iterations: Number of generation-evaluation iterations
        start_iteration: Iteration number to start from (for resuming)
        engine: Engine name for file organization

        # Novelty-specific parameters
        acc_target: Target accuracy range for novelty (e.g., "0.1--0.4")
        use_ragas: Whether to use RAGAS for question generation
        embedding_model: Embedding model name for RAGAS

        # Safety-specific parameters
        refusal_target: Target refusal rate for safety (e.g., "0.7--0.9")
        num_categories: Number of categories per iteration
        num_prompts_per_category: Number of prompts per category
        quality_threshold: Minimum quality score (0-10)
        mutations_per_source: Number of mutations per source prompt
        mutation_operators: List of mutation operator names

        # I/O settings
        output_dir: Output directory for results
        cuda_device: CUDA device ID(s)

        # Logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """

    # Model configurations
    agent_model: str = "gpt-4.1-mini-aixamine"
    agent_api_url: str = ""
    agent_api_token: str = ""
    agent_api_version: str = "2024-12-01-preview"
    agent_type: str = "openai"

    test_model: str = "/export/cyb-llm-benchmarking/hallucination/models/google_gemma-2-2b-it"
    test_model_type: str = "huggingface"

    # Steering panel: comma-separated "name@url" entries for multi-model evaluation
    # e.g. "gemma-2-9b-it@http://localhost:8001/v1,Llama-3.1-8B@http://localhost:8002/v1"
    steering_models: Optional[str] = None
    # Steering panel (local): comma-separated "name:path" entries loaded into GPU memory
    # e.g. "gemma-2-9b-it:/path/to/google_gemma-2-9b-it,Llama-3.1-8B:/path/to/model"
    steering_models_local: Optional[str] = None

    eval_model: str = "gpt-oss"
    eval_api_url: str = "http://10.4.8.217:8000/v1"
    eval_api_token: str = ""
    eval_api_version: str = "2024-12-01-preview"
    eval_type: str = "openai"

    # Theme configuration
    theme: Optional[str] = None
    themes: Optional[List[str]] = None
    seed_index_path: Optional[str] = None

    # Engine parameters
    max_iterations: int = 5
    start_iteration: int = 1
    engine: str = "novelty"

    # Novelty-specific
    acc_target: str = "0.1--0.4"
    use_ragas: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Safety-specific
    refusal_target: str = "0.7--0.9"
    num_categories: int = 15
    num_prompts_per_category: int = 8
    quality_threshold: int = 6
    mutations_per_source: int = 3
    mutation_operators: Optional[List[str]] = None

    # I/O
    output_dir: Optional[str] = None
    cuda_device: str = "0"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "") -> 'BenchmarkConfig':
        """Create config from environment variables.

        Reads configuration from environment variables, with optional prefix.
        Falls back to defaults if environment variables are not set.

        Args:
            prefix: Optional prefix for environment variables (e.g., "NOVELTY_")

        Returns:
            BenchmarkConfig instance

        Examples:
            >>> config = BenchmarkConfig.from_env()
            >>> config = BenchmarkConfig.from_env(prefix="NOVELTY_")
        """
        def getenv(key: str, default: str = "") -> str:
            """Get environment variable with optional prefix."""
            return os.getenv(f"{prefix}{key}", default)

        return cls(
            # Agent model
            agent_model=getenv("JUDGE_EXTERNAL_MODEL") or getenv("AGENT_MODEL", "gpt-4.1-mini-aixamine"),
            agent_api_url=getenv("JUDGE_EXTERNAL_ENDPOINT") or getenv("AGENT_API_URL", ""),
            agent_api_token=getenv("JUDGE_EXTERNAL_TOKEN") or getenv("AGENT_API_TOKEN", ""),
            agent_api_version=getenv("JUDGE_EXTERNAL_VERSION", "2024-12-01-preview"),

            # Test model
            test_model=getenv("TEST_MODEL", ""),

            # Eval model
            eval_model=getenv("EVAL_MODEL", "gpt-oss"),
            eval_api_url=getenv("EVAL_ENDPOINT", "http://10.4.8.217:8000/v1"),
            eval_api_token=getenv("EVAL_TOKEN", ""),

            # CUDA
            cuda_device=getenv("CUDA_VISIBLE_DEVICES", "0"),
        )

    @classmethod
    def from_args(cls, args: Any) -> 'BenchmarkConfig':
        """Create config from argparse Namespace.

        Maps command-line arguments to config attributes.

        Args:
            args: Parsed argparse.Namespace from ArgumentParser

        Returns:
            BenchmarkConfig instance

        Examples:
            >>> parser = argparse.ArgumentParser()
            >>> # ... add arguments ...
            >>> args = parser.parse_args()
            >>> config = BenchmarkConfig.from_args(args)
        """
        # Start with defaults from environment
        config = cls.from_env()

        # Override with explicit args
        for key in config.__dataclass_fields__:
            if hasattr(args, key):
                value = getattr(args, key)
                if value is not None:
                    setattr(config, key, value)

        # Special handling for themes
        if hasattr(args, 'themes') and args.themes:
            # Parse comma-separated themes
            if isinstance(args.themes, str):
                config.themes = [t.strip() for t in args.themes.split(',')]
            else:
                config.themes = args.themes
            # Clear single theme if themes list provided
            config.theme = None

        # Special handling for seed_index_path
        if hasattr(args, 'seed_index_path') and args.seed_index_path:
            config.seed_index_path = args.seed_index_path
            config.theme = None
            config.themes = None

        # Special handling for no_ragas flag
        if hasattr(args, 'no_ragas'):
            config.use_ragas = not args.no_ragas

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config

        Examples:
            >>> config = BenchmarkConfig()
            >>> config_dict = config.to_dict()
        """
        return asdict(self)

    def copy(self) -> 'BenchmarkConfig':
        """Create a deep copy of this config.

        Returns:
            New BenchmarkConfig instance with same values

        Examples:
            >>> config1 = BenchmarkConfig()
            >>> config2 = config1.copy()
            >>> config2.max_iterations = 10  # Doesn't affect config1
        """
        from copy import deepcopy
        return deepcopy(self)

    def update(self, overrides: Dict[str, Any]):
        """Update config with dictionary of overrides.

        Args:
            overrides: Dictionary of attribute names and values

        Examples:
            >>> config = BenchmarkConfig()
            >>> config.update({"max_iterations": 10, "theme": "science"})
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"BenchmarkConfig has no attribute '{key}'")

    def __repr__(self) -> str:
        """String representation showing key settings."""
        return (
            f"BenchmarkConfig("
            f"agent={self.agent_model}, "
            f"test={self.test_model or 'None'}, "
            f"eval={self.eval_model}, "
            f"theme={self.theme or self.themes or self.seed_index_path}, "
            f"iters={self.max_iterations})"
        )
