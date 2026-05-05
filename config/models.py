"""Model lifecycle management with lazy loading and caching."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model creation, caching, and lifecycle.

    Models are created lazily on first access and cached for reuse.
    This avoids loading expensive models unnecessarily.

    Attributes:
        config: BenchmarkConfig instance with model settings

    Examples:
        >>> from sspbench.config import BenchmarkConfig, ModelManager
        >>> config = BenchmarkConfig()
        >>> manager = ModelManager(config)
        >>> agent = manager.agent_model  # Loads on first access
        >>> agent_again = manager.agent_model  # Returns cached instance
    """

    def __init__(self, config: 'BenchmarkConfig'):
        """Initialize model manager with configuration.

        Args:
            config: BenchmarkConfig instance
        """
        self.config = config
        self._agent_model = None
        self._test_model = None
        self._eval_model = None
        self._steering_panel = None

        # Note: CUDA_VISIBLE_DEVICES must be set BEFORE torch import (in the script)
        # Logging current device configuration
        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        logger.info(f"CUDA device: {cuda_device} (set before torch import)")

    @property
    def agent_model(self) -> Any:
        """Get agent model (creates on first access).

        Returns:
            Model instance for agent/generation

        Examples:
            >>> manager = ModelManager(config)
            >>> agent = manager.agent_model
            >>> response = agent.generate("Hello")
        """
        if self._agent_model is None:
            logger.info("Creating agent model...")
            self._agent_model = self._create_agent_model()
            logger.info("✓ Agent model ready")
        return self._agent_model

    @property
    def test_model(self) -> Any:
        """Get test model (creates on first access).

        Returns:
            Model instance for testing

        Examples:
            >>> manager = ModelManager(config)
            >>> test = manager.test_model
            >>> response = test.generate("Test prompt")
        """
        if self._test_model is None:
            logger.info("Creating test model...")
            self._test_model = self._create_test_model()
            logger.info("✓ Test model ready")
        return self._test_model

    @property
    def eval_model(self) -> Any:
        """Get eval model (creates on first access).

        Returns:
            Model instance for evaluation/judging

        Examples:
            >>> manager = ModelManager(config)
            >>> evaluator = manager.eval_model
            >>> judgment = evaluator.generate("Is this safe?")
        """
        if self._eval_model is None:
            logger.info("Creating eval model...")
            self._eval_model = self._create_eval_model()
            logger.info("✓ Eval model ready")
        return self._eval_model

    @property
    def steering_panel(self) -> dict:
        """Get steering panel models as {name: model} dict (creates on first access).

        If --steering-models is set, loads those. Otherwise falls back to
        the single test_model under the key derived from its path.
        """
        if self._steering_panel is None:
            logger.info("Creating steering panel...")
            self._steering_panel = self._create_steering_panel()
            names = list(self._steering_panel.keys())
            logger.info(f"Steering panel ready: {names}")
        return self._steering_panel

    def _create_steering_panel(self) -> dict:
        """Create steering panel from config.

        If --steering-models is set, parses "name@url" entries and creates
        OpenAI API clients. If --steering-models-local is set, parses
        "name:path" entries and loads HuggingFace models into memory.
        Otherwise falls back to the single test_model.
        """
        from ..utils.llm_utils import create_model_from_config

        if self.config.steering_models and self.config.steering_models_local:
            raise ValueError(
                "Provide only one of --steering-models (API) or "
                "--steering-models-local (in-memory), not both."
            )

        if self.config.steering_models:
            panel = {}
            for entry in self.config.steering_models.split(","):
                entry = entry.strip()
                if "@" in entry:
                    name, url = entry.split("@", 1)
                else:
                    name, url = entry, entry
                config_dict = {
                    "type": "openai",
                    "model": name.strip(),
                    "api_url": url.strip(),
                    "api_token": "dummy",
                }
                try:
                    model = create_model_from_config(config_dict)
                    panel[name.strip()] = model
                    logger.info(f"  Connected to steering model: {name.strip()} @ {url.strip()}")
                except Exception as e:
                    logger.error(f"  Failed to connect to steering model {name}: {e}")
                    raise
            return panel

        if self.config.steering_models_local:
            panel = {}
            for entry in self.config.steering_models_local.split(","):
                entry = entry.strip()
                parts = entry.split(":")
                if len(parts) >= 3:
                    name, path, gpus = parts[0], parts[1], parts[2]
                elif len(parts) == 2:
                    name, path = parts
                    gpus = None
                else:
                    from pathlib import Path as _P
                    name, path, gpus = _P(entry).name, entry, None
                config_dict = {
                    "type": "huggingface",
                    "model": path.strip(),
                }
                if gpus is not None:
                    cuda_devices = gpus.strip().replace("+", ",")
                    config_dict["cuda_devices"] = cuda_devices
                    config_dict["tensor_parallel_size"] = len(cuda_devices.split(","))
                try:
                    model = create_model_from_config(config_dict)
                    panel[name.strip()] = model
                    gpu_info = f" on GPU(s) {gpus.strip()}" if gpus else ""
                    logger.info(f"  Loaded steering model: {name.strip()}{gpu_info}")
                except Exception as e:
                    logger.error(f"  Failed to load steering model {name}: {e}")
                    raise
            return panel

        # Fallback: single test_model
        if self.config.test_model:
            config_dict = {
                "type": self.config.test_model_type,
                "model": self.config.test_model,
            }
            model = create_model_from_config(config_dict)
            from pathlib import Path
            name = Path(self.config.test_model).name
            return {name: model}

        return {}

    def _create_agent_model(self) -> Any:
        """Create agent model from config.

        Returns:
            Model instance
        """
        from ..utils.llm_utils import create_model_from_config

        config_dict = {
            "type": self.config.agent_type,
            "model": self.config.agent_model,
            "api_url": self.config.agent_api_url,
            "api_token": self.config.agent_api_token,
            "api_version": self.config.agent_api_version,
        }

        try:
            model = create_model_from_config(config_dict)
            # Model created successfully (testing is done by the pipeline itself)
            return model
        except Exception as e:
            logger.error(f"Failed to create agent model: {e}")
            raise

    def _create_test_model(self) -> Any:
        """Create test model from config.

        Returns:
            Model instance
        """
        from ..utils.llm_utils import create_model_from_config

        if not self.config.test_model:
            logger.warning("No test model configured")
            return None

        config_dict = {
            "type": self.config.test_model_type,
            "model": self.config.test_model,
        }

        try:
            model = create_model_from_config(config_dict)
            # Model created successfully (testing is done by the pipeline itself)
            return model
        except Exception as e:
            logger.error(f"Failed to create test model: {e}")
            raise

    def _create_eval_model(self) -> Any:
        """Create eval model from config.

        Returns:
            Model instance
        """
        from ..utils.llm_utils import create_model_from_config

        config_dict = {
            "type": self.config.eval_type,
            "model": self.config.eval_model,
            "api_url": self.config.eval_api_url,
            "api_token": self.config.eval_api_token,
            "api_version": self.config.eval_api_version,
        }

        try:
            model = create_model_from_config(config_dict)
            # Model created successfully (testing is done by the pipeline itself)
            return model
        except Exception as e:
            logger.error(f"Failed to create eval model: {e}")
            raise

    def clear_cache(self):
        """Clear all cached models.

        This frees memory by clearing model references.
        Next access will reload models.

        Examples:
            >>> manager = ModelManager(config)
            >>> agent = manager.agent_model  # Loads model
            >>> manager.clear_cache()  # Clears reference
            >>> agent = manager.agent_model  # Reloads model
        """
        from ..utils.llm_utils import clear_model_cache

        clear_model_cache()
        self._agent_model = None
        self._test_model = None
        self._eval_model = None
        self._steering_panel = None
        logger.info("Model cache cleared")

    def __repr__(self) -> str:
        """String representation showing loaded models."""
        loaded = []
        if self._agent_model is not None:
            loaded.append("agent")
        if self._test_model is not None:
            loaded.append("test")
        if self._eval_model is not None:
            loaded.append("eval")

        loaded_str = ", ".join(loaded) if loaded else "none"
        return f"ModelManager(loaded=[{loaded_str}])"
