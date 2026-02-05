"""
LLM utilities for the Novelty Engine.
Functions for interacting with language models.
"""

import sys
import os

plugins_path = os.path.join(os.path.dirname(__file__), '..', '..', 'plugins')
if plugins_path not in sys.path:
    sys.path.insert(0, plugins_path)

from models.llm_factory import LLMFactory
from llm_config import LLMConfig
from conversation import Conversation
from vllm import SamplingParams

from .config import DEFAULT_SAMPLING_PARAMS

# Global cache for loaded models to avoid reloading
_MODEL_CACHE = {}


def gen_from_prompt(model, prompt, temperature=0., max_tokens=20, top_p=1.0, system_prompt=None):
    """
    Generate response from a language model.

    Args:
        model: The language model instance
        prompt: Input prompt (string or list)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt for the conversation

    Returns:
        Generated response(s)
    """
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    is_single = isinstance(prompt, str)
    prompts = [prompt] if is_single else prompt

    convs = []
    for p in prompts:
        conv = Conversation(system_prompt=system_prompt)
        conv.add_message("user", p)
        convs.append(conv)

    responses = model.generate(convs, sampling_params)
    # print("Conversation", convs[0].to_list(), "Response:", responses[0])
    return responses[0] if is_single else responses


def create_model_from_config(config_dict):
    """
    Create a model instance from configuration dictionary.
    Uses caching to avoid reloading the same model multiple times.

    Args:
        config_dict: Dictionary containing model configuration

    Returns:
        Model instance
        
    Raises:
        RuntimeError: If model creation fails
    """
    # Create a hashable key from the config
    config_key = tuple(sorted(config_dict.items()))

    # Check if model is already cached
    if config_key in _MODEL_CACHE:
        print(f"Using cached model for config: {config_dict.get('model', 'Unknown')}")
        return _MODEL_CACHE[config_key]

    # Model not cached, create it
    print(f"Loading new model: {config_dict.get('model', 'Unknown')}")
    config = LLMConfig(**config_dict)
    model = LLMFactory.from_config(config)
    
    if model is None:
        model_name = config_dict.get('model', 'Unknown')
        model_type = config_dict.get('type', 'Unknown')
        raise RuntimeError(f"Failed to create {model_type} model '{model_name}'. Check model path/credentials and system resources.")
    
    # Cache the model
    _MODEL_CACHE[config_key] = model

    return model


def clear_model_cache():
    """
    Clear the model cache. Useful for freeing memory or forcing reloads.
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    print("Model cache cleared")


def get_default_sampling_params():
    """Get default sampling parameters."""
    return SamplingParams(**DEFAULT_SAMPLING_PARAMS)