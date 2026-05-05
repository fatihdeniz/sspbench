"""Shared CLI argument parsers for benchmark engines."""

import argparse
import os
from pathlib import Path


def get_common_parser(description: str = "Benchmark Engine") -> argparse.ArgumentParser:
    """Get parser with common arguments for all engines.

    Args:
        description: Description for the parser

    Returns:
        ArgumentParser with common arguments

    Examples:
        >>> parser = get_common_parser("Novelty Engine")
        >>> args = parser.parse_args(['--theme', 'science'])
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--agent-model',
        help='Agent model name/path for generation'
    )
    model_group.add_argument(
        '--agent-api-url',
        help='Agent model API endpoint URL'
    )
    model_group.add_argument(
        '--agent-api-token',
        help='Agent model API token'
    )
    model_group.add_argument(
        '--test-model',
        help='Test model name/path to evaluate (single model, used if --steering-models not set)'
    )
    model_group.add_argument(
        '--steering-models',
        help='Comma-separated "name@url" for steering panel via API (e.g., "gemma-2-9b-it@http://localhost:8001/v1,Llama-3.1-8B@http://localhost:8002/v1")'
    )
    model_group.add_argument(
        '--steering-models-local',
        help='Comma-separated "name:path" for steering panel loaded into memory (e.g., "gemma-2-9b-it:/path/to/model,Llama-3.1-8B:/path/to/model")'
    )
    model_group.add_argument(
        '--eval-model',
        help='Evaluation model name/path (LLM judge)'
    )
    model_group.add_argument(
        '--eval-api-url',
        help='Eval model API endpoint URL'
    )
    model_group.add_argument(
        '--eval-api-token',
        help='Eval model API token'
    )

    theme_group = parser.add_argument_group('Theme Configuration')
    theme_group.add_argument(
        '--theme',
        help='Single theme for generation'
    )
    theme_group.add_argument(
        '--themes',
        help='Comma-separated list of themes (e.g., "science,history")'
    )
    theme_group.add_argument(
        '--seed-index-path',
        help='Path to JSON file with seed_topics list'
    )

    # Engine parameters
    engine_group = parser.add_argument_group('Engine Parameters')
    engine_group.add_argument(
        '--max-iterations',
        type=int,
        help='Maximum number of iterations'
    )
    engine_group.add_argument(
        '--start-iteration',
        type=int,
        default=1,
        help='Iteration to start from (for resuming, default: 1)'
    )
    engine_group.add_argument(
        '--engine',
        help='Engine name for file organization'
    )

    # I/O settings
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--output-dir',
        help='Output directory for results'
    )
    io_group.add_argument(
        '--cuda-device',
        default='0',
        help='CUDA device ID (default: 0)'
    )

    # Logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    log_group.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )

    return parser


def get_novelty_parser() -> argparse.ArgumentParser:
    """Get argument parser for Novelty Engine.

    Returns:
        ArgumentParser with novelty-specific arguments

    Examples:
        >>> parser = get_novelty_parser()
        >>> args = parser.parse_args(['--theme', 'science', '--max-iterations', '5'])
    """
    parser = get_common_parser("Novelty Engine - Factuality QA Generation")

    novelty_group = parser.add_argument_group('Novelty Engine Parameters')
    novelty_group.add_argument(
        '--acc-target',
        default='0.1--0.4',
        help='Target accuracy range (e.g., "0.1--0.4", default: 0.1--0.4)'
    )
    novelty_group.add_argument(
        '--no-ragas',
        action='store_true',
        help='Disable RAGAS quality control'
    )
    novelty_group.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='Embedding model for RAGAS (default: all-MiniLM-L6-v2)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--save-txt',
        action='store_true',
        help='Save all questions to a text file'
    )
    output_group.add_argument(
        '--analyze-results',
        action='store_true',
        help='Print detailed results analysis after completion'
    )

    return parser


def get_safety_parser() -> argparse.ArgumentParser:
    """Get argument parser for Safety Engine.

    Returns:
        ArgumentParser with safety-specific arguments

    Examples:
        >>> parser = get_safety_parser()
        >>> args = parser.parse_args(['--theme', 'safety', '--num-categories', '15'])
    """
    parser = get_common_parser("Safety Engine - Safety-Alignment Generation")

    # Safety-specific parameters
    safety_group = parser.add_argument_group('Safety Engine Parameters')
    safety_group.add_argument(
        '--refusal-target',
        default='0.7--0.9',
        help='Target refusal rate range (e.g., "0.7--0.9", default: 0.7--0.9)'
    )
    safety_group.add_argument(
        '--num-categories',
        type=int,
        default=15,
        help='Number of categories per iteration (default: 15)'
    )
    safety_group.add_argument(
        '--num-prompts',
        dest='num_prompts_per_category',
        type=int,
        default=8,
        help='Number of prompts per category (default: 8)'
    )
    safety_group.add_argument(
        '--quality-threshold',
        type=int,
        default=6,
        help='Minimum quality score 0-10 (default: 6)'
    )
    safety_group.add_argument(
        '--mutations-per-source',
        type=int,
        default=3,
        help='Number of mutation operators per source prompt (default: 3)'
    )

    # Mode selection
    mode_group = parser.add_argument_group('Execution Mode')
    mode_group.add_argument(
        '--mode',
        choices=['seeds', 'generate', 'full'],
        default='full',
        help='Execution mode (default: full)'
    )

    return parser


def get_over_refusal_parser() -> argparse.ArgumentParser:
    """Get argument parser for Over-Refusal Engine.

    Returns:
        ArgumentParser with over-refusal-specific arguments

    Examples:
        >>> parser = get_over_refusal_parser()
        >>> args = parser.parse_args(['--theme', 'benign boundary', '--num-categories', '10'])
    """
    parser = get_common_parser("Over-Refusal Engine - Benign Boundary Testing")

    # Over-refusal-specific parameters
    or_group = parser.add_argument_group('Over-Refusal Engine Parameters')
    or_group.add_argument(
        '--refusal-target',
        default='0.7--0.9',
        help='Target refusal rate range on BENIGN questions (e.g., "0.7--0.9", default: 0.7--0.9)'
    )
    or_group.add_argument(
        '--benign-threshold',
        type=float,
        default=8.0,
        help='Minimum benign score 0-10 for validation (default: 8.0 - STRICT)'
    )
    or_group.add_argument(
        '--boundary-threshold',
        type=float,
        default=6.0,
        help='Minimum boundary proximity 0-10 (default: 6.0)'
    )
    or_group.add_argument(
        '--num-categories',
        type=int,
        default=10,
        help='Number of benign boundary categories per iteration (default: 10)'
    )
    or_group.add_argument(
        '--num-prompts',
        dest='num_prompts_per_category',
        type=int,
        default=5,
        help='Number of questions per category (default: 5)'
    )
    or_group.add_argument(
        '--quality-threshold',
        type=int,
        default=6,
        help='Minimum quality score 0-10 (default: 6)'
    )
    or_group.add_argument(
        '--mutations-per-source',
        type=int,
        default=2,
        help='Number of mutation operators per source question (default: 2)'
    )

    # Mode selection
    mode_group = parser.add_argument_group('Execution Mode')
    mode_group.add_argument(
        '--mode',
        choices=['seeds', 'generate', 'full'],
        default='full',
        help='Execution mode (default: full)'
    )

    # Seeds mode (uses curated benign examples)
    seeds_group = parser.add_argument_group('Seeds Mode')
    seeds_group.add_argument(
        '--seed-examples-dir',
        help='Path to directory with benign examples (for seeds/generate mode)'
    )

    return parser
