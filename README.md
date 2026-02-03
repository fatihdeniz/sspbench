# SSP Bench

Safety, Security, and Privacy Dynamic Benchmark Generation Framework

## Overview

SSP Bench is a comprehensive framework for generating and evaluating dynamic benchmarks focused on the safety, security, and privacy aspects of large language models (LLMs). The framework provides tools to create adversarial examples, assess model vulnerabilities, and measure privacy risks.

## Features

- **Dynamic Benchmark Generation**: Generate custom benchmarks for safety, security, and privacy evaluation
- **Modular Evaluators**: Evaluate models using specialized evaluators for each domain
- **Comprehensive Metrics**: Compute relevant metrics for safety, security, and privacy
- **AutoBencher Engine**: Novelty engine that generates new questions using Wikipedia as privileged information
- **Variation Engine**: Apply semantic-preserving transformations (typos, contextualization, grammar)
- **Extensible Architecture**: Easy to add new generators, evaluators, and metrics

## Installation

```bash
pip install sspbench
```

Or from source:

```bash
git clone https://github.com/yourusername/sspbench.git
cd sspbench
pip install -e .
```

## Quick Start

```python
from sspbench.generators import SafetyGenerator
from sspbench.evaluators import SafetyEvaluator

# Generate safety benchmarks
generator = SafetyGenerator()
samples = generator.generate(num_samples=10)

# Evaluate a model
evaluator = SafetyEvaluator(model=your_model)
results = evaluator.evaluate(samples)
print(results)
```

## AutoBencher Engine

The AutoBencher engine implements a dynamic benchmark generation system:

```python
from autobencher import run_autobencher_with_variations

# Run dynamic benchmark generation with variations
history = run_autobencher_with_variations(
    theme="science and technology",
    max_iterations=3,
    use_variations=True
)
```

Features:

- **Novelty Generation**: Uses Wikipedia to generate new knowledge-intensive questions
- **Iterative Refinement**: Adapts categories based on model performance
- **Variation Application**: Adds typos, contextualization, and other transformations
- **Automatic Evaluation**: Tests models and provides detailed summaries

## Project Structure

- `sspbench/generators/`: Benchmark generation modules
- `sspbench/evaluators/`: Model evaluation modules
- `sspbench/metrics/`: Metric computation functions
- `sspbench/utils/`: Utility functions
- `benchmarks/`: Pre-generated benchmark datasets
- `tools/`: Integration with existing benchmark tools
- `autobencher_v2.ipynb`: Implementation notebook for AutoBencher engine

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md).

## License

MIT License
