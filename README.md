# SSP Bench

Safety, Security, and Privacy Dynamic Benchmark Generation Framework

## Overview

SSP Bench is a comprehensive framework for generating and evaluating dynamic benchmarks focused on the safety, security, and privacy aspects of large language models (LLMs). The framework provides tools to create adversarial examples, assess model vulnerabilities, and measure privacy risks.

## Features

- **Dynamic Benchmark Generation**: Generate custom benchmarks for safety, security, and privacy evaluation
- **Modular Evaluators**: Evaluate models using specialized evaluators for each domain
- **Comprehensive Metrics**: Compute relevant metrics for safety, security, and privacy
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

## Project Structure

- `sspbench/generators/`: Benchmark generation modules
- `sspbench/evaluators/`: Model evaluation modules
- `sspbench/metrics/`: Metric computation functions
- `sspbench/utils/`: Utility functions
- `benchmarks/`: Pre-generated benchmark datasets
- `tools/`: Integration with existing benchmark tools

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md).

## License

MIT License
