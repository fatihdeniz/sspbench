# Novelty Engine Scripts

This directory contains scripts for running the Novelty Engine from the command line.

## Quick Start

### Basic Usage

```bash
# Run with default settings
cd /home/local/QCRI/fdeniz/projects/sspbench
./scripts/run_novelty.sh
```

### Custom Configuration

```bash
# Run with custom iterations and theme
./scripts/run_novelty.sh --max-iterations 3 --theme "science"

# Run without RAGAS quality control
./scripts/run_novelty.sh --no-ragas

# Save questions to text file and show detailed analysis
./scripts/run_novelty.sh --save-txt --analyze-results

# Use different CUDA device
./scripts/run_novelty.sh --cuda-device 0
```

## Scripts Overview

### `run_novelty.sh` (Recommended)

Bash wrapper that:

- Activates the conda environment automatically
- Sets up proper error handling
- Provides colored output for better readability
- Passes all arguments to the Python script

### `run_novelty_engine.py`

Main Python script that:

- Initializes all models (agent, test, eval)
- Runs the novelty engine with RAGAS quality control
- Generates factuality QA benchmark questions
- Saves results to `data/novelty/` directory
- Provides detailed logging

## Command-Line Options

### Model Configuration

- `--agent-model`: Agent model name (default: gpt-4.1-mini-aixamine)
- `--agent-api-url`: Agent model API URL
- `--agent-api-token`: Agent model API token
- `--test-model`: Test model path (default: local gemma-2-2b)
- `--eval-model`: Evaluation model name (default: gpt-oss)
- `--eval-api-url`: Evaluation model API URL
- `--eval-api-token`: Evaluation model API token

### Engine Parameters

- `--theme`: Theme for question generation (default: "general knowledge")
- `--max-iterations`: Maximum iterations (default: 5)
- `--acc-target`: Target accuracy range (default: "0.1--0.4")
- `--engine`: Engine name for file organization (default: "novelty")
- `--seed-index-path`: Path to seed topics JSON file

### RAGAS Options

- `--no-ragas`: Disable RAGAS quality control
- `--embedding-model`: Embedding model for RAGAS (default: all-MiniLM-L6-v2)

### Output Options

- `--save-txt`: Save all questions to a text file
- `--analyze-results`: Print detailed results analysis

### System Options

- `--cuda-device`: CUDA device ID (default: 7)

## Environment Variables

```bash
# Change conda environment (default: autobencher)
export CONDA_ENV=myenv
./scripts/run_novelty.sh

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
./scripts/run_novelty.sh
```

## Output Files

Results are saved in `data/novelty/`:

### Per-Iteration Files

- `{theme}_iter_{N}.KI_questions.json` - Generated questions with metadata
- `{theme}_iter_{N}.categories_augmented.json` - Generated categories
- `{theme}_iter_{N}.compare_answers.json` - Model comparison results

### Question Metadata

Each question includes:

- `id`, `question`, `gold_answer`
- `category`, `parent_category`
- `wiki_entity`, `wiki_url`, `context`
- `faithfulness`: Faithfulness score (0-1)
- `answer_relevancy`: Answer relevancy score (0-1)
- `quality`: Quality assessment from LLM judge
  - `is_suitable`: Boolean
  - `scores`: 5-dimensional scores (each 0-2)
  - `total_score`: Total quality score (0-10)
  - `reasoning`: LLM judge explanation
  - `round_generated`: Which regeneration round (0, 1, 2)

## Examples

### Run 3 iterations on science theme

```bash
./scripts/run_novelty.sh \
    --theme "science" \
    --max-iterations 3 \
    --save-txt \
    --analyze-results
```

### Run without quality control (faster)

```bash
./scripts/run_novelty.sh \
    --no-ragas \
    --max-iterations 2
```

### Use specific GPU

```bash
./scripts/run_novelty.sh \
    --cuda-device 0 \
    --theme "history"
```

### Direct Python execution (if conda already activated)

```bash
cd /home/local/QCRI/fdeniz/projects/sspbench
python scripts/run_novelty_engine.py --help
```

## Logs

Logs are saved to:

- `logs/novelty_engine.log` - Detailed execution log
- Console output - Real-time progress

## Troubleshooting

### Conda environment not found

```bash
# List available environments
conda env list

# Set correct environment
export CONDA_ENV=autobencher
./scripts/run_novelty.sh
```

### CUDA out of memory

```bash
# Use different GPU
./scripts/run_novelty.sh --cuda-device 0

# Or reduce batch size in config
```

### Missing dependencies

```bash
conda activate autobencher
pip install ragas langchain-core sentence-transformers json5
```

### Script exits immediately

Check logs:

```bash
tail -n 50 logs/novelty_engine.log
```

## Quality Control Pipeline

When `--no-ragas` is **not** specified (default):

1. **RAGAS Question Generation**: Uses single-hop specific query synthesizer
2. **Quality Assessment**: LLM judge evaluates each question (5 dimensions, 0-2 each)
3. **Adaptive Regeneration**: Up to 3 rounds if quality is low
4. **Feedback Learning**: Failed reasons guide next generation round

Quality threshold: 6/10 (configurable in `sspbench/novelty/config.py`)

## Performance

Typical runtime per iteration:

- With RAGAS quality control: ~5-10 minutes
- Without quality control: ~2-3 minutes
- Depends on: number of categories, model speed, regeneration rounds

## Tips

1. **Start small**: Test with `--max-iterations 1` first
2. **Monitor quality**: Check quality scores in KI_questions files
3. **Use logging**: Check `logs/novelty_engine.log` for detailed info
4. **Save text files**: Use `--save-txt` for easy question review
5. **GPU memory**: Monitor with `nvidia-smi` during execution
