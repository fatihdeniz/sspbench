#!/bin/bash
# Novelty Engine Runner Script
# Wrapper script that activates conda environment and runs the Python script

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Novelty Engine Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# Activate conda environment
CONDA_ENV="${CONDA_ENV:-autobencher}"
echo -e "\n${YELLOW}Activating conda environment: ${CONDA_ENV}${NC}"

# Source conda
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo -e "${RED}✗ Could not find conda. Please ensure conda is installed.${NC}"
    exit 1
fi

# Activate environment
if conda activate "${CONDA_ENV}" 2>/dev/null; then
    echo -e "${GREEN}✓ Activated conda environment: ${CONDA_ENV}${NC}"
else
    echo -e "${RED}✗ Failed to activate conda environment: ${CONDA_ENV}${NC}"
    echo -e "${YELLOW}Available environments:${NC}"
    conda env list
    exit 1
fi

# Check Python version
echo -e "\n${YELLOW}Python version:${NC}"
python --version

# Navigate to project root
cd "${PROJECT_ROOT}"
echo -e "\n${YELLOW}Working directory: ${PWD}${NC}"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Run the Python script
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Novelty Engine...${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Pass all arguments to the Python script
python "${SCRIPT_DIR}/run_novelty_engine.py" "$@"
EXIT_CODE=$?

echo -e "\n${GREEN}========================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Script completed successfully${NC}"
else
    echo -e "${RED}✗ Script exited with code: ${EXIT_CODE}${NC}"
fi
echo -e "${GREEN}========================================${NC}"

exit $EXIT_CODE
