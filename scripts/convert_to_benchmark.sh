#!/bin/bash

# Convert Novelty Engine questions to aiXamine benchmark format
# This script activates the conda environment and runs the conversion

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Activate conda environment
echo "Activating conda environment: autobencher"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobencher

# Run conversion script
python scripts/convert_to_benchmark_format.py "$@"

echo ""
echo "Conversion complete!"
