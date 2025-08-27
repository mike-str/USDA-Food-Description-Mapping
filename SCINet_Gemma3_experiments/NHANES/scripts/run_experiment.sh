#!/bin/bash
# ================================================================================
# Run Gemma 3 Prompt Strategy Evaluation Experiment
# Version: 1.0
# ================================================================================
#
# Purpose:
#   Execute prompt strategy evaluation for Gemma 3 food database matching
#   on SCINet Atlas vLLM GPU farm infrastructure.
#
# Usage:
#   source "$BASE/env.farm"
#   bash run_experiment.sh [sample_size] [model_size]
#
#   Examples:
#   bash run_experiment.sh 20           # Test with 20 items on default 27b
#   bash run_experiment.sh 100 27b      # Test with 100 items on 27b
#   bash run_experiment.sh -1 27b       # Full dataset on 27b
#   bash run_experiment.sh 100 12b      # Test with 100 items on 12b
#
# ================================================================================

# Exit on error
set -e

# Source environment if needed
if [ -z "$BASE" ]; then
    export BASE="/90daydata/lemay_diet_guthealth/richard.stoker/testapi-a100"
    source "$BASE/env.farm"
fi

# Parse arguments
SAMPLE_SIZE="${1:-20}"  # Default to 20 for quick test
MODEL_SIZE="${2:-27b}"  # Default to 27b

# Set model name based on size
MODEL_NAME="gemma3-${MODEL_SIZE}"

# Configuration
EXPERIMENT_DIR="$BASE/projects/SCINet_Gemma3_experiments/NHANES"
SCRIPTS_DIR="$EXPERIMENT_DIR/scripts"
DATA_DIR="$EXPERIMENT_DIR/data"
RESULTS_DIR="$EXPERIMENT_DIR/results"

# Display configuration
echo "=================================================================================="
echo "Gemma 3 Prompt Strategy Evaluation Experiment"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  Dataset: NHANES"
echo "  Model Family: Gemma3"
echo "  Model Size: $MODEL_SIZE"
echo "  Full Model: $MODEL_NAME"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Data Directory: $DATA_DIR"
echo "  Results Directory: $RESULTS_DIR"
echo ""

# Check data files exist
if [ ! -f "$DATA_DIR/input_desc_list_noquotes.txt" ]; then
    echo "ERROR: Input data file not found!"
    exit 1
fi

if [ ! -f "$DATA_DIR/target_desc_list_noquotes.txt" ]; then
    echo "ERROR: Target data file not found!"
    exit 1
fi

if [ ! -f "$DATA_DIR/nhanes_dfg2_labels.csv" ]; then
    echo "ERROR: Ground truth file not found!"
    exit 1
fi

echo "Data files verified."
echo ""

# Run the experiment
echo "Starting experiment..."
echo "Command:"
echo "  $ENV_PREFIX/bin/python $SCRIPTS_DIR/gemma3_prompt_strategy_evaluation.py \\"
echo "    --dataset NHANES \\"
echo "    --model-family Gemma3 \\"
echo "    --model-size $MODEL_SIZE \\"
echo "    --model $MODEL_NAME \\"
echo "    --sample-size $SAMPLE_SIZE \\"
echo "    --data-dir $DATA_DIR \\"
echo "    --concurrency 64"
echo ""
echo "=================================================================================="
echo ""

# Execute
"$ENV_PREFIX/bin/python" "$SCRIPTS_DIR/gemma3_prompt_strategy_evaluation.py" \
    --dataset NHANES \
    --model-family Gemma3 \
    --model-size "$MODEL_SIZE" \
    --model "$MODEL_NAME" \
    --sample-size "$SAMPLE_SIZE" \
    --data-dir "$DATA_DIR" \
    --concurrency 64

echo ""
echo "=================================================================================="
echo "Experiment complete!"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "To run with different configurations:"
echo "  Small test (20 items):    bash run_experiment.sh 20 27b"
echo "  Medium test (100 items):  bash run_experiment.sh 100 27b"
echo "  Full dataset:             bash run_experiment.sh -1 27b"
echo "  Different model size:     bash run_experiment.sh 100 12b"
echo "=================================================================================="