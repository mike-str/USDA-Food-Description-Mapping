# NHANES Scripts

This folder contains scripts for running the NHANES food matching experiments with Gemma 3.

## Main Scripts

### `gemma3_prompt_strategy_evaluation.py`
Main experiment script that tests 20 different prompting strategies (10 number-based, 10 text-based) for food matching accuracy

### `run_experiment.sh`
Convenience wrapper for running experiments

#### Usage
```bash
bash run_experiment.sh [sample_size] [model_size]
```

#### Examples
```bash
bash run_experiment.sh 20      # Quick test with 20 items
bash run_experiment.sh -1 27b  # Full dataset with 27b model
```

## Full Command with All Options

**NOTE**: Replace these paths with your own SCINet project directories. This example uses Richard Stoker's setup - adjust for your own paths.

### First, set up and verify environment (in new terminal):
```bash
# Replace with YOUR base path
export BASE=/90daydata/lemay_diet_guthealth/richard.stoker/testapi-a100
source "$BASE/env.farm"  # Assumes you have env.farm configured - see vLLM setup docs

# Verify environment is correct:
echo "BASE: $BASE"
echo "ENV_PREFIX: $ENV_PREFIX"
echo "ROUTER_URL: $ROUTER_URL"
echo "Python path: $ENV_PREFIX/bin/python"
if [ -n "$LITELLM_MASTER_KEY" ]; then
    echo "API Key: Loaded"
else
    echo "API Key: NOT LOADED"
fi
"$ENV_PREFIX/bin/python" --version
```

### Then run the experiment:
```bash
"$ENV_PREFIX/bin/python" "$BASE/projects/SCINet_Gemma3_experiments/NHANES/scripts/gemma3_prompt_strategy_evaluation.py" \
    --dataset NHANES \              # Dataset name for organizing results
    --model-family gemma3 \         # Model family name  
    --model-size 27b \              # Model size: 4b, 12b, or 27b
    --sample-size -1 \              # -1 for full dataset, or any number for testing
    --data-dir "$BASE/projects/SCINet_Gemma3_experiments/NHANES/data" \  # Data directory
    --concurrency 64 \              # Parallel requests (typical: 64-96)
    --base-url http://localhost:8080 \  # vLLM server URL (optional)
    --terminal-output               # Include terminal output in results (optional)
```

## Features

- Tests both number outputs (S01-S10) and text outputs (S11-S20)
- Handles "none" responses for items without matches (~50% of dataset)
- Parallel processing with configurable concurrency
- Detailed metrics and error analysis
- CSV output for further analysis

## Requirements

- vLLM server running with Gemma 3 model (requires separate vLLM stack setup)
- MAXLEN=8192 for handling 256-item target lists
- Python conda environment with httpx and pandas
- SCINet account with access to `/90daydata/` or `/project/` storage
- Note: Users need to set up their own vLLM stack and conda environment first