# NHANES Food Database Matching Experiment

## Overview

This experiment evaluates Google's Gemma 3 language models (4b, 12b, 27b parameters) on automated food database matching tasks. We test how well different model sizes can map NHANES dietary intake descriptions to a standardized food database using various prompting strategies.

## What We're Testing

- **3 Model Sizes**: Gemma3-4b, Gemma3-12b, Gemma3-27b
- **20 Prompt Strategies**: 10 number-based outputs, 10 text-based outputs
- **1,317 Source Items**: NHANES dietary intake descriptions
- **256 Target Items**: Standardized food database entries
- **Ground Truth**: ~50% items have matches, ~50% have no valid match

## Technical Challenges

1. **Token Efficiency**: Number outputs (1-3 tokens) vs text outputs (5-20+ tokens)
2. **Attention Drift**: Gemma 3 struggles with numerical position tracking
3. **No-Match Detection**: Models must correctly identify when no match exists
4. **Context Window**: Requires MAXLEN=8192 for 256-item target lists

## Quick Start

```bash
# Test with small sample (20 items):
bash scripts/run_experiment.sh 20

# Full dataset evaluation:
bash scripts/run_experiment.sh -1

# Custom configuration:
# See scripts/README.md for full command options
```

## Project Structure

```
NHANES/
├── data/
│   ├── README.md                          # Explains input files and ground truth format
│   ├── input_desc_list_noquotes.txt      # 1,317 NHANES food descriptions
│   ├── target_desc_list_noquotes.txt     # 256 standardized database entries
│   └── nhanes_dfg2_labels.csv            # Ground truth matching labels
│
├── scripts/
│   ├── README.md                          # How to run experiments with different configurations
│   ├── SCRIPT_ARCHITECTURE.md             # Technical details of the evaluation system
│   ├── gemma3_prompt_strategy_evaluation.py  # Main experiment script
│   └── run_experiment.sh                  # Convenience script for running experiments
│
└── results/
    ├── README.md                          # Analysis of results across all model sizes
    ├── NHANES_gemma3_4b_Run01/           # Individual experiment outputs
    ├── NHANES_gemma3_4b_Run02/
    ├── NHANES_gemma3_12b_Run01/
    ├── NHANES_gemma3_12b_Run02/
    ├── NHANES_gemma3_27b_Run01/
    └── NHANES_gemma3_27b_Run02/
```

## Documentation

For detailed documentation on specific components:
- **Prompt Strategies**: [`../docs/prompt_strategies.md`](../docs/prompt_strategies.md)
- **Script Architecture**: [`scripts/SCRIPT_ARCHITECTURE.md`](scripts/SCRIPT_ARCHITECTURE.md)
- **Results Analysis**: [`results/README.md`](results/README.md)
- **Reproduction Guide**: [`../docs/reproduction_guide.md`](../docs/reproduction_guide.md)

## Current Status

Completed testing shows:
- **27b model**: Achieves 65.3% overall accuracy (S11: text_minimal strategy)
- **12b model**: Achieves 57.9% overall accuracy (S05: number_strict strategy)
- **4b model**: Achieves 55.7% overall accuracy (S13: text_detailed strategy)
- **Diminishing returns**: 12b model provides optimal cost-performance ratio

All models correctly identify no-match cases but struggle with:
- Baby food categories
- Imitation/substitute foods
- Complex mixed dishes with multiple ingredients

## Next Steps

1. Fine-tune Gemma 3 on problematic categories
2. Implement hybrid pipeline
3. Run experiment on larger food databases
