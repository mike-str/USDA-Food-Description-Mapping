# Gemma 3 Language Model Evaluation for Food Database Matching

## Experiment Overview

This experiment evaluates Google's Gemma 3 language models as part of the larger USDA Food Description Mapping project. The focus is testing how different model sizes (4b, 12b, 27b parameters) perform on automated food database matching tasks using various prompt engineering strategies. All evaluations run on SCINet's Atlas HPC infrastructure to determine optimal configurations for production deployment.

## Context Within Larger Project

This Gemma 3 evaluation builds on previous work testing Claude and other language models for food matching tasks. The experiment specifically addresses:
- Token efficiency requirements for high-volume processing
- Model size vs accuracy trade-offs on SCINet infrastructure (and others) 
- Prompt strategy optimization for Gemma's unique architecture

## Completed and Planned Evaluations

### NHANES Food Matching (Completed)
Tested Gemma 3 models on matching NHANES dietary intake descriptions to standardized food databases.

**Results:**
- Evaluated 20 distinct prompt strategies across 3 model sizes
- Best accuracy: 65.3% (Gemma3-27b with minimal text prompting)
- Identified 12b model as optimal balance between performance and compute costs
- Successfully handles ~50% no-match cases in ground truth dataset

**See [`NHANES/README.md`](NHANES/README.md) for detailed methodology and results**

### ASA24 Food Matching (Planned)
Will replicate the NHANES experiment methodology using ASA24 dataset to test model performance across different food description formats.

## Experiment Structure

```
SCINet_Gemma3_experiments/
│
├── archive/                        # Archived files and backups
│   └── scripts_txt_backup/         # Original .txt versions of documentation
│
├── docs/                           # Strategy documentation
│   ├── prompt_strategies.md       # Details on all 20 prompt strategies
│   ├── reproduction_guide.md      # Guide for reproducing experiments
│   └── SCINet_Atlas_GPU_usage.png # GPU usage visualization
│
└── NHANES/                         # NHANES dataset evaluation
    ├── data/                       # Input files and ground truth
    │   ├── README.md               # Data format specifications
    │   ├── input_desc_list_noquotes.txt
    │   ├── target_desc_list_noquotes.txt
    │   └── nhanes_dfg2_labels.csv
    ├── scripts/                    # Implementation code
    │   ├── README.md               # Execution instructions
    │   ├── SCRIPT_ARCHITECTURE.md  # Technical implementation details
    │   ├── gemma3_prompt_strategy_evaluation.py
    │   └── run_experiment.sh
    └── results/                    # Evaluation outputs
        ├── README.md               # Result analysis across models
        ├── NHANES_gemma3_4b_Run01/
        ├── NHANES_gemma3_4b_Run02/
        ├── NHANES_gemma3_12b_Run01/
        ├── NHANES_gemma3_12b_Run02/
        ├── NHANES_gemma3_27b_Run01/
        └── NHANES_gemma3_27b_Run02/
```

## Running the Experiments

### Quick Test Run
```bash
cd NHANES
bash scripts/run_experiment.sh 20   # Test with 20 sample items
```

### Full Evaluation
```bash
cd NHANES  
bash scripts/run_experiment.sh -1   # Complete dataset evaluation
```

For detailed setup and configuration options, see [`NHANES/scripts/README.md`](NHANES/scripts/README.md)

## Technical Configuration

- **Infrastructure**: SCINet Atlas vLLM Farm with 6x A100 80GB GPUs
- **Models**: Google Gemma 3 (4b, 12b, 27b parameter variants)
- **Context Window**: MAXLEN=8192 required for 256-item target lists
- **Parallelization**: 64-96 concurrent requests for optimal throughput

## Performance Summary

| Model Size | Best Overall Accuracy | Optimal Strategy | Token Efficiency |
|------------|----------------------|------------------|------------------|
| Gemma3-27b | 65.3% | S11 (text_minimal) | Text output (5-20 tokens) |
| Gemma3-12b | 57.9% | S05 (number_strict) | Number output (1-3 tokens) |
| Gemma3-4b | 55.7% | S13 (text_detailed) | Text output (5-20 tokens) |

## Identified Challenges

All Gemma 3 model sizes consistently struggle with:
- Baby food categories
- Imitation/substitute foods  
- Complex mixed dishes with multiple ingredients

These categories represent opportunities for targeted model improvements.

## Documentation

Each component has detailed documentation:
- **Data specifications**: See `data/README.md` in each dataset folder
- **Execution instructions**: See `scripts/README.md` for commands and options
- **Result analysis**: See `results/README.md` for performance metrics
- **Prompt strategies**: See [`docs/prompt_strategies.md`](docs/prompt_strategies.md)
- **Technical architecture**: See [`scripts/SCRIPT_ARCHITECTURE.md`](NHANES/scripts/SCRIPT_ARCHITECTURE.md)
- **Reproduction guide**: See [`docs/reproduction_guide.md`](docs/reproduction_guide.md)

## Next Steps

1. **ASA24 Dataset Evaluation** - Test performance on alternative food description formats
2. **Fine-tuning** - Target problematic food categories identified in NHANES
3. **Production Pipeline** - Deploy optimized 12b model for inference
4. **Transfer Learning** - Evaluate on additional food databases

## Requirements

- SCINet account with Atlas access
- vLLM server setup with Gemma 3 models
- Python environment with httpx, pandas
- Minimum 8192 token context window
