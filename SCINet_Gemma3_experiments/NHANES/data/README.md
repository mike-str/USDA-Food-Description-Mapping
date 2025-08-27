# NHANES Data Files

This folder contains the data files for the NHANES food matching experiment.

## Files

| File | Description |
|------|-------------|
| `input_desc_list_noquotes.txt` | 1,317 NHANES food descriptions (source items to match) |
| `target_desc_list_noquotes.txt` | 256 standardized food database items (target items) |
| `nhanes_dfg2_labels.csv` | Ground truth labels (1 = match exists, 0 = no match) |

## Format

- **Text files**: One food item per line, no quotes
- **CSV file**: Columns include source item, target item (if match exists), and label

## Usage

These files are read by the `gemma3_prompt_strategy_evaluation.py` script to test different prompting strategies for food matching accuracy.