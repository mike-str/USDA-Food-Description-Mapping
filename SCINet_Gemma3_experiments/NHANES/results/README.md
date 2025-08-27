# NHANES Gemma 3 Model Evaluation Results

This folder contains evaluation results from testing Google's Gemma 3 models (4b, 12b, 27b parameters) on NHANES food database matching tasks using 20 different prompt strategies.

## Folder Structure

Each model size has 2 runs for validation:

### Gemma3 27b (27 billion parameters)
- `NHANES_gemma3_27b_Run01`: First evaluation run
- `NHANES_gemma3_27b_Run02`: Second evaluation run

### Gemma3 12b (12 billion parameters)
- `NHANES_gemma3_12b_Run01`: First evaluation run
- `NHANES_gemma3_12b_Run02`: Second evaluation run

### Gemma3 4b (4 billion parameters)
- `NHANES_gemma3_4b_Run01`: First evaluation run
- `NHANES_gemma3_4b_Run02`: Second evaluation run

### Failure Analysis Tools
- `failure_analysis_tools`: Scripts and outputs for analyzing failure patterns across models

## Files in Each Run Folder

| File Pattern | Description |
|--------------|-------------|
| `*.json` | Complete results with all metrics for each strategy |
| `*_detailed_matches.csv` | Every prediction with correctness flags |
| `*_strategy_matrix.csv` | Performance comparison matrix |
| `*_summary.csv` | Quick summary metrics |
| `*_report.txt` | Console output from the run |

## Model Size Comparison

### Overall Performance by Model Size

| Model | Best Overall Accuracy | Top Strategy | Match Accuracy | No-Match Accuracy |
|-------|----------------------|--------------|----------------|-------------------|
| **Gemma3 27b** | 65.3% | S11 (text_minimal) | 48.9% | 83.7% |
| **Gemma3 12b** | 57.9% | S05 (number_strict) | 25.7% | 93.9% |
| **Gemma3 4b** | 55.7% | S13 (text_detailed) | 20.7% | 94.8% |

## Top Performing Strategies

### Best Overall Balance (Match + No-Match)

1. **S11 (text_minimal)**: Simple text matching, top performer for 27b
2. **S05 (number_strict)**: Strict number matching with default-to-none bias
3. **S18 (text_smart_categ)**: Category-aware text matching

### Best for Matching (when items should match)

- **S17 (text_contextual)**: Up to 75% accuracy on 27b model
- **S15 (text_semantic)**: Consistent across model sizes

### Best for No-Match Detection (when items shouldn't match)

- **S16 (text_strict)**: 93% accuracy on 27b
- **S14 (text_ruler_fewshot)**: 95-100% accuracy, excellent at rejection

## Model-Specific Patterns

### Gemma3 27b
- Excels with minimal prompts (S11)
- Text strategies outperform number strategies

### Gemma3 12b
- Prefers structured prompts (S05 number_strict)
- Similar performance between number and text strategies

### Gemma3 4b
- Needs detailed instructions (S13 text_detailed)
- Struggles more with number strategies

## Common Failure Patterns

### 1. Ground Meat Products (15 items with 100% failure)
- "beef, ground, 80% lean meat / 20% fat, crumbles, cooked, pan-browned" → Expected: beef round tip steak
- "beef, ground, 85% lean meat / 15% fat, patty, cooked, broiled" → Expected: beef round tip steak
- Models incorrectly match to ground pork or fail to match at all

### 2. Canned Vegetables (35 items with high failure)
- "beans, snap, green, canned, no salt added, drained solids" → Expected: steamed green beans
- "beans, snap, yellow, canned, regular pack, drained solids" (100% failure)
- Consistently matched to wrong bean types (black beans, kidney beans)

### 3. Baby Food Categories (5 items with 100% failure)
- "babyfood, fortified cereal bar, fruit filling" (all strategies failed)
- "babyfood, cereal, barley, dry fortified" (all strategies failed)
- Models ignore the "babyfood" context entirely

### 4. Hot Peppers (100% failure rate)
- "peppers, hot chili, green, raw" → Expected: jalapeno
- "peppers, hot chili, red, canned, excluding seeds, solids and liquids" → Expected: jalapeno
- All strategies match to bell peppers instead

### 5. Processed Meats (100% failure on bacon/ham)
- "bacon, turkey, low sodium" → Expected: ground turkey
- "canadian bacon, cooked, pan-fried" → Expected: ground pork
- Models match to plant-based bacon or fail entirely

## Technical Notes

### Error Types
- **missed_match**: Model said "none" when a match exists
- **wrong_match**: Model selected incorrect match
- **false_positive**: Model found match when none exists

Models were told to say "none" for ~50% of items that have no valid match. However, they also incorrectly say "none" for many items that DO have matches (missed_match errors). Smaller models are overly cautious - 4b model's errors are 81% missed matches, meaning it says "none" too often even for valid matches.

### Prompt Strategy Performance
- Text strategies generally outperform number strategies
- Simpler prompts work better for larger models
- Smaller models need more detailed instructions
- All models benefit from explicit "none" option instructions

For detailed metrics and individual predictions, refer to the JSON and CSV files in each run folder.
