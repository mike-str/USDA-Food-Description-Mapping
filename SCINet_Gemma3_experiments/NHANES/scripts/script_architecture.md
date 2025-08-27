# GEMMA3_PROMPT_STRATEGY_EVALUATION.PY - TECHNICAL ARCHITECTURE

## OVERVIEW

This script implements a prompt engineering evaluation system for testing 
Google's Gemma 3 language models on food database matching tasks. The
architecture supports parallel processing, multiple prompt strategies, and 
performance metrics collection in a single executable.

---

## CORE ARCHITECTURE

### 1. DATA STRUCTURES

The script uses dataclasses for type safety and clean data management:

- **PromptStrategy**: Encapsulates prompt configuration
  * `strategy_id` (S01-S20): Unique identifier for tracking
  * `response_format`: "number" or "text" output mode
  * `system_prompt`: Instructions merged into user message (Gemma 3 limitation)
  * `user_prompt_template`: Template with {source} and {targets} placeholders
  * `build_messages()`: Constructs API-compatible message format

- **TestResult**: Captures individual test outcomes
  * `source/expected/predicted`: Food items and predictions
  * `has_match`: Boolean from ground truth
  * `correct`: Calculated accuracy flag
  * `error_type`: Categorized failure mode
  * `response_time`: API latency measurement
  * `raw_response`: Debugging reference

### 2. PROMPT STRATEGY SYSTEM

20 strategies implemented (S01-S20) testing different approaches:

**Number Strategies (S01-S10):**
- Return list position numbers (1-256)
- Minimal token usage (1-3 tokens for numbers)
- Challenge: Gemma 3 attention drift with numbers

**Text Strategies (S11-S20):**
- Return exact text matches
- Higher token usage (5-20+ tokens depending on description length)
- More reliable for Gemma 3

Each strategy tests a different prompting technique:
- Direct matching (S01/S11)
- Chain-of-thought (S02/S12)
- Expert role-play (S04/S14)
- Categorization (S08/S18)
- Gamification (S09/S19)
- Verification steps (S10/S20)

### 3. RESPONSE PARSING ENGINE

`parse_response()` implements strict validation:

**For Number Format:**
- Regex pattern: `r"(?:none|[1-9]\d*)"`
- Validates range: 1 <= number <= len(targets)
- Maps to target text by index
- Returns "none" for invalid/out-of-range

**For Text Format:**
- Case-insensitive exact matching
- No fuzzy matching (prevents false positives)
- Returns "none" if no exact match

**Error Handling:**
- Technical failures return "error" (not "none")
- Maintains separation between model decisions and API failures

### 4. PARALLEL PROCESSING ARCHITECTURE

Async/await pattern with semaphore-controlled concurrency:

**`async def test_single_item()`:**
- Acquires semaphore slot
- Builds prompt messages
- Makes async HTTP POST to vLLM
- Handles HTTP errors → returns "error"
- Handles exceptions → returns "error"
- Parses successful responses
- Calculates correctness
- Returns TestResult

**`async def test_strategy()`:**
- Creates semaphore(limit=concurrency)
- Launches all items as async tasks
- Gathers results with `asyncio.gather()`
- Calculates aggregate metrics

**Concurrency Control:**
- Default: 64 parallel requests
- Tunable: 32-128 depending on vLLM capacity
- Prevents server overload
- Maintains throughput optimization

### 5. CORRECTNESS CALCULATION

Five-way classification matrix:

**When predicted == "error" (technical failure):**
- Always INCORRECT (technical_error)
- Tracked separately from model errors

**When has_match=True (item should have a match):**
- predicted == target → CORRECT (true positive)
- predicted == "none" → INCORRECT (missed_match)
- predicted != target → INCORRECT (wrong_match)

**When has_match=False (no valid match exists):**
- predicted == "none" → CORRECT (true negative)
- predicted != "none" → INCORRECT (false_positive)

### 6. METRICS AGGREGATION

Per-strategy metrics collected:

**Performance Metrics:**
- `overall_accuracy`: (correct / total)
- `match_accuracy`: Accuracy on items with matches
- `no_match_accuracy`: Accuracy on items without matches
- `actual_throughput`: items/second achieved

**Error Analysis:**
- `missed_match`: Should have found match but said "none"
- `wrong_match`: Found incorrect match
- `false_positive`: Said match when should be "none"
- `technical_error`: API failure, timeout, or HTTP error

**Token Efficiency:**
- Tracked implicitly through response_format
- Number strategies: 1-3 tokens per response
- Text strategies: 5-20+ tokens per response (varies by food description length)

### 7. DATA PIPELINE

**Input Flow:**
1. Load source items (1,317 NHANES descriptions)
2. Load target items (256 database entries)
3. Load ground truth (CSV with has_match flags)
4. Sample if requested (--sample-size parameter)

**Processing Flow:**
1. Initialize vLLM client connection
2. For each strategy (sequential):
   - Test all items (parallel)
   - Collect results
   - Calculate metrics
   - Store for comparison
3. Generate comparative analysis
4. Output results

**Output Generation:**
- Console output with formatted tables
- JSON summaries for programmatic access
- CSV files for statistical analysis
- Detailed match records for debugging

---

## IMPLEMENTATION OPTIMIZATIONS

### 1. MEMORY EFFICIENCY

- Streaming results processing (no full dataset in memory)
- Async generators for large result sets
- Immediate metric calculation (no storage of raw responses)

### 2. ERROR RESILIENCE

- Try/except blocks around API calls
- Timeout handling (30s default)
- Failed requests marked as "error"
- Graceful degradation on API errors
- Technical errors tracked separately from model errors

### 3. GEMMA 3 SPECIFIC ADAPTATIONS

- No system prompt (merged into user message)
- Ruler info prepended for context awareness
- Strict response parsing (Gemma 3 varies output format)
- MAXLEN=8192 requirement handled externally

### 4. PROGRESS TRACKING

- Real-time progress updates per strategy
- Throughput calculations (items/sec)
- ETA estimates for long runs
- Verbose logging option

---

## CONFIGURATION PARAMETERS

**Required Parameters:**
- `--dataset`: Names the experiment run
- `--model-family`: For result organization
- `--model-size`: 4b/12b/27b variant

**Performance Tuning:**
- `--concurrency`: Parallel request limit (default: 64)
- `--sample-size`: -1 for full dataset, N for testing

**Infrastructure:**
- `--base-url`: vLLM server endpoint (default: localhost:8080)
- `--data-dir`: Path to input files

---

## OUTPUT FILE STRUCTURE

**Results Directory:** `results/[timestamp]_[dataset]_[model]_[size]/`

**Files Generated:**

1. **summary.json**
   - Experiment metadata
   - Overall metrics
   - Best performer identification

2. **strategy_results.json**
   - Detailed per-strategy metrics
   - Error breakdowns
   - Throughput measurements

3. **[experiment]_[timestamp]_strategy_matrix.csv**
   - Comparative matrix format
   - All strategies x all metrics
   - Ready for statistical analysis

4. **[experiment]_[timestamp]_detailed_matches.csv**
   - Every prediction recorded
   - Correctness flags
   - Response times
   - Error classifications

---

## USAGE PATTERNS

**Quick Testing:**
- Use `--sample-size 20` for rapid iteration
- Validates prompt changes
- Checks API connectivity

**Full Evaluation:**
- Use `--sample-size -1` for complete dataset
- Generates statistically significant results

**Model Comparison:**
- Run same experiment with different `--model-size`
- Compare accuracy vs model size
- Identify minimum viable model

**Prompt Development:**
- Modify strategy definitions in `get_all_prompt_strategies()`
- Test with small sample first
- Scale to full dataset once validated

---

## PERFORMANCE CHARACTERISTICS

**Throughput Expectations:**
- Concurrency=64: ~60-80 items/sec
- Concurrency=96: ~80-100 items/sec
- Limited by vLLM server capacity

**CPU Memory Usage:**
- Base: ~200MB Python overhead
- Per-strategy: ~50MB result storage
- Peak: ~500MB for full run

**Network Requirements:**
- Sustained: ~5-10 Mbps to vLLM server
- Latency sensitive: <100ms ideal
- Handles transient failures

---

## EXTENSION POINTS

**Adding New Prompt Strategies:**
1. Define in `get_all_prompt_strategies()`
2. Assign unique strategy_id (S21, S22, etc.)
3. Set response_format ("number" or "text")
4. Implement prompt template

**Custom Metrics:**
- Extend TestResult dataclass
- Add calculation in `test_strategy()`
- Include in output formatting

**Alternative Models:**
- Modify API parameters in `test_single_item()`
- Adjust parsing logic if needed
- Update MAXLEN requirements

---

**AUTHOR:** Richard Stoker  
**BASED ON:** NHANES Experiment 4 by Dr. Danielle Lemay  
**ENVIRONMENT:** SCINet Atlas vLLM Farm