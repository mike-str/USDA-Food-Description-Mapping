#!/usr/bin/env python3
"""
================================================================================
Gemma 3 Prompt Strategy Evaluation for Food Database Matching
Version: 1.0
================================================================================

Purpose:
    All-in-one evaluation of prompt engineering strategies for automated food database
    matching using Google's Gemma 3 model variants (4b, 12b, 27b).
    Tests both number-based and text-based response formats to measure accuracy,
    token efficiency, and performance characteristics across different model sizes
    on SCINet Atlas vLLM GPU farm infrastructure.

Experiment Design:
    - Compares 20 distinct prompting strategies (10 number, 10 text)
    - Tests accuracy vs speed trade-offs across Gemma 3 model sizes
    - Evaluates token efficiency (number outputs vs full text descriptions)
    - Measures performance on 6x A100 80GB GPU vLLM farm setup
    - Validates against NHANES ground truth labels

Key Capabilities:
    - Model-specific prompt optimization (Gemma 3 has no system prompt support)
    - Parallel processing with configurable concurrency (64-96 typical)
    - Detailed metrics tracking and error analysis
    - Structured output formats for cross-model comparison

Output Files:
    - Detailed performance metrics for each prompting strategy
    - Comparison matrices between response formats
    - Error pattern analysis and recommendations
    - Structured JSON for cross-model/strategy analysis
    - CSV datasets for statistical analysis

Author: Richard Stoker
Date: August 2024
Environment: SCINet Atlas vLLM GPU Farm (6x A100 80GB)
Based on: NHANES Experiment 4 by Dr. Danielle Lemay
Original: https://github.com/mike-str/USDA-Food-Description-Mapping/tree/main/Claude_API_experiments/NHANES%20experiment%204
================================================================================
"""

import argparse
import asyncio
import json
import os
import random
import time
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
from collections import defaultdict

import httpx


# ============================= Data Classes =============================

@dataclass
class PromptStrategy:
    """Represents a single prompt strategy configuration."""
    name: str
    strategy_id: str  # Unique ID like S01, S02, etc.
    response_format: str  # "number" or "text"
    system_prompt: str
    user_prompt_template: str
    description: str
    
    def build_messages(self, source: str, targets: List[str]) -> List[Dict]:
        """Build the message list for this strategy (Gemma 3 compatible - no system role)."""
        list_len = len(targets)  # Calculate list length for all prompts
        
        # Build ruler info based on response format
        if self.response_format == "number":
            # 1-based numbering to match your prompts
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(targets))
            user_body = self.user_prompt_template.format(
                source=source, 
                targets=numbered, 
                list_type="numbered list", 
                list_len=list_len
            )
            # Always prepend ruler info for number strategies
            ruler_info = f"N = {list_len}\nReturn ONLY one token: 1..N or none"
        else:
            text_list = "\n".join(f"- {t}" for t in targets)
            user_body = self.user_prompt_template.format(
                source=source, 
                targets=text_list, 
                list_type="list", 
                list_len=list_len
            )
            # Always prepend ruler info for text strategies
            ruler_info = f"TARGET LIST LENGTH = {list_len}\nReturn ONLY exact text from this list, or 'none'"

        # Format system_prompt with list_len if it contains the placeholder
        system_prompt_formatted = self.system_prompt
        if "{list_len}" in system_prompt_formatted:
            system_prompt_formatted = system_prompt_formatted.format(list_len=list_len)

        # Merge ruler_info + system_prompt + user_body into single user message
        merged = "\n\n".join(
            s for s in [ruler_info.strip(), (system_prompt_formatted or "").strip(), user_body.strip()] if s
        )

        # IMPORTANT: single user turn; NO system role for Gemma 3
        return [{"role": "user", "content": merged}]


@dataclass
class TestResult:
    """Results from testing a single item."""
    source: str
    expected: Optional[str]
    predicted: str
    has_match: bool
    correct: bool
    error_type: Optional[str]
    response_time: float
    raw_response: str


# ============================= Prompt Strategies =============================

def get_all_prompt_strategies() -> List[PromptStrategy]:
    """Define all prompt strategies to test."""
    strategies = []
    
    # ========== NUMBER-BASED STRATEGIES ==========
    
    strategies.append(PromptStrategy(
        name="number_minimal",
        strategy_id="S01",
        response_format="number",
        system_prompt="Match foods. Reply number or none if no match.",
        user_prompt_template="Match: {source}\n\nList:\n{targets}\n\nNumber (or none):",
        description="Minimal instructions with number response"
    ))
    
    strategies.append(PromptStrategy(
        name="number_simple",
        strategy_id="S02",
        response_format="number",
        system_prompt="You match food items. Given a source and targets, pick the BEST match number. If no good match exists (similarity < 50%), say none.\n\nReply ONLY the number or none.",
        user_prompt_template="Match this: {source}\n\nTo:\n{targets}\n\nBest match:",
        description="Simple clear instructions with number response"
    ))
    
    strategies.append(PromptStrategy(
        name="number_detailed",
        strategy_id="S03",
        response_format="number",
        system_prompt="""Food matching expert. Match based on:
1. Same food/ingredient type
2. Processing level (raw, cooked, canned)
3. Preparation form (whole, sliced, ground)

Output ONLY a number or none if no match.""",
        user_prompt_template="SOURCE: {source}\n\nTARGETS:\n{targets}\n\nMatch (number or none):",
        description="Detailed criteria with number response"
    ))
    
    strategies.append(PromptStrategy(
        name="number_ruler_fewshot",
        strategy_id="S04",
        response_format="number",
        system_prompt="""Food matching with strict numeric output.

N = {list_len}

Return ONLY one token: a number between 1 and N, or none.""",
        user_prompt_template="""SOURCE: {source}

TARGETS:
{targets}

Examples:
butter → 2
dragonfruit puree → none

Now match SOURCE. Your answer (1..N or none):""",
        description="Ruler + few-shot guidance for numeric output"
    ))
    
    strategies.append(PromptStrategy(
        name="number_strict",
        strategy_id="S05",
        response_format="number",
        system_prompt="""CRITICAL MATCHING RULES:
    
DEFAULT TO none. Only match when CERTAIN (similarity > 50%).

Match ONLY when ALL conditions met:
1. EXACT same base food (beef→beef, NOT beef→pork)
2. SAME preparation state (raw→raw, cooked→cooked)
3. SAME form (ground→ground, whole→whole)

ALWAYS return none for:
- Different food categories
- Different cooking methods
- Mixed/prepared foods without exact match
- ANY uncertainty

Output: Single number or none. Nothing else.""",
        user_prompt_template="""Item: {source}

Options:
{targets}

Match (number or none - default to none when uncertain):""",
        description="Strict matching with default-to-none bias"
    ))
    
    strategies.append(PromptStrategy(
        name="number_semantic",
        strategy_id="S06",
        response_format="number",
        system_prompt="Match based on semantic similarity. Consider synonyms and related terms. Output number or none if no match.",
        user_prompt_template="Semantically match: {source}\n\nTo list:\n{targets}\n\nBest semantic match:",
        description="Semantic similarity focus with number response"
    ))
    
    strategies.append(PromptStrategy(
        name="number_contextual",
        strategy_id="S07",
        response_format="number",
        system_prompt="""You are matching food descriptions. Return the number of the best match or none.
Consider: cooking method, preservation, form, and primary ingredient. Say none if similarity < 50%.""",
        user_prompt_template="""Food to match: "{source}"

Available options:
{targets}

Instructions:
- Identify the most similar food
- Return its number
- Return none if no good match exists
- No explanations, just the number or none

Your answer:""",
        description="Contextual understanding with number response"
    ))
    
    strategies.append(PromptStrategy(
        name="number_smart_categ",
        strategy_id="S08",
        response_format="number",
        system_prompt="""Expert food matcher using intelligent categorization.

Your process:
1. Analyze the target list to identify natural food categories
2. Determine if source fits any identified category
3. Use category fit to guide matching confidence
4. Output ONLY a number (1-N) or none

Key insights:
- Items outside target categories likely have no match
- Saying none when appropriate demonstrates expertise
- About 50% of items have no match""",
        user_prompt_template="""Matching task requiring category awareness:

SOURCE: {source}

TARGET LIST:
{targets}

Step-by-step approach:
1. Mentally group targets into natural categories based on what you see
2. Check: Does the source belong to any of these categories?
   - If YES: Search for best match within that category (but none is still valid)
   - If NO: Source is likely outside scope (probably none, but still check)
3. Make your decision with appropriate confidence

Remember:
- Correctly saying none is as valuable as identifying matches
- Don't force matches - about half the time none is correct
- If source doesn't fit target categories, none is likely right
- Still attempt matching, but with calibrated skepticism

Answer (number or none):""",
        description="Smart categorization with dynamic analysis"
    ))
    
    strategies.append(PromptStrategy(
        name="number_point_game",
        strategy_id="S09",
        response_format="number",
        system_prompt="""Food matching game - maximize your score!

SCORING RULES:
+1 point: Correct match (selecting the right target)
+1 point: Correct rejection (saying none when no match exists)
-1 point: Wrong match (picking incorrect target)
-1 point: False rejection (saying none when match exists)

Your goal: Maximum points through accurate decisions.
Output: Number (1-N) or none only.""",
        user_prompt_template="""FOOD MATCHING GAME - Maximize your score!

SOURCE: {source}

AVAILABLE TARGETS:
{targets}

SCORING SYSTEM:
✓ Find correct match = +1 point
✓ Correctly say none = +1 point
✗ Pick wrong match = -1 point
✗ Say none when match exists = -1 point

STRATEGY TIPS:
- About 50% have matches, 50% don't
- Being right about none scores just as much as identifying matches
- Forcing bad matches costs points
- Trust your judgment - uncertain matches are often none

Make your play (number or none):""",
        description="Gamified matching with point-based motivation for balanced accuracy"
    ))
    
    strategies.append(PromptStrategy(
        name="number_validation",
        strategy_id="S10",
        response_format="number",
        system_prompt="""Food matcher with verification process.

Step 1: Find potential match
Step 2: Verify it's actually the same food

Only output number if BOTH steps confirm match.
Otherwise output none.

Verification checklist:
- Same animal/plant source?
- Same part/cut?
- Same processing?
- Would have similar nutrition?

If ANY answer is NO → output none""",
        user_prompt_template="""Match with verification:

SOURCE: {source}

CANDIDATES:
{targets}

First find best candidate, then verify.
Output (number if verified match, otherwise none):""",
        description="Two-step validation process"
    ))
    
    # ========== TEXT-BASED STRATEGIES ==========
    
    strategies.append(PromptStrategy(
        name="text_minimal",
        strategy_id="S11",
        response_format="text",
        system_prompt="Match foods. Reply exact text or 'none' if no match.",
        user_prompt_template="Match: {source}\n\nList:\n{targets}\n\nText (or 'none'):",
        description="Minimal instructions with text response"
    ))
    
    strategies.append(PromptStrategy(
        name="text_simple",
        strategy_id="S12",
        response_format="text",
        system_prompt="Match foods. Return the exact text from the list or 'none'. No other text.",
        user_prompt_template="Match: {source}\n\nTo one of:\n{targets}\n\nReturn exact text or 'none':",
        description="Simple text matching"
    ))
    
    strategies.append(PromptStrategy(
        name="text_detailed",
        strategy_id="S13",
        response_format="text",
        system_prompt="""Food matching expert. Return EXACTLY the matching text from the list or 'none'.
Match based on: ingredient type, processing, preparation.
CRITICAL: Copy the text exactly as shown in the list.""",
        user_prompt_template="SOURCE: {source}\n\nFIND MATCH IN:\n{targets}\n\nExact text from list:",
        description="Detailed criteria with text response"
    ))
    
    strategies.append(PromptStrategy(
        name="text_ruler_fewshot",
        strategy_id="S14",
        response_format="text",
        system_prompt="""Food matching with strict text output.

Return ONLY the exact text from the list, or 'none'.""",
        user_prompt_template="""SOURCE: {source}

TARGET LIST ({list_len} items):
{targets}

Examples:
butter → butter
dragonfruit puree → none

Now match SOURCE. Your answer (exact text or 'none'):""",
        description="Ruler + few-shot guidance for text output"
    ))
    
    strategies.append(PromptStrategy(
        name="text_semantic",
        strategy_id="S15",
        response_format="text",
        system_prompt="Match based on meaning and synonyms. Return the EXACT text from the list or 'none'.",
        user_prompt_template="Find semantic match for: {source}\n\nIn:\n{targets}\n\nExact matching text:",
        description="Semantic text matching"
    ))
    
    strategies.append(PromptStrategy(
        name="text_strict",
        strategy_id="S16",
        response_format="text",
        system_prompt="""CRITICAL MATCHING RULES:
    
DEFAULT TO 'none'. Only match when CERTAIN.

Match ONLY when ALL conditions met:
1. EXACT same base food
2. SAME preparation state
3. SAME form

ALWAYS return 'none' for:
- Different food categories
- Different cooking methods
- Mixed/prepared foods without exact match
- ANY uncertainty

Return EXACT text from list or 'none'. Nothing else.""",
        user_prompt_template="""Item: {source}

Options:
{targets}

Match (exact text or 'none' - default to 'none' when uncertain):""",
        description="Strict text matching with default-to-none bias"
    ))
    
    strategies.append(PromptStrategy(
        name="text_contextual",
        strategy_id="S17",
        response_format="text",
        system_prompt="""You are matching food descriptions. Return the exact text of the best match or 'none'.
Consider: cooking method, preservation, form, and primary ingredient.
IMPORTANT: Your response must be copied exactly from the list.""",
        user_prompt_template="""Food to match: "{source}"

Available options:
{targets}

Instructions:
- Identify the most similar food
- Return its text EXACTLY as shown above
- Return 'none' if no good match exists
- No explanations, just the matching text or 'none'

Your answer:""",
        description="Contextual understanding with text response"
    ))
    
    strategies.append(PromptStrategy(
        name="text_smart_categ",
        strategy_id="S18",
        response_format="text",
        system_prompt="""Expert food matcher using intelligent categorization.

Your process:
1. Analyze the target list to identify natural food categories
2. Determine if source fits any identified category
3. Use category fit to guide matching confidence
4. Output ONLY exact text from list or 'none'

Key insights:
- Items outside target categories likely have no match
- Saying 'none' when appropriate demonstrates expertise
- About 50% of items have no match""",
        user_prompt_template="""Matching task requiring category awareness:

SOURCE: {source}

TARGET LIST:
{targets}

Step-by-step approach:
1. Mentally group targets into natural categories based on what you see
2. Check: Does the source belong to any of these categories?
   - If YES: Search for best match within that category (but 'none' is still valid)
   - If NO: Source is likely outside scope (probably 'none', but still check)
3. Make your decision with appropriate confidence

Remember:
- Correctly saying 'none' is as valuable as identifying matches
- Don't force matches - about half the time 'none' is correct
- If source doesn't fit target categories, 'none' is likely right
- Still attempt matching, but with calibrated skepticism

Answer (exact text or 'none'):""",
        description="Smart categorization with text output"
    ))
    
    strategies.append(PromptStrategy(
        name="text_point_game",
        strategy_id="S19",
        response_format="text",
        system_prompt="""Food matching game - maximize your score!

SCORING RULES:
+1 point: Correct match (selecting the right target)
+1 point: Correct rejection (saying 'none' when no match exists)
-1 point: Wrong match (picking incorrect target)
-1 point: False rejection (saying 'none' when match exists)

Your goal: Maximum points through accurate decisions.
Output: Exact text from list or 'none' only.""",
        user_prompt_template="""FOOD MATCHING GAME - Maximize your score!

SOURCE: {source}

AVAILABLE TARGETS:
{targets}

SCORING SYSTEM:
✓ Find correct match = +1 point
✓ Correctly say 'none' = +1 point
✗ Pick wrong match = -1 point
✗ Say 'none' when match exists = -1 point

STRATEGY TIPS:
- About 50% have matches, 50% don't
- Being right about 'none' scores just as much as identifying matches
- Forcing bad matches costs points
- Trust your judgment - uncertain matches are often 'none'

Make your play (exact text or 'none'):""",
        description="Gamified text matching with balanced scoring"
    ))
    
    strategies.append(PromptStrategy(
        name="text_validation",
        strategy_id="S20",
        response_format="text",
        system_prompt="""Food matcher with verification process.

Step 1: Find potential match
Step 2: Verify it's actually the same food

Only output exact text if BOTH steps confirm match.
Otherwise output 'none'.

Verification checklist:
- Same animal/plant source?
- Same part/cut?
- Same processing?
- Would have similar nutrition?

If ANY answer is NO → output 'none'""",
        user_prompt_template="""Match with verification:

SOURCE: {source}

CANDIDATES:
{targets}

First find best candidate, then verify.
Output (exact text if verified match, otherwise 'none'):""",
        description="Two-step validation with text response"
    ))
    
    return strategies


# ============================= Response Parsing =============================

def parse_response(
    raw_response: str,
    targets: List[str],
    response_format: str
) -> str:
    """Parse model response based on expected format (strict parsing for Gemma 3)."""
    
    cleaned = raw_response.strip()
    
    # Fast path for none/NONE (first 20 chars guard kept)
    if cleaned[:20].lower().strip() == "none" or cleaned.lower().strip() == "none":
        return "none"
    
    if response_format == "number":
        import re
        # Accept ONLY a single token: integer or none
        if re.fullmatch(r"(?:none|[1-9]\d*)", cleaned.strip(), flags=re.IGNORECASE):
            if cleaned.strip().lower() == "none":
                return "none"
            idx = int(cleaned.strip())
            if 1 <= idx <= len(targets):
                return targets[idx - 1]
        # Anything else → none
        return "none"
    
    # text format: exact match only (case-insensitive), no substring heuristics
    lowered = cleaned.lower()
    for t in targets:
        if t.lower() == lowered:
            return t
    return "none"


# ============================= Testing Functions =============================

async def test_single_item(
    item: Dict,
    strategy: PromptStrategy,
    targets: List[str],
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    sem: asyncio.Semaphore
) -> TestResult:
    """Test a single item with a specific prompt strategy."""
    
    async with sem:
        messages = strategy.build_messages(item["source"], targets)
        
        start_time = time.time()
        
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 50 if strategy.response_format == "text" else 10,
                },
                timeout=30.0,
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                raw_response = data["choices"][0]["message"]["content"]
                predicted = parse_response(raw_response, targets, strategy.response_format)
            else:
                raw_response = f"HTTP {response.status_code}"
                predicted = "error"
                
        except Exception as e:
            response_time = time.time() - start_time
            raw_response = str(e)
            predicted = "error"
        
        # Evaluate correctness
        if predicted == "error":
            # Technical error - not a model prediction
            correct = False
            error_type = "technical_error"
        elif item["has_match"]:
            if predicted == item["target"]:
                correct = True
                error_type = None
            elif predicted == "none":
                correct = False
                error_type = "missed_match"
            else:
                correct = False
                error_type = "wrong_match"
        else:
            if predicted == "none":
                correct = True
                error_type = None
            else:
                correct = False
                error_type = "false_positive"
        
        return TestResult(
            source=item["source"],
            expected=item.get("target"),
            predicted=predicted,
            has_match=item["has_match"],
            correct=correct,
            error_type=error_type,
            response_time=response_time,
            raw_response=raw_response[:200]
        )


async def evaluate_strategy(
    strategy: PromptStrategy,
    test_items: List[Dict],
    targets: List[str],
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int
) -> Dict[str, Any]:
    """Evaluate a single prompt strategy."""
    
    print(f"\nTesting: {strategy.name} ({strategy.strategy_id})")
    print(f"  Format: {strategy.response_format}")
    print(f"  Description: {strategy.description}")
    
    sem = asyncio.Semaphore(concurrency)
    
    # Create tasks for all test items
    tasks = [
        test_single_item(item, strategy, targets, client, base_url, api_key, model, sem)
        for item in test_items
    ]
    
    # Run with progress reporting
    results = []
    start_time = time.time()
    strategy_start_time = time.time()  # Track total time for this strategy
    
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(test_items)} ({rate:.1f} items/sec)")
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    strategy_elapsed_time = time.time() - strategy_start_time
    actual_throughput = total / strategy_elapsed_time if strategy_elapsed_time > 0 else 0
    
    # Detailed metrics
    match_results = [r for r in results if r.has_match]
    no_match_results = [r for r in results if not r.has_match]
    
    metrics = {
        "strategy_name": strategy.name,
        "strategy_id": strategy.strategy_id,
        "response_format": strategy.response_format,
        "description": strategy.description,
        "total_items": total,
        "correct": correct,
        "overall_accuracy": correct / total if total > 0 else 0,
        "match_accuracy": sum(1 for r in match_results if r.correct) / len(match_results) if match_results else 0,
        "no_match_accuracy": sum(1 for r in no_match_results if r.correct) / len(no_match_results) if no_match_results else 0,
        "avg_response_time": sum(r.response_time for r in results) / len(results) if results else 0,
        "total_elapsed_time": strategy_elapsed_time,
        "actual_throughput": actual_throughput,
        "errors": {
            "missed_match": sum(1 for r in results if r.error_type == "missed_match"),
            "wrong_match": sum(1 for r in results if r.error_type == "wrong_match"),
            "false_positive": sum(1 for r in results if r.error_type == "false_positive"),
            "technical_error": sum(1 for r in results if r.error_type == "technical_error")
        },
        "raw_results": results
    }
    
    return metrics


# ============================= Main Testing Function =============================

async def run_model_specific_testing(
    sample_size: int,
    base_url: str,
    api_key: str,
    model: str,
    dataset_name: str,
    model_family: str,
    model_size: str,
    data_dir: str,
    concurrency: int
) -> None:
    """Run prompt optimization testing across all strategies."""
    
    print("="*80)
    print("GEMMA 3 PROMPT STRATEGY EVALUATION")
    print("="*80)
    print(f"\nExperiment Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model Family: {model_family}")
    print(f"  Model Size: {model_size}")
    print(f"  Full Model Name: {model}")
    print(f"  Sample Size: {sample_size if sample_size != -1 else 'Full dataset'}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Infrastructure: SCINet Atlas vLLM Farm (6x A100 80GB)")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading data...")
    
    with open(f"{data_dir}/input_desc_list_noquotes.txt", "r") as f:
        all_sources = [line.strip() for line in f if line.strip()]
    
    with open(f"{data_dir}/target_desc_list_noquotes.txt", "r") as f:
        targets = [line.strip() for line in f if line.strip()]
    
    ground_truth_df = pd.read_csv(f"{data_dir}/nhanes_dfg2_labels.csv")
    
    # Create balanced test set
    test_items = []
    
    # Check if using entire dataset
    if sample_size == -1:
        # Use entire dataset
        print("Using entire dataset...")
        
        # Get ALL items with matches
        match_items = ground_truth_df[ground_truth_df['label'] == 1]
        for _, row in match_items.iterrows():
            test_items.append({
                "source": row['ingred_desc'].lower().strip(),
                "target": row['simple_name'].lower().strip(),
                "has_match": True
            })
        
        # Get ALL items without matches
        no_match_items = ground_truth_df[ground_truth_df['label'] == 0]
        for _, row in no_match_items.iterrows():
            test_items.append({
                "source": row['ingred_desc'].lower().strip(),
                "target": None,
                "has_match": False
            })
    else:
        # Use sample size for balanced subset
        # Get items with matches
        match_items = ground_truth_df[ground_truth_df['label'] == 1].sample(
            n=min(sample_size // 2, len(ground_truth_df[ground_truth_df['label'] == 1]))
        )
        for _, row in match_items.iterrows():
            test_items.append({
                "source": row['ingred_desc'].lower().strip(),
                "target": row['simple_name'].lower().strip(),
                "has_match": True
            })
        
        # Get items without matches
        no_match_items = ground_truth_df[ground_truth_df['label'] == 0].sample(
            n=min(sample_size // 2, len(ground_truth_df[ground_truth_df['label'] == 0]))
        )
        for _, row in no_match_items.iterrows():
            test_items.append({
                "source": row['ingred_desc'].lower().strip(),
                "target": None,
                "has_match": False
            })
    
    random.shuffle(test_items)
    
    print(f"\nTest Set Composition:")
    print(f"  Total items: {len(test_items)}")
    print(f"  Items with matches: {sum(1 for t in test_items if t['has_match'])}")
    print(f"  Items without matches: {sum(1 for t in test_items if not t['has_match'])}")
    print(f"  Target list size: {len(targets)} items")
    
    # Get all strategies
    strategies = get_all_prompt_strategies()
    print(f"\nTesting {len(strategies)} prompt strategies...")
    
    # Test each strategy
    all_results = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for strategy in strategies:
            result = await evaluate_strategy(
                strategy,
                test_items,
                targets,
                client,
                base_url,
                api_key,
                model,
                concurrency
            )
            all_results.append(result)
    
    # Analyze results and capture output
    import io
    import sys
    
    # Capture terminal output
    output_buffer = io.StringIO()
    
    # Print to both console and buffer
    class DualWriter:
        def __init__(self, terminal, buffer):
            self.terminal = terminal
            self.buffer = buffer
        def write(self, message):
            self.terminal.write(message)
            self.buffer.write(message)
        def flush(self):
            self.terminal.flush()
            self.buffer.flush()
    
    original_stdout = sys.stdout
    sys.stdout = DualWriter(original_stdout, output_buffer)
    
    print_optimization_report(all_results, model, len(test_items), test_items, 
                            dataset_name, model_family, model_size)
    
    # Restore stdout
    sys.stdout = original_stdout
    terminal_output = output_buffer.getvalue()
    
    # Save detailed results
    save_results(all_results, model, test_items, terminal_output, 
                dataset_name, model_family, model_size)


def print_optimization_report(results: List[Dict], model: str, test_size: int, 
                             test_items: List[Dict], dataset_name: str, 
                             model_family: str, model_size: str):
    """Print optimization analysis report."""
    
    print("\n" + "="*80)
    print("STRATEGY EVALUATION RESULTS")
    print("="*80)
    
    # Sort by overall accuracy
    results.sort(key=lambda x: x["overall_accuracy"], reverse=True)
    
    # 1. Overall Rankings
    print("\n1. OVERALL PERFORMANCE RANKINGS")
    print("-"*80)
    print(f"{'Rank':<5} {'Strategy':<25} {'ID':<6} {'Format':<8} {'Overall':<10} {'Match':<10} {'No-Match':<10}")
    print("-"*80)
    
    for i, r in enumerate(results, 1):  # Show all strategies
        print(f"{i:<5} {r['strategy_name']:<25} ({r['strategy_id']}) {r['response_format']:<8} "
              f"{r['overall_accuracy']:.1%}{'':<5} "
              f"{r['match_accuracy']:.1%}{'':<5} "
              f"{r['no_match_accuracy']:.1%}")
    
    # 2. Format Comparison
    print("\n2. RESPONSE FORMAT COMPARISON")
    print("-"*80)
    
    number_results = [r for r in results if r["response_format"] == "number"]
    text_results = [r for r in results if r["response_format"] == "text"]
    
    if number_results:
        avg_number_acc = sum(r["overall_accuracy"] for r in number_results) / len(number_results)
        avg_number_time = sum(r["avg_response_time"] for r in number_results) / len(number_results)
        avg_number_throughput = sum(r["actual_throughput"] for r in number_results) / len(number_results)
        best_number = max(number_results, key=lambda x: x["overall_accuracy"])
    else:
        avg_number_acc = avg_number_time = avg_number_throughput = 0
        best_number = None
    
    if text_results:
        avg_text_acc = sum(r["overall_accuracy"] for r in text_results) / len(text_results)
        avg_text_time = sum(r["avg_response_time"] for r in text_results) / len(text_results)
        avg_text_throughput = sum(r["actual_throughput"] for r in text_results) / len(text_results)
        best_text = max(text_results, key=lambda x: x["overall_accuracy"])
    else:
        avg_text_acc = avg_text_time = avg_text_throughput = 0
        best_text = None
    
    print(f"Number-based strategies:")
    print(f"  Average accuracy: {avg_number_acc:.1%}")
    print(f"  Average per-item response time: {avg_number_time:.3f}s")
    print(f"  Average throughput: {avg_number_throughput:.1f} items/sec")
    if best_number:
        print(f"  Best performer: {best_number['strategy_name']} ({best_number['strategy_id']}) ({best_number['overall_accuracy']:.1%})")
    
    print(f"\nText-based strategies:")
    print(f"  Average accuracy: {avg_text_acc:.1%}")
    print(f"  Average per-item response time: {avg_text_time:.3f}s")
    print(f"  Average throughput: {avg_text_throughput:.1f} items/sec")
    if best_text:
        print(f"  Best performer: {best_text['strategy_name']} ({best_text['strategy_id']}) ({best_text['overall_accuracy']:.1%})")
    
    # 3. Error Analysis
    print("\n3. ERROR PATTERN ANALYSIS")
    print("-"*80)
    
    # Calculate per-strategy averages instead of totals
    avg_missed = sum(r["errors"]["missed_match"] for r in results) / len(results)
    avg_wrong = sum(r["errors"]["wrong_match"] for r in results) / len(results)
    avg_false = sum(r["errors"]["false_positive"] for r in results) / len(results)
    avg_technical = sum(r["errors"].get("technical_error", 0) for r in results) / len(results)
    
    # Get sample counts for percentage calculation
    match_count = sum(1 for item in test_items if item.get('has_match', False))
    no_match_count = len(test_items) - match_count
    
    print(f"Average error rates per strategy (sample size: {test_size}):")
    print(f"  Missed matches: {avg_missed:.1f} ({avg_missed/match_count*100:.1f}% of items that should match)")
    print(f"  Wrong matches: {avg_wrong:.1f} ({avg_wrong/match_count*100:.1f}% of items that should match)")
    print(f"  False positives: {avg_false:.1f} ({avg_false/no_match_count*100:.1f}% of items that should be 'none')")
    if avg_technical > 0:
        print(f"  Technical errors: {avg_technical:.1f} ({avg_technical/test_size*100:.1f}% of all requests)")
    
    # Identify strategies with lowest error rates
    min_missed = min(results, key=lambda x: x["errors"]["missed_match"])
    min_wrong = min(results, key=lambda x: x["errors"]["wrong_match"])
    min_false = min(results, key=lambda x: x["errors"]["false_positive"])
    min_technical = min(results, key=lambda x: x["errors"].get("technical_error", 0))
    
    print(f"\nBest strategies by error type:")
    print(f"  Fewest missed matches: {min_missed['strategy_name']} ({min_missed['strategy_id']}) ({min_missed['errors']['missed_match']})")
    print(f"  Fewest wrong matches: {min_wrong['strategy_name']} ({min_wrong['strategy_id']}) ({min_wrong['errors']['wrong_match']})")
    print(f"  Fewest false positives: {min_false['strategy_name']} ({min_false['strategy_id']}) ({min_false['errors']['false_positive']})")
    if avg_technical > 0:
        print(f"  Fewest technical errors: {min_technical['strategy_name']} ({min_technical['strategy_id']}) ({min_technical['errors'].get('technical_error', 0)})")
    
    # 4. Performance Characteristics
    print("\n4. PERFORMANCE CHARACTERISTICS")
    print("-"*80)
    
    # Analyze strategy performance characteristics
    high_precision = [r for r in results if r["no_match_accuracy"] > 0.8]
    high_recall = [r for r in results if r["match_accuracy"] > 0.8]
    consistent = [r for r in results if abs(r["match_accuracy"] - r["no_match_accuracy"]) < 0.1]
    
    if high_precision:
        print(f"Conservative strategies (high precision at rejecting non-matches): {len(high_precision)}")
        for r in high_precision[:3]:
            print(f"  - {r['strategy_name']} ({r['strategy_id']}): {r['no_match_accuracy']:.1%}")
    
    if high_recall:
        print(f"\nAggressive strategies (high recall for valid matches): {len(high_recall)}")
        for r in high_recall[:3]:
            print(f"  - {r['strategy_name']} ({r['strategy_id']}): {r['match_accuracy']:.1%}")
    
    if consistent:
        print(f"\nConsistent performance strategies (similar match/no-match accuracy): {len(consistent)}")
        for r in consistent[:3]:
            print(f"  - {r['strategy_name']} ({r['strategy_id']}): Match:{r['match_accuracy']:.1%} NoMatch:{r['no_match_accuracy']:.1%}")
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-"*80)
    
    best_overall = results[0]
    fastest = max(results, key=lambda x: x.get("actual_throughput", 0))  # Highest throughput
    most_conservative = max(results, key=lambda x: x["no_match_accuracy"])
    most_aggressive = max(results, key=lambda x: x["match_accuracy"])
    
    print(f"For {model_family} {model_size}:")
    print(f"\nBest Overall Performance:")
    print(f"  Strategy: {best_overall['strategy_name']} ({best_overall['strategy_id']})")
    print(f"  Format: {best_overall['response_format']}")
    print(f"  Accuracy: {best_overall['overall_accuracy']:.1%}")
    print(f"  Description: {best_overall['description']}")
    
    print(f"\nHighest Throughput:")
    print(f"  Strategy: {fastest['strategy_name']} ({fastest['strategy_id']})")
    print(f"  Throughput: {fastest.get('actual_throughput', 0):.1f} items/sec")
    print(f"  Total time for {test_size} items: {fastest.get('total_elapsed_time', 0):.1f}s")
    print(f"  Average per-item latency: {fastest['avg_response_time']:.3f}s")
    
    print(f"\nMost Conservative (fewer false positives):")
    print(f"  Strategy: {most_conservative['strategy_name']} ({most_conservative['strategy_id']})")
    print(f"  No-match accuracy: {most_conservative['no_match_accuracy']:.1%}")
    
    print(f"\nMost Aggressive (fewer missed matches):")
    print(f"  Strategy: {most_aggressive['strategy_name']} ({most_aggressive['strategy_id']})")
    print(f"  Match accuracy: {most_aggressive['match_accuracy']:.1%}")
    
    # Format preference analysis
    if avg_number_acc > avg_text_acc + 0.05:
        print(f"\nPERFORMANCE OBSERVATION: Number-based prompts achieve {(avg_number_acc - avg_text_acc)*100:.1f}% higher accuracy")
        print(f"  {model_family} {model_size} successfully handles numerical indexing with 256-item target database.")
        print("  Token efficiency gain: ~80% reduction in output tokens vs text-based.")
    elif avg_text_acc > avg_number_acc + 0.05:
        print(f"\nPERFORMANCE OBSERVATION: Text-based prompts achieve {(avg_text_acc - avg_number_acc)*100:.1f}% higher accuracy")
        print(f"  Numerical indexing encounters challenges with 256-item target database.")
        print("  Trade-off: Higher accuracy at cost of increased output tokens.")
    else:
        print(f"\nPERFORMANCE OBSERVATION: Number and text formats show equivalent performance")
        print("  Number format recommended for token efficiency (80% reduction).")
        print("  Both formats viable for this model and database size.")


def save_results(results: List[Dict], model: str, test_items: List[Dict], 
                terminal_output: str, dataset_name: str, model_family: str, model_size: str):
    """Save optimization results to organized folders with experiment naming."""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create experiment-specific results folder
    experiment_name = f"{dataset_name}_{model_family}_{model_size}"
    results_dir = f"/90daydata/lemay_diet_guthealth/richard.stoker/testapi-a100/projects/SCINet_Gemma3_experiments/{dataset_name}/results/{experiment_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    base_filename = os.path.join(results_dir, f"{experiment_name}_{timestamp}")
    
    # Get sample size for filename
    sample_size = len(test_items)
    
    # Save full results JSON with experiment metadata
    full_results = {
        "experiment": {
            "dataset": dataset_name,
            "model_family": model_family,
            "model_size": model_size,
            "full_model": model,
            "infrastructure": "SCINet Atlas vLLM Farm (6x A100 80GB)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "test_configuration": {
            "test_size": len(test_items),
            "with_matches": sum(1 for t in test_items if t["has_match"]),
            "without_matches": sum(1 for t in test_items if not t["has_match"]),
            "strategies_tested": len(results)
        },
        "results": [
            {
                "strategy_name": r["strategy_name"],
                "strategy_id": r.get("strategy_id", ""),
                "response_format": r["response_format"],
                "description": r["description"],
                "overall_accuracy": r["overall_accuracy"],
                "match_accuracy": r["match_accuracy"],
                "no_match_accuracy": r["no_match_accuracy"],
                "avg_response_time": r["avg_response_time"],
                "actual_throughput": r.get("actual_throughput", 0),
                "total_elapsed_time": r.get("total_elapsed_time", 0),
                "errors": r["errors"]
            }
            for r in results
        ]
    }
    
    with open(f"{base_filename}.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    # Save summary CSV for easy analysis
    summary_data = []
    for r in results:
        summary_data.append({
            "experiment": experiment_name,
            "strategy": r["strategy_name"],
            "strategy_id": r.get("strategy_id", ""),
            "format": r["response_format"],
            "overall_acc": f"{r['overall_accuracy']:.3f}",
            "match_acc": f"{r['match_accuracy']:.3f}",
            "no_match_acc": f"{r['no_match_accuracy']:.3f}",
            "avg_time": f"{r['avg_response_time']:.3f}",
            "throughput": f"{r.get('actual_throughput', 0):.2f}",
            "missed_matches": r["errors"]["missed_match"],
            "wrong_matches": r["errors"]["wrong_match"],
            "false_positives": r["errors"]["false_positive"],
            "technical_errors": r["errors"].get("technical_error", 0)
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f"{base_filename}_summary.csv", index=False)
    
    # Save terminal output if provided
    if terminal_output:
        with open(f"{base_filename}_report.txt", "w") as f:
            f.write(terminal_output)
    
    # Create detailed CSV with all predictions for each item
    detailed_rows = []
    
    for item in test_items:
        row = {
            "experiment": experiment_name,
            "source_item": item["source"],
            "expected_match": item.get("target", "none") if item["has_match"] else "none",
            "has_match": item["has_match"],
            "model": model,
            "sample_size": sample_size
        }
        
        # Add predictions from each strategy
        for strategy_result in results:
            strategy_id = strategy_result.get("strategy_id", "")
            strategy_name = strategy_result["strategy_name"]
            
            # Find this item's result in the strategy's raw results
            item_result = None
            if "raw_results" in strategy_result:
                for raw_result in strategy_result["raw_results"]:
                    if raw_result.source == item["source"]:
                        item_result = raw_result
                        break
            
            if item_result:
                # Determine error classification
                error_type = ""
                is_correct = item_result.correct
                if not is_correct:
                    error_type = item_result.error_type or ""
                
                # Add strategy-specific columns
                row[f"{strategy_id}_predicted"] = item_result.predicted
                row[f"{strategy_id}_correct"] = is_correct
                row[f"{strategy_id}_error_type"] = error_type
                row[f"{strategy_id}_response_time"] = f"{item_result.response_time:.3f}"
        
        detailed_rows.append(row)
    
    # Save detailed CSV
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv_path = f"{base_filename}_detailed_matches.csv"
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    # Create strategy performance matrix
    matrix_rows = []
    for strategy_result in results:
        matrix_row = {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "model_family": model_family,
            "model_size": model_size,
            "strategy_id": strategy_result.get("strategy_id", ""),
            "strategy_name": strategy_result["strategy_name"],
            "response_format": strategy_result["response_format"],
            "overall_accuracy": f"{strategy_result['overall_accuracy']:.3f}",
            "match_accuracy": f"{strategy_result['match_accuracy']:.3f}",
            "no_match_accuracy": f"{strategy_result['no_match_accuracy']:.3f}",
            "avg_response_time": f"{strategy_result['avg_response_time']:.3f}",
            "throughput": f"{strategy_result.get('actual_throughput', 0):.2f}",
            "missed_matches": strategy_result["errors"]["missed_match"],
            "wrong_matches": strategy_result["errors"]["wrong_match"],
            "false_positives": strategy_result["errors"]["false_positive"],
            "technical_errors": strategy_result["errors"].get("technical_error", 0),
            "sample_size": sample_size
        }
        matrix_rows.append(matrix_row)
    
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_csv_path = f"{base_filename}_strategy_matrix.csv"
    matrix_df.to_csv(matrix_csv_path, index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - {experiment_name}_{timestamp}.json (full results)")
    print(f"  - {experiment_name}_{timestamp}_summary.csv (summary table)")
    print(f"  - {experiment_name}_{timestamp}_detailed_matches.csv (all predictions)")
    print(f"  - {experiment_name}_{timestamp}_strategy_matrix.csv (strategy performance matrix)")
    if terminal_output:
        print(f"  - {experiment_name}_{timestamp}_report.txt (terminal output)")


# ============================= CLI Interface =============================

def main():
    """Main entry point."""
    
    env_base = os.environ.get("BASE_URL") or os.environ.get("ROUTER_URL") or "http://localhost:8000/v1"
    env_key = os.environ.get("API_KEY") or os.environ.get("LITELLM_MASTER_KEY") or ""
    
    parser = argparse.ArgumentParser(
        description="Gemma 3 prompt strategy evaluation for food database matching",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Experiment naming parameters
    parser.add_argument(
        "--dataset",
        default="NHANES",
        help="Dataset name for experiment (default: NHANES)"
    )
    parser.add_argument(
        "--model-family",
        default="Gemma3",
        help="Model family name (default: Gemma3)"
    )
    parser.add_argument(
        "--model-size",
        default="27b",
        help="Model size variant (e.g., 4b, 12b, 27b)"
    )
    
    # Testing parameters
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of items to test (default: 100, use -1 for entire dataset)"
    )
    parser.add_argument(
        "--model",
        default="gemma3-27b",
        help="Full model name for API (default: gemma3-27b)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Number of parallel requests (default: 64, typically 64-96 for vLLM farm)"
    )
    parser.add_argument(
        "--data-dir",
        default="/90daydata/lemay_diet_guthealth/richard.stoker/testapi-a100/projects/SCINet_Gemma3_experiments/NHANES/data",
        help="Path to data directory"
    )
    parser.add_argument("--base-url", default=env_base, help="API base URL")
    parser.add_argument("--api-key", default=env_key, help="API key")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: No API key provided. Set LITELLM_MASTER_KEY or use --api-key")
        return 1
    
    asyncio.run(run_model_specific_testing(
        args.sample_size,
        args.base_url,
        args.api_key,
        args.model,
        args.dataset,
        args.model_family,
        args.model_size,
        args.data_dir,
        args.concurrency
    ))
    
    return 0


if __name__ == "__main__":
    exit(main())