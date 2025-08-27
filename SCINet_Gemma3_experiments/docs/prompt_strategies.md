# How the Prompt Strategies Work - Basic Guide

## How Gemma 3 Handles Prompts (Technical Setup)

Gemma 3 is different from other models like Claude or GPT-4o. Here's what makes it unique:

1. **NO SYSTEM PROMPTS**: Most models let you set a "system" role that gives background instructions. Gemma 3 doesn't support this - everything has to go in one user message.

2. **MERGED INSTRUCTIONS**: Because of this limitation, we take what would normally be three parts (system instructions, ruler info, user question) and combine them into one message that Gemma sees as coming from the user.

3. **STRICT OUTPUT PARSING**: Gemma 3 tends to be less consistent with output formats than other models, so we have to be very explicit about what we want back.

4. **CONTEXT WINDOW**: We need MAXLEN=8192 to handle the 256-item target list. Without this, Gemma starts producing garbage when the list gets too long.

The actual prompt structure looks like:
```
[Ruler Info: "N = 256, Return ONLY one token..."]
[System-like instructions merged here]
[The actual matching task]
```

## The 20 Strategies Explained

### Number Strategies (S01-S10) - These ask for just a number back

#### S01 - NUMBER MINIMAL
- **What it does**: Bare bones approach. Just says "match this food, give me a number."
- **How restrictive**: Not at all. Gemma decides everything.
- **When it might work**: Simple, obvious matches where there's no ambiguity.
- **The prompt basically says**: "Here's a food, here's a list, pick a number or say none."

#### S02 - NUMBER SIMPLE
- **What it does**: Adds a bit more guidance - tells Gemma to pick the BEST match and mentions that saying "none" is okay if nothing fits well (using a 50% threshold).
- **How restrictive**: Slightly more than S01, but still pretty open.
- **When it might work**: When you want Gemma to be a bit more thoughtful but not overthink.
- **The prompt basically says**: "Find the best match, but if nothing's really similar (less than 50% match), just say none."

#### S03 - NUMBER DETAILED
- **What it does**: Gives Gemma specific things to look for - same food type, how it's processed (raw vs cooked), and its form (whole vs sliced).
- **How restrictive**: Moderate - provides criteria but doesn't force strict rules.
- **When it might work**: Complex foods where processing matters (like "roasted chicken" vs "fried chicken").
- **The prompt basically says**: "Consider these three factors when matching."

#### S04 - NUMBER RULER FEWSHOT
- **What it does**: Shows Gemma examples of how to respond. Uses "ruler" technique (showing the list length) and gives sample answers.
- **How restrictive**: Moderate - the examples guide behavior.
- **When it might work**: Helps Gemma understand the format you want.
- **The prompt basically says**: "Here's how others answered: butter → 2, dragonfruit → none. Now you try."

#### S05 - NUMBER STRICT
- **What it does**: Super strict rules. Default answer is "none" unless very specific conditions are met. Must be exact same food, same cooking, same form.
- **How restrictive**: VERY - this is the strictest number strategy.
- **When it might work**: When you only want extremely confident matches.
- **The prompt basically says**: "Say none unless you're absolutely certain. If beef isn't cooked the same way, it's none. If it's ground vs whole, it's none."

#### S06 - NUMBER SEMANTIC
- **What it does**: Focuses on meaning and synonyms rather than exact words.
- **How restrictive**: Low - allows flexible matching based on meaning.
- **When it might work**: When items might be described differently but mean the same thing.
- **The prompt basically says**: "Look for similar meanings, not just exact words."

#### S07 - NUMBER CONTEXTUAL
- **What it does**: Considers the full context - cooking method, how it's preserved, its form, main ingredients. More thoughtful than S02.
- **How restrictive**: Moderate - gives things to consider but doesn't mandate rules.
- **When it might work**: Complex prepared foods with multiple components.
- **The prompt basically says**: "Think about all aspects of the food before deciding."

#### S08 - NUMBER SMART CATEG
- **What it does**: Tells Gemma to first figure out what categories are in the target list, then see if the source food fits any category. Smart approach to the 50% no-match problem.
- **How restrictive**: Moderate but intelligent - guides thinking without hard rules.
- **When it might work**: When the target list has clear food groups.
- **The prompt basically says**: "First, mentally group the targets. Then ask: does my source food belong to any of these groups? If not, it's probably none."

#### S09 - NUMBER POINT GAME
- **What it does**: Turns it into a game where correct matches AND correct "none" answers both score points. Wrong answers lose points.
- **How restrictive**: Moderate - the scoring system encourages careful decisions.
- **When it might work**: Might trigger better decision-making through gamification.
- **The prompt basically says**: "You get points for being right, whether that's finding a match OR correctly saying none. Play to win!"

#### S10 - NUMBER VALIDATION
- **What it does**: Two-step process - first find a candidate, then verify it's actually the same food using a checklist.
- **How restrictive**: High - must pass verification checklist.
- **When it might work**: Reduces false positives by adding a verification step.
- **The prompt basically says**: "Find your best guess, then check: same source? same cut? same processing? same nutrition? If any answer is NO, say none."

### Text Strategies (S11-S20) - These ask for the actual food text back

#### S11 - TEXT MINIMAL
- **What it does**: Bare bones, just like S01 but returns text instead of numbers.
- **How restrictive**: Not at all - Gemma decides everything.
- **When it might work**: Simple matches, no ambiguity.
- **The prompt basically says**: "Match this food, return the exact text or none."

#### S12 - TEXT SIMPLE
- **What it does**: Like S02 but returns text. Basic instructions, mentions "none" is okay.
- **How restrictive**: Low - just asks for exact text, nothing more.
- **When it might work**: Straightforward matching tasks.
- **The prompt basically says**: "Find the match, return its exact text, or say none."

#### S13 - TEXT DETAILED
- **What it does**: Like S03 but returns text. Considers ingredient type, processing, preparation.
- **How restrictive**: Moderate - gives criteria to consider.
- **When it might work**: When details matter for the match.
- **The prompt basically says**: "Think about these factors, then return the exact matching text."

#### S14 - TEXT RULER FEWSHOT
- **What it does**: Like S04 but returns text. Shows examples of good responses.
- **How restrictive**: Moderate - examples guide behavior.
- **When it might work**: Helps Gemma understand expected format.
- **The prompt basically says**: "Here's how to answer: butter → 'butter', dragonfruit → 'none'."

#### S15 - TEXT SEMANTIC
- **What it does**: Like S06 but returns text. Focuses on meaning over exact words.
- **How restrictive**: Low - flexible meaning-based matching.
- **When it might work**: When descriptions vary but mean the same thing.
- **The prompt basically says**: "Match based on what it means, return the exact text."

#### S16 - TEXT STRICT
- **What it does**: Like S05 but returns text. VERY strict - defaults to "none" unless certain.
- **How restrictive**: VERY - the strictest text strategy.
- **When it might work**: Only want very confident matches.
- **The prompt basically says**: "Default to none. Only match if EVERYTHING lines up perfectly."

#### S17 - TEXT CONTEXTUAL
- **What it does**: Like S07 but returns text. Considers full context of the food.
- **How restrictive**: Moderate - thoughtful but not rigid.
- **When it might work**: Complex foods needing thoughtful matching.
- **The prompt basically says**: "Consider all aspects, then return the matching text."

#### S18 - TEXT SMART CATEG
- **What it does**: Like S08 but returns text. Smart categorization approach.
- **How restrictive**: Moderate but intelligent - guides without forcing.
- **When it might work**: When targets have natural groupings.
- **The prompt basically says**: "Figure out the categories in the list. If the source doesn't fit any category, it's probably none. But still check to be sure."

#### S19 - TEXT POINT GAME
- **What it does**: Like S09 but returns text. Gamification with scoring.
- **How restrictive**: Moderate - scoring encourages good decisions.
- **When it might work**: Game mechanics might improve accuracy.
- **The prompt basically says**: "Score points by being right, whether matching or saying none."

#### S20 - TEXT VALIDATION
- **What it does**: Like S10 but returns text. Two-step verification process.
- **How restrictive**: High - must pass the checklist.
- **When it might work**: When you need high confidence in matches.
- **The prompt basically says**: "Find a candidate, verify it thoroughly, only return it if it passes all checks."

## Quick Reference - Restrictiveness Scale

### LEAST RESTRICTIVE (will try to match almost anything)
- **S01/S11**: Minimal - No rules at all
- **S06/S15**: Semantic - Flexible meaning-based

### MODERATELY RESTRICTIVE (balanced approach)
- **S02/S12**: Simple - Basic guidance
- **S03/S13**: Detailed - Considers specific factors
- **S07/S17**: Contextual - Thoughtful but flexible
- **S08/S18**: Smart Categ - Intelligent categorization
- **S09/S19**: Point Game - Scoring system guides decisions

### HIGHLY RESTRICTIVE (will say "none" more often)
- **S04/S14**: Ruler Fewshot - Examples constrain behavior
- **S10/S20**: Validation - Must pass checklist
- **S05/S16**: Strict - Defaults to "none" unless perfect match
