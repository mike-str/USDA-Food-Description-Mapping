# USDA Food Description Matching

This repository contains two experiments that evaluate string-matching methods for the purpose of linking input strings to their most appropriate match in a predefined set of target matches.

- **Experiment ASA24** (`asa24/asa24.py`)

  - Assumes **every** input description *does* have a valid match in the target set.
  - Goal: Return the single best match for each input description..

- **Experiment NHANES** (`nhanes/nhanes.py`)

  - Some input descriptions may have **no suitable match**.
  - Goal: Return the best match if one exists; otherwise, flag the input as `NO MATCH` if similarity score falls below a predefined threshold.

---

## Data
- **ASA24**<br>
***File***: `ASA24_FooDB_codematches_6-26-2025.xlsx`<br>
This dataset assumes that every input description has a corresponding match in the target set.

<br>

- **NHANES**<br>
***File***: `nhanes_dfg2_labels.csv`<br>
In this dataset, each ingredient description (ingred_desc) derived from the dietary data was labeled as either having a valid match (label = 1) or not (label = 0) to the food description (simple_name) in the Davis Food Glycopedia 2.0. Matches were considered to exist if there was a good match for the exact food **or** for a food with similar carbohydrate content, which served as a proxy when no direct match was available.

**Data provenance**   > *[add citation / download URL here]*

---

## Matching Algorithms (`matching_algorithms/`)

| Method         | File              | Description                                                                                                                                                                                                                   |
|----------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fuzzy          | `fuzzy_match.py` | Selects the best match based on fuzzy string similarity using the RapidFuzz library. For each input, the algorithm chooses the closest target string according to `fuzz.ratio`. Text is cleaned (lowercased, punctuation removed, whitespace collapsed) before matching. |
| TF‑IDF         | `tfidf_match.py` | Selects the best match by computing cosine similarity between TF‑IDF vectors of cleaned input and target descriptions. Text is cleaned before vectorization (lowercased, punctuation removed, and whitespace collapsed).                                                                                                                              |
| Embedding      | `embed_match.py` | Selects the best match by computing cosine similarity between dense embeddings generated using `SentenceTransformer("thenlper/gte-large")`. Input and target texts are used **without** cleaning or preprocessing applied.                                                                 |
| ChatGPT‑o3     | —                | Selects the best match by prompting GPT‑o3 directly with the input and list of possible targets. Prompt file: `llm/prompts/o3_asa24_prompt.txt`. Inputs: `data/llm/asa24_input.csv` and `data/llm/asa24_unique_target.csv`. The model returned a JSON file of predicted matches, which was then evaluated within the respective experiment script.                                                                 |
