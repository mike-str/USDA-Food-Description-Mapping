from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import ollama

def load_nhanes():
    df = pd.read_csv("data/nhanes_dfg2_labels.csv")

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input_desc", "target_desc", "label"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

def embed_match_top_n(input_list, target_list, n=5):
    assert isinstance(input_list, list)
    assert isinstance(target_list, list)

    model = SentenceTransformer("thenlper/gte-large")
    
    input_vecs = model.encode(input_list, normalize_embeddings=True)
    target_vecs = model.encode(target_list, normalize_embeddings=True)

    sim_matrix = cosine_similarity(input_vecs, target_vecs)

    results = [
        {
            "input_desc": input_list[i],
            # Get indices of top-n highest scores (sorted descending)
            "top_matches": [{"target": target_list[j], "score": row[j]} for j in row.argsort()[-n:][::-1]]
        }
        for i, row in enumerate(sim_matrix)
    ]

    return results

def compute_top_k_accuracy(df, top_k_matches, k=5):
    correct = 0
    total = len(df)
    wrong_matches = []

    input_to_true_target = dict(zip(df["input_desc"], df["target_desc"]))

    for match in top_k_matches:
        input_desc = match["input_desc"]
        true_target = input_to_true_target.get(input_desc)
        predicted_targets = [m["target"] for m in match["top_matches"][:k]]
        
        if true_target in predicted_targets:
            correct += 1
        else:
            wrong_matches.append((input_desc, true_target))

    accuracy = correct / total if total > 0 else 0.0

    return accuracy


"""
### Task
Choose the **single best match** for the given input food description, from the list of candidates provided.

### Constraints
- You **must** return a JSON object with two fields: `"input_desc"` and `"predicted_target"`.
- `"predicted_target"` **must exactly match one of the provided candidates** — do **not** paraphrase, reword, or create anything new.
- You **must not** return an empty object, "none", or "no match".
- **No explanation. No commentary. Just the JSON.**

### JSON Format
"""

def generate_prompt(input_description: str, target_descriptions: list[str]) -> str:
    # targets = "\n".join([f"- {desc}" for desc in target_descriptions])
    targets = "\n".join([f"- {desc}" for desc in target_descriptions])

    prompt = f"""
### Task
Choose the **single best match** for the given input food description, from the list of candidates provided.

### Constraints
- You **must** return a JSON object with two fields: `"input_desc"` and `"predicted_target"`.
- `"predicted_target"` **must exactly match one of the provided candidates** — do **not** paraphrase, reword, or create anything new.
- You **must not** return an empty object, "none", or "no match".
- **No explanation. No commentary. Just the JSON.**

### JSON Format
```json
{{ 
    "input_desc": "{input_description}",
    "predicted_target": "<string>" 
}}
```

### Input
- {input_description}

### Candidates
{targets}
""".strip()

    return prompt

def prompt_llm(prompt, system_message=""):
    response = ollama.chat(
        model="qwen3:14b",
        messages=[
            {"role": "system", 
             "content": system_message},
            {"role": "user", 
             "content": prompt}
        ],
        format="json",
        stream=False,
        options={
            "temperature": 1.0,
            "num_ctx": 2048,
            "num_predict": 256,
        },
        keep_alive="1m"
    )

    content = response["message"]["content"]
    try:
        json_obj = json.loads(content)
        return json_obj
    except json.JSONDecodeError as e:
        # print("JSON parsing failed:", e)
        # print("Raw response:\n", content)
        # return {"input_desc": "", "predicted_target": ""}
        # not sure if i want to return empty dict / json
        return {}

    return json_obj