from util import load_nhanes
from util import embed_match_top_n, compute_top_k_accuracy
from util import generate_prompt, prompt_llm
import pandas as pd
import json
import pickle

if __name__ == "__main__":
    df = load_nhanes()
    df = df[df["label"] == 1]

    # Load responses from pickle
    with open("responses.pkl", "rb") as f:
        responses_arr = pickle.load(f)

    # Convert to DataFrame
    responses_df = pd.DataFrame(responses_arr)
    responses_df["input_desc"] = responses_df["input_desc"].str.strip().str.lower()

    # Left join on 'input_desc'
    merged_df = df.merge(responses_df, on="input_desc", how="left")

    # Example: add a correctness flag
    merged_df["correct"] = (
        merged_df["target_desc"].str.strip().str.lower() ==
        merged_df["predicted_target"].fillna("").str.strip().str.lower()
    )

    # Print stats
    accuracy = merged_df["correct"].mean()
    print(f"Accuracy (joined): {accuracy:.2%}")

    # Optional: save to CSV
    merged_df.to_csv("nhanes_joined_predictions.csv", index=False)

    