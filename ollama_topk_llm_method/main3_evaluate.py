from util import load_nhanes
from util import embed_match_top_n, compute_top_k_accuracy
from util import generate_prompt, prompt_llm
import pandas as pd
import json
import pickle

def load_asa():
    df = pd.read_excel("data_asa/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description_uncleaned", "orig_food_common_name_uncleaned"]]
    df.columns = ["input_desc", "target_desc"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    # remove row if input or target is NA
    df = df.dropna(subset=["input_desc", "target_desc"])

    # if the input exists twice, it should map to the same target - no reason to keep this
    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

if __name__ == "__main__":
    df = load_asa()

    # Load responses from pickle
    with open("asa_responses.pkl", "rb") as f:
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
    merged_df.to_csv("asa_joined_predictions.csv", index=False)

    