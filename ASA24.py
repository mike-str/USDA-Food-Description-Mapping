from util import load_asa
from util import clean_text, save_results
from util import remove_dupe_and_na, remove_rows_where_columns_match, incorrect_matches
from util_metrics import compute_metrics
from string_matcher import match
import pandas as pd
import numpy as np

if __name__ == "__main__":
    asa_df, id_dict = load_asa()

    # this can be all possile targets in the original database - or it can be just the target matches in the provided dataframe
    candidate_matches = list(set(asa_df["target"]))
    candidate_matches_clean = clean_text(candidate_matches)

    num_correct = (asa_df["input_id"] == asa_df["target_id"]).sum()
    print(num_correct)
    print(num_correct / len(asa_df))

    df = asa_df[asa_df["input_id"] != asa_df["target_id"]]
    df["index"] = [i for i in range(len(df))]

    input_desc = df["input"].to_list()

    print(f"len_df: {len(df)}")
    df_results = match(input_desc, candidate_matches, method="embed")
    merged_df = pd.merge(df, df_results, on="index", how="left")

    print((merged_df["target"] == merged_df["match_embed"]).sum())
    num_correct += (merged_df["target"] == merged_df["match_embed"]).sum()

    print(f"len_df: {len(merged_df)}")
    print(num_correct / len(asa_df))

  
    incorrect_matches_df = merged_df[merged_df["target"] != merged_df["match_embed"]]
    del incorrect_matches_df["index"]
    incorrect_matches_df.to_csv("results/incorrect_predictions_asa24.csv", index=False)