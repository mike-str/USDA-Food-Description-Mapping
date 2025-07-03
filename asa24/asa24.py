from .util import load_asa
from .util import clean_text
from .util import remove_dupe_and_na, remove_rows_where_columns_match, incorrect_matches
from .util import compute_accuracy
from matching_algorithms.fuzzy_match import fuzzy_match
from matching_algorithms.tfidf_match import tfidf_match
from matching_algorithms.embed_match import embed_match
import pandas as pd
import numpy as np
import os

def run_asa():
    df = load_asa()

    df["index"] = [i for i in range(len(df))]

    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    df["input_desc_clean"]  = clean_text(input_desc_list)
    df["target_desc_clean"] = clean_text(target_desc_list)
    input_desc_clean_list, target_desc_clean_list = df["input_desc_clean"].to_list(), df["target_desc_clean"].to_list()

    # 
    clean_to_raw_target_dict = dict()
    for i, target_desc_clean in enumerate(df["target_desc_clean"]):
        """
        unfortunately I have to do this because when cleaning the input / target descriptions I noticed
        that some of them will be different before cleaning, and then be the same after cleaning.

        this can lead to inflated accuracy
        """
        if target_desc_clean not in clean_to_raw_target_dict:
            clean_to_raw_target_dict[target_desc_clean] = target_desc_list[i]

    df_fuzzy = fuzzy_match(input_desc_clean_list, target_desc_clean_list)
    df_fuzzy["match_fuzzy"] = df_fuzzy["match_fuzzy"].map(clean_to_raw_target_dict)

    df_tfidf = tfidf_match(input_desc_clean_list, target_desc_clean_list)
    df_tfidf["match_tfidf"] = df_tfidf["match_tfidf"].map(clean_to_raw_target_dict)

    df_embed = embed_match(input_desc_list, target_desc_list)

    df = df.join(df_fuzzy, on="index", how="left")
    df = df.join(df_tfidf, on="index", how="left")
    df = df.join(df_embed, on="index", how="left")

    # results
    acc_fuzzy = compute_accuracy(df, "fuzzy")
    acc_tfidf = compute_accuracy(df, "tfidf")
    acc_embed = compute_accuracy(df, "embed")

    print(f"Fuzzy Match Accuracy: {acc_fuzzy:.2f}")
    print(f"TF-IDF Match Accuracy: {acc_tfidf:.2f}")
    print(f"Embed Match Accuracy: {acc_embed:.2f}")

    output_lines = [
        f"Fuzzy Match Accuracy: {acc_fuzzy:.2f}",
        f"TF-IDF Match Accuracy: {acc_tfidf:.2f}",
        f"Embed Match Accuracy: {acc_embed:.2f}",
    ]
    output_text = "\n".join(output_lines) + "\n"

    os.makedirs("results", exist_ok=True)
    with open("results/asa24_results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    print(output_text)

    # this can be all possile targets in the original database - or it can be just the target matches in the provided dataframe
    # candidate_matches = list(set(asa_df["target"]))
    # candidate_matches_clean = clean_text(candidate_matches)

    # num_correct = (asa_df["input_id"] == asa_df["target_id"]).sum()
    # print(num_correct)
    # print(num_correct / len(asa_df))

    # df = asa_df[asa_df["input_id"] != asa_df["target_id"]]
    # df["index"] = [i for i in range(len(df))]

    # input_desc = df["input"].to_list()

    # print(f"len_df: {len(df)}")
    # df_results = embed_match(input_desc, candidate_matches)
    # df_results["index"] = [i for i in range(len(df_results))]
    # merged_df = pd.merge(df, df_results, on="index", how="left")

    # print((merged_df["target"] == merged_df["match_embed"]).sum())
    # num_correct += (merged_df["target"] == merged_df["match_embed"]).sum()

    # print(f"len_df: {len(merged_df)}")
    # print(num_correct / len(asa_df))

  
    # incorrect_matches_df = merged_df[merged_df["target"] != merged_df["match_embed"]]
    # del incorrect_matches_df["index"]
    # incorrect_matches_df.to_csv("results/incorrect_predictions_asa24.csv", index=False)