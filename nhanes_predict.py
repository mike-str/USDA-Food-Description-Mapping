from util import load_nhanes, load_asa
from util import clean_text, save_results
from util import remove_dupe_and_na, remove_rows_where_columns_match, incorrect_matches
from util_metrics import compute_metrics
from string_matcher import match
import pandas as pd

if __name__ == "__main__":
    # df = load_asa()
    df = load_nhanes()
    nhanes_targets = df["target"].to_list()

    df["index"] = [i for i in range(len(df))]

    df["input_clean"] = clean_text(df["input"])
    df["target_clean"] = clean_text(df["target"])
    nhanes_targets_clean = clean_text(df["target_clean"])

    arr_1 = list(df["input_clean"])
    arr_2 = nhanes_targets_clean

    df_results = match(arr_1, arr_2, method="fuzzy")
    df_results2 = match(arr_1, arr_2, method="tfidf")
    df_results3 = match(df["input"].to_list(), nhanes_targets, method="embed")

    merged_df = pd.merge(df, df_results, on="index", how="left")
    merged_df = pd.merge(merged_df, df_results2, on="index", how="left")
    merged_df = pd.merge(merged_df, df_results3, on="index", how="left")

    print(merged_df.columns)

    res_df = compute_metrics(merged_df)
    print(res_df)

    save_results(merged_df, "output_ASA24.csv")

    incorrect_matches_df = incorrect_matches(merged_df, method="embed")[["input", "target", "match_embed", "score_embedding"]]
    print(len(incorrect_matches_df))
    print(len(incorrect_matches_df[incorrect_matches_df["input"] == incorrect_matches_df["match_embed"]]))
    incorrect_matches_df.to_csv("results/incorrect_predictions_nhanes.csv", index=False)