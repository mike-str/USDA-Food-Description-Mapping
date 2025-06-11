import pandas as pd
from util import clean_text, compute_metrics, save_results
from util import remove_dupe_and_na, remove_rows_where_columns_match, incorrect_matches
from string_matcher import match

if __name__ == "__main__":
    df = pd.read_excel("data/ASA24_FooDB_codematches.xlsx", sheet_name=0)
    df = df[["Ingredient_description", "orig_food_common_name"]]
    df.columns = ["input", "target"]

    df = remove_dupe_and_na(df)
    df = remove_rows_where_columns_match(df, "input", "target")
    print(len(df))

    df["index"] = [i for i in range(len(df))]

    df["input_clean"] = clean_text(df["input"])
    df["target_clean"] = clean_text(df["target"])

    arr_1 = list(df["input_clean"])
    arr_2 = list(df["target_clean"])

    df_results = match(arr_1, arr_2, method="fuzzy")
    df_results2 = match(arr_1, arr_2, method="tfidf")

    merged_df = pd.merge(df, df_results, on="index", how="left")
    merged_df = pd.merge(merged_df, df_results2, on="index", how="left")

    res_df = compute_metrics(merged_df)
    print(res_df)

    save_results(merged_df, "output_ASA24.csv")

    incorrect_matches(merged_df, method="fuzzy").to_csv("results/incorrect_matches.csv", index=False)