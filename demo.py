import pandas as pd
from util import clean_text, compute_metrics, save_results, remove_dupe_and_na
from string_matcher import match

if __name__ == "__main__":
    arr_1 = ["apple pie", "banana bread", "cheddar", "cherry pie"]
    arr2 = ["apple", "banana muffin", "cheddar cheese", "banana bre"]
    zipped = list(zip(arr_1, arr2))

    df = pd.DataFrame(zipped, columns=["arr_1", "arr2"])
    df.columns = ["ingredient_description", "orig_food_common_name"]

    df = remove_dupe_and_na(df)

    df["index"] = [i for i in range(len(df))]

    df["orig_food_common_name_clean"] = clean_text(df["orig_food_common_name"])
    df["ingredient_description_clean"] = clean_text(df["ingredient_description"])

    arr_1 = list(df["orig_food_common_name_clean"])
    arr_2 = list(df["ingredient_description_clean"])

    df_results = match(arr_1, arr_2, method="fuzzy")
    df_results2 = match(arr_1, arr_2, method="tfidf")

    merged_df = pd.merge(df, df_results, on="index", how="left")
    merged_df = pd.merge(merged_df, df_results2, on="index", how="left")

    res_df = compute_metrics(merged_df)
    print(res_df)