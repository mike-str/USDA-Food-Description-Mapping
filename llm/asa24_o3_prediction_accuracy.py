import pandas as pd

if __name__ == "__main__":
    # data load and preprocess
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description", "orig_food_common_name", "Ingredient_code", "orig_food_id"]]
    df.columns = ["input_desc", "target_desc", "input_id", "target_id"]

    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    df = df.dropna(subset=["input_desc", "target_desc"])
    df = df.drop_duplicates(subset="input_desc", keep="first")

    # checking accuracy
    df_predictions = pd.read_json("data/llm/asa24_o3_predicted_matches.json")

    df_merged = df.merge(df_predictions, on="input_desc", how="left")
    df_merged["correct"] = df_merged["target_desc"] == df_merged["predicted_target"]

    accuracy = df_merged["correct"].mean()
    print(f"Matching Accuracy: {accuracy:.2%}")