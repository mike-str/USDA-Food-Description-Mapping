import pandas as pd
import glob

if __name__ == "__main__":
    # data load and preprocess
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description", "orig_food_common_name", "Ingredient_code", "orig_food_id"]]
    df.columns = ["input_desc", "target_desc", "input_id", "target_id"]

    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    df = df.dropna(subset=["input_desc", "target_desc"])
    df = df.drop_duplicates(subset="input_desc", keep="first")
    print(f"len original file {len(df)}")

    # loading predictions
    file_paths = glob.glob("data/llm/claude_predictions/*.txt")

    dfs = [pd.read_csv(path, header=None, sep="\t") for path in file_paths]
    df_predictions = pd.concat(dfs, ignore_index=True)
    df_predictions.columns = ["input_desc", "predicted_target"]
    columns_to_lower = ["input_desc", "predicted_target"]
    df_predictions[columns_to_lower] = df_predictions[columns_to_lower].apply(lambda x: x.str.lower())

    print(f"len predictions {len(df_predictions)}")
    df_predictions = df_predictions.drop_duplicates(subset="input_desc", keep="first")
    print(f"len predictions dupes dropped {len(df_predictions)}")

    unique_input_descriptions_from_original_df = set(df["input_desc"].unique())
    unique_predicted_input_descriptions = set(df_predictions["input_desc"].unique())

    intersection = unique_input_descriptions_from_original_df.intersection(unique_predicted_input_descriptions)
    print(len(intersection))
    print(len(intersection) / len(df))