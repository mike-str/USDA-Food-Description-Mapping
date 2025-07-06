import pandas as pd

if __name__ == "__main__":
    # ASA
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description", "orig_food_common_name", "Ingredient_code", "orig_food_id"]]
    df.columns = ["input_desc", "target_desc", "input_id", "target_id"]

    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    # remove row if input or target is NA
    df = df.dropna(subset=["input_desc", "target_desc"])
    df = df.drop_duplicates(subset="input_desc", keep="first")

    df["input_desc"].to_csv("data/llm/asa24_input.csv", index=False)
    unique_targets = pd.Series(df["target_desc"].unique(), name="unique_target_desc").sample(frac=1, random_state=0).reset_index(drop=True)
    unique_targets.to_csv("data/llm/asa24_unique_target.csv", index=False)

    # nhanes
    df = pd.read_csv("data/nhanes_dfg2_labels.csv")

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input_desc", "target_desc", "label"]

    df = df.drop_duplicates(subset="input_desc", keep="first")

    df["input_desc"].to_csv("data/llm/nhanes_input.csv", index=False)
    unique_targets = pd.Series(df["target_desc"].unique(), name="unique_target_desc").sample(frac=1, random_state=0).reset_index(drop=True)
    unique_targets.to_csv("data/llm/nhanes_unique_target.csv", index=False)