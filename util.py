import spacy
import pandas as pd
import re
import os
import numpy as np

def load_nhanes():
    # df = pd.read_csv("data/nhanes_dfg2_map_all.csv")
    # df.loc[df["replace"].notna() & (df["replace"] != ""), "simple_name"] = df["replace"]

    # # Drop rows where 'match' == "n" and 'replace' is empty
    # df = df[~((df["match"] == "n") & (df["replace"].isna() | (df["replace"] == "")))]
    # df = df[df["match"] != "n"]

    # df = df[["ingred_desc", "simple_name", "match"]]
    # # df.columns = ["input", "target", "match"]

    # df = df[["ingred_desc", "simple_name"]]
    # df.columns = ["input", "target"]

    # df = df.drop_duplicates(subset="input", keep="first")

    df = pd.read_csv("data/nhanes_dfg2_labels.csv")

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input", "target", "label"]

    return df

def load_nhanes2():
    df = pd.read_csv("archive/data/nhanes_dfg2_map_all.csv")
    # df.loc[df["replace"].notna() & (df["replace"] != ""), "simple_name"] = df["replace"]

    df["label"] = df["match"].apply(lambda x: 0 if x == "n" else 1)

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input", "target", "label"]

    df = df.drop_duplicates(subset="input", keep="first")

    return df

def load_asa():
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-24-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description", "orig_food_common_name", "Ingredient_code", "orig_food_id"]]
    df.columns = ["input", "target", "input_id", "target_id"]

    columns_to_lower = ["input", "target"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    # remove row if input or target is NA
    df = df.dropna(subset=["input", "target"])
    print(f"Number of Rows After NA Removed: {len(df)}")

    # if the input exists twice, it should map to the same target - no reason to keep this
    df = df.drop_duplicates(subset="input", keep="first")
    print(f"Number of Rows After Duplicate Inputs Removed: {len(df)}")

    id_dict = dict()
    for i in range(len(df)):
        input_desc, input_id = df.iloc[i]["input"], df.iloc[i]["input_id"]
        target_desc, target_id = df.iloc[i]["target"], df.iloc[i]["target_id"]

        if pd.notna(input_id): id_dict[input_desc] = int(input_id)
        if pd.notna(target_id): id_dict[target_desc] = int(target_id)

    return df, id_dict

def remove_dupe_and_na(df):
    print(f"Number of Rows: {len(df)}")

    
    
    df = df.drop_duplicates()
    print(f"Number of Rows After Duplicates Dropped: {len(df)}")

    return df

def remove_rows_where_columns_match(df, col_1, col_2):
    return df[df[col_1] != df[col_2]]

def is_valid_token(token):
    return (
        not token.is_stop and 
        not token.is_punct and 
        not token.is_space and
        token.is_alpha
    )

def clean_text(arr):
    assert isinstance(arr, (list, pd.Series)), "arr must be a list or pd.Series"

    arr = [re.sub(r"[^a-zA-Z\s]", " ", s).lower().strip() for s in arr]

    nlp = spacy.load("en_core_web_sm")

    cleaned_tokens = [" ".join([token.lemma_.lower() for token in nlp(s) if is_valid_token(token)]) for s in arr]
    return cleaned_tokens

def incorrect_matches(df, method="fuzzy"):
    col = f"match_{method}"
    return df[df["target"] != df[col]]

def merge_results():
    pass

def save_results(df, file_name):
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/{file_name}", index=False)
    return 0

if __name__ == "__main__":
    text = ["This is a 4444##$simple example, showing how to clean text using spaCy!"]
    cleaned = clean_text(text)
    print(cleaned)