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

def load_asa():
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description", "orig_food_common_name", "Ingredient_code", "orig_food_id"]]
    df.columns = ["input_desc", "target_desc", "input_id", "target_id"]

    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    # remove row if input or target is NA
    df = df.dropna(subset=["input_desc", "target_desc"])
    # print(f"Number of Rows After NA Removed: {len(df)}")

    # if the input exists twice, it should map to the same target - no reason to keep this
    df = df.drop_duplicates(subset="input_desc", keep="first")
    # print(f"Number of Rows After Duplicate Inputs Removed: {len(df)}")

    return df

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

def compute_accuracy(df, match_algorithm):
    correct_match = df[f"match_{match_algorithm}"] == df["target_desc"]
    accuracy = correct_match.mean()

    return accuracy