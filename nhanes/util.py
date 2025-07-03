import pandas as pd
import spacy
import re

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

def load_nhanes():
    df = pd.read_csv("data/nhanes_dfg2_labels.csv")

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input_desc", "target_desc", "label"]

    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

def compute_accuracy(df, score_thresh, match_algorithm):
    correct_match = (
            ((df["label"] == 0) & (df[f"score_{match_algorithm}"] < score_thresh))
            |
            ((df["label"] == 1) & (df[f"match_{match_algorithm}"] == df["target_desc"]))
    )

    accuracy = correct_match.mean()
    print(f"Fuzzy accuracy (rule-based): {accuracy:.2f}")

    return accuracy