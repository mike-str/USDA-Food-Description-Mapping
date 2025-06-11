import spacy
import pandas as pd
import re
import os

def remove_dupe_and_na(df):
    print(f"Number of Rows: {len(df)}")

    df = df.dropna()
    print(f"Number of Rows After NA Removed: {len(df)}")
    
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

def compute_metrics(df, methods_arr=["fuzzy", "tfidf"]):
    """
    I was checking the index before but because there are duplicates 
    (multiple foods can map to the same ingredients and vice-versa (I think)) - 
    decided to just check if the match is what we actually expected - so even if
    there are duplicates we can still evaluate the results... uhh this comment is sort of just true for foodb
    where i misunderstood the problem a bit / was trying stuff - want to keep it here for a bit so i can refresh
    my memory later
    """
    res = []

    for method in methods_arr:
        col = f"match_{method}"

        # assuming "ingredient_description_clean" is the output we are looking for
        # TODO right now this is hard coded but eventually change to output
        if "ingredient_description_clean" in df.columns:
            correct = (df["ingredient_description_clean"] == df[col]).sum()
        else:
            correct = (df["food_name"] == df[col]).sum()

        total = len(df)
        
        accuracy = correct / total

        res.append({
            "method": method,
            "accuracy": accuracy
        })

    return pd.DataFrame(res)

def incorrect_matches(df, method="fuzzy"):
    col = f"match_{method}"
    return df[df["ingredient_description_clean"] != df[col]]

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