import pandas as pd

def accuracy_fuzzy(df, col):
    correct = (df["target_clean"] == df[col]).sum()
    return correct / len(df)

def accuracy_tfidf(df, col):
    correct = (df["target_clean"] == df[col]).sum()
    return correct / len(df)

def accuracy_embed(df, col):
    correct = (df["target"] == df[col]).sum()
    return correct / len(df)

def compute_metrics(df, methods_arr=["fuzzy", "tfidf", "embed"]):
    res = []

    for method in methods_arr:
        assert "target_clean" in df.columns
        assert f"match_{method}" in df.columns

        col = f"match_{method}"

        if method == "fuzzy":
            accuracy = accuracy_fuzzy(df, col)
        elif method == "tfidf":
            accuracy = accuracy_tfidf(df, col)
        elif method == "embed":
            accuracy = accuracy_embed(df, col)

        res.append({
            "method": method,
            "accuracy": accuracy
        })

    return pd.DataFrame(res)