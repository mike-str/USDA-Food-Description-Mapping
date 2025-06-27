from util import load_nhanes, load_nhanes2
from util import clean_text, save_results
from util import remove_dupe_and_na, remove_rows_where_columns_match, incorrect_matches
from util_metrics import compute_metrics
from string_matcher import match
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

if __name__ == "__main__":
    df = load_nhanes()
    df["index"] = [i for i in range(len(df))]
    
    print(df.columns)


    # # this can be all possile targets in the original database - or it can be just the target matches in the provided dataframe
    candidate_matches = list(set(df["target"]))
    input_desc = df["input"].to_list()

    df_results = match(input_desc, candidate_matches, method="embed")
    merged_df = pd.merge(df, df_results, on="index", how="left")
    
    merged_df["label_embed"] = [0 if score < 0.90 else 1 for score in merged_df["score_embedding"]]

    # Accuracy
    accuracy = accuracy_score(merged_df["label"], merged_df["label_embed"])
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(merged_df["label"], merged_df["label_embed"])
    print("Confusion Matrix:")
    print(conf_matrix)