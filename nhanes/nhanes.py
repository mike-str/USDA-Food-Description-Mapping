from matching_algorithms.fuzzy_match import fuzzy_match
from matching_algorithms.tfidf_match import tfidf_match
from matching_algorithms.embed_match import embed_match
from .util import load_nhanes, clean_text, compute_accuracy
import os

def run_nhanes():
    df = load_nhanes()
    df["index"] = [i for i in range(len(df))]

    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    df["input_desc_clean"]  = clean_text(input_desc_list)
    df["target_desc_clean"] = clean_text(target_desc_list)
    input_desc_clean_list, target_desc_clean_list = df["input_desc_clean"].to_list(), df["target_desc_clean"].to_list()
    
    clean_to_raw_target_dict = dict()
    for i, target_desc_clean in enumerate(df["target_desc_clean"]):
        """
        unfortunately I have to do this because when cleaning the input / target descriptions I noticed
        that some of them will be different before cleaning, and then be the same after cleaning.

        this can lead to inflated accuracy
        """
        if target_desc_clean not in clean_to_raw_target_dict:
            clean_to_raw_target_dict[target_desc_clean] = target_desc_list[i]

    df_fuzzy = fuzzy_match(input_desc_clean_list, target_desc_clean_list)
    df_fuzzy["match_fuzzy"] = df_fuzzy["match_fuzzy"].map(clean_to_raw_target_dict)

    df_tfidf = tfidf_match(input_desc_clean_list, target_desc_clean_list)
    df_tfidf["match_tfidf"] = df_tfidf["match_tfidf"].map(clean_to_raw_target_dict)

    df_embed = embed_match(input_desc_list, target_desc_list)

    df = df.join(df_fuzzy, on="index", how="left")
    df = df.join(df_tfidf, on="index", how="left")
    df = df.join(df_embed, on="index", how="left")
    
    correct_fuzzy = df["match_fuzzy"] == df["target_desc"]
    correct_tfidf = df["match_tfidf"] == df["target_desc"]
    correct_embed = df["match_embed"] == df["target_desc"]

    # compute accuracy
    acc_fuzzy = correct_fuzzy.mean()
    acc_tfidf = correct_tfidf.mean()
    acc_embed = correct_embed.mean()

    print(f"Fuzzy Match Accuracy: {acc_fuzzy:.2f}")
    print(f"TF-IDF Match Accuracy: {acc_tfidf:.2f}")
    print(f"Embed Match Accuracy: {acc_embed:.2f}")

    print(df.columns)

    print(len(df))

    score_thresh = 0.97

    # compute accuracy again
    acc_fuzzy = compute_accuracy(df, score_thresh, "fuzzy")
    acc_tfidf = compute_accuracy(df, score_thresh, "tfidf")
    acc_embed = compute_accuracy(df, score_thresh, "embed")

    print(f"Fuzzy Match Accuracy: {acc_fuzzy:.2f}")
    print(f"TF-IDF Match Accuracy: {acc_tfidf:.2f}")
    print(f"Embed Match Accuracy: {acc_embed:.2f}")

    output_lines = [
        f"Fuzzy Match Accuracy: {acc_fuzzy:.2f}",
        f"TF-IDF Match Accuracy: {acc_tfidf:.2f}",
        f"Embed Match Accuracy: {acc_embed:.2f}",
    ]
    output_text = "\n".join(output_lines) + "\n"

    os.makedirs("results", exist_ok=True)
    with open("results/nhanes_results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    print(output_text)

if __name__ == "__main__":
    run_nhanes()