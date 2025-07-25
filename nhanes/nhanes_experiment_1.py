from util import load_nhanes, clean_text, compute_accuracy_simple
from matching_algorithms.fuzzy_match import fuzzy_match
from matching_algorithms.tfidf_match import tfidf_match
from matching_algorithms.embed_match import embed_match
import pandas as pd

def nhanes_experiment_1_run():
    df = load_nhanes()
    df["index"] = [i for i in range(len(df))]

    df = df[df["label"] == 1]

    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    target_desc_list = list(set(target_desc_list))

    input_desc_clean_list = clean_text(input_desc_list)
    target_desc_clean_list = clean_text(target_desc_list)

    """
        After cleaning, some distinct target descriptions may become identical.
        For example, two different raw targets might clean to the same string.
        
        If we don’t track the original (pre-cleaned) target descriptions, 
        this can cause inflated accuracy scores due to multiple cleaned targets
        mapping to the same cleaned string.

        To avoid this, we create a mapping from each unique cleaned target back
        to its original raw form.
    """ 
    clean_to_raw_target_dict = dict()
    for raw, clean in zip(target_desc_list, target_desc_clean_list):
        if clean not in clean_to_raw_target_dict:
            clean_to_raw_target_dict[clean] = raw

    # and then we want only unique values
    target_desc_clean_list = list(set(target_desc_clean_list))

    # matching algorithm stuff happens here
    df_fuzzy = fuzzy_match(input_desc_clean_list, target_desc_clean_list)
    df_fuzzy["match_fuzzy"] = df_fuzzy["match_fuzzy"].map(clean_to_raw_target_dict)

    df_tfidf = tfidf_match(input_desc_clean_list, target_desc_clean_list)
    df_tfidf["match_tfidf"] = df_tfidf["match_tfidf"].map(clean_to_raw_target_dict)

    df_embed = embed_match(input_desc_list, target_desc_list)

    df = df.join(df_fuzzy, on="index", how="left")
    df = df.join(df_tfidf, on="index", how="left")
    df = df.join(df_embed, on="index", how="left")

    # LLM results read in here:
    df_o3 = pd.read_json("data/llm/nhanes_experiment_1/o3_predictions.json")

    # keeping naming convention from other mapping methods
    df_o3 = df_o3.rename(columns={"predicted_target": "match_o3"})

    # filter predictions to only those where input_desc exists in df
    df_o3_filtered = df_o3[df_o3["input_desc"].isin(df["input_desc"])]

    # join filtered predictions into df
    df = df.merge(df_o3_filtered, on="input_desc", how="left")

    # results
    acc_fuzzy = compute_accuracy_simple(df, "fuzzy")
    acc_tfidf = compute_accuracy_simple(df, "tfidf")
    acc_embed = compute_accuracy_simple(df, "embed")
    acc_o3    = compute_accuracy_simple(df, "o3")

    # saving the results
    df_accuracy = pd.DataFrame({
        "method": ["fuzzy", "tfidf", "embed", "o3"],
        "accuracy": [acc_fuzzy, acc_tfidf, acc_embed, acc_o3]
    })

    df_accuracy.to_csv("results/accuracy_tables/nhanes_experiment_1_accuracy.csv", index=False)
    df.to_csv("results/csv_files/nhanes_experiment_1.csv", index=False)