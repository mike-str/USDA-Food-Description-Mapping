##################################################################################################
# checking_nhanes_exp_4_claude_api.py
# Purpose: this code checkes the results of NHANES experiment 4 against ground truth
# 
# Input, ground truth: nhanes_dfg2_labels.csv
# Input, Claude results: matched_NHANES_exp4_results.txt
# Output prefix to create 
#   PREFIX_stdout.txt
#   PREFIX_merged_preditions.csv
#   PREFIX_label_is_1_and_pred_is_wrong.csv
#
# Example usage: python checking_nhanes_exp_4_claude_api.py nhanes_dfg2_labels.csv matched_haiku_NHANES_exp4_results.txt haiku_NHANESexp4
# 
# Authors: Michael Strohmeier & Danielle Lemay
#################################################################################################
import sys
import pandas as pd
import numpy as np
import re
import csv

# Check if correct number of arguments provided
if len(sys.argv) != 4:
    print("Usage: python check_nhanes_exp_4.py <input_file1.csv> <input_file2.csv> <Outputfile_prefix>")
    sys.exit(1)

# Get filenames from command line arguments
input_file1 = sys.argv[1]
input_file2 = sys.argv[2]
prefix = sys.argv[3]

# Redirect stdout to file based on prefix
stdout_file = f"{prefix}stdout.txt"
sys.stdout = open(stdout_file, 'w')

# function to read ground truth
def load_nhanes():
    df = pd.read_csv("nhanes_dfg2_labels.csv")

    print("there were 256 unique targets for nhanes exp 4\n")

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input_desc", "target_desc", "label"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

if __name__ == "__main__":
    df = load_nhanes()

    # checking original dataset
    num_0s = len(df[df["label"] == 0]) # ground truth number of nones
    num_1s = len(df[df["label"] == 1])
    print("Length of ground truth table", len(df))
    print("Ground truth number of 0s (nones):", num_0s)
    print("Ground truth number of 1s (match exists):", num_1s)


    # df = df[df["label"] == 1]
    # print(df.columns)

    # read claude's results
    claude_df = pd.read_csv(input_file2, sep="\t", quotechar='"')
    
    claude_df.columns = ["input_desc", "match_claude"]

    # Remove pesky quotes
    df = df.map(lambda x: re.sub(r'[\'"]', '', str(x)))
    claude_df = claude_df.map(lambda x: re.sub(r'[\'"]', '', str(x)))
    #claude_df = claude_df.map(
     #   lambda x: str(x).replace('"', '').replace('""', '').replace('"""', '')
    #)
    #convert label column to integers
   
    print()

    num_pred_none = len(claude_df[claude_df["match_claude"] == "none"])
    print("Length of claude table", len(claude_df))
    print("claude number of predicted 0s (none)", num_pred_none)

    max_possible_accuracy = ((num_0s - num_pred_none + num_1s) / len(df))
    print("max POSSIBLE accuracy IF every match WAS correct", max_possible_accuracy)

    # check that every ground truth input was also evaluated/output by Claude
    unique_ground_truth_inputs = df["input_desc"].unique()
    claude_inputs_not_in_ground_truth = []
    for input_desc in claude_df["input_desc"]:
        if input_desc not in unique_ground_truth_inputs:
            claude_inputs_not_in_ground_truth.append(input_desc)
    
    print()
    print("If this next part is 0, we did not hallucinate any inputs")
    print("Number of claude inputs not in ground truth:", len(claude_inputs_not_in_ground_truth))
    print()

    # merge the ground truth and the predictions into one data frame, linking on the input description
    merged_df = pd.merge(df, claude_df, on="input_desc", how="left")
     # Convert the label column to integers
    merged_df['label'] = merged_df['label'].astype(int)
    # check
    #merged_df.to_csv("tmp_predictions.csv", index=False, sep="\t")

    # this is just checking if (match=="none" and label == 0), which is a valid match
    # OR (match==target and label == 1), which is a valid match
    # else in either case of label 0 or 1 if the  additional criteria is not met then it is not a match
    is_match = []
    for i in range(0, len(merged_df)):
        if merged_df.iloc[i]["label"] == 1:
            if merged_df.iloc[i]["target_desc"] == merged_df.iloc[i]["match_claude"]:
                is_match.append(1)
            else:
                is_match.append(0)
        elif merged_df.iloc[i]["label"] == 0:
            if ((merged_df.iloc[i]["match_claude"] == "none")):
                is_match.append(1)
            else:
                is_match.append(0)

    #print()
    #print("length is ", len(merged_df))
    #print("merged_df.iloc[0][label] is ", merged_df.iloc[0]["label"])
    #print("merged_df.iloc[0][target_desc] is ", merged_df.iloc[0]["target_desc"] )
    #print("merged_df.iloc[0][match_claude] is ", merged_df.iloc[0]["match_claude"])
    #print("is_match is", is_match)
    #print()  

    # Check data types
    #print("Column dtypes:")
    #print(merged_df[['label', 'target_desc', 'match_claude']].dtypes)

    # Check for string vs integer labels
    #print("Unique labels:", merged_df['label'].unique())
    #print("Label types:", [type(x) for x in merged_df['label'].unique()])

    merged_df["is_match"] = is_match
    
    print("Accuracy from first 100 matches:", merged_df["is_match"][:100].sum() / 100)
    print("Accuracy (whole dataset)", merged_df["is_match"].sum() / len(merged_df))

    print()
    merged_df_label_1_only = merged_df[merged_df["label"] == 1]
    print("merged_df_label_1_only number of rows: ", len(merged_df_label_1_only))
    print("accuracy when only checking label == 1 (a valid match does exist):", merged_df_label_1_only["is_match"].sum() / len(merged_df_label_1_only))

    # output mismatches
    debug_file = f"{prefix}_label_is_1_pred_is_wrong.csv"
    merged_df_label_1_only[merged_df_label_1_only["is_match"] == 0].to_csv(debug_file, index=False)
    # output all predictions
    all_pred_file = f"{prefix}_predictions_merged.csv"
    merged_df.to_csv(all_pred_file, index=False)
