import pandas as pd
import os
from glob import glob

def load_asa():
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description_uncleaned", "orig_food_common_name_uncleaned"]]
    df.columns = ["input_desc", "target_desc"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    # remove row if input or target is NA
    df = df.dropna(subset=["input_desc", "target_desc"])

    # if the input exists twice, it should map to the same target - no reason to keep this
    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

import pandas as pd
import os
from glob import glob

def load_predictions(predictions_folder):
    all_txt_files = glob(os.path.join(predictions_folder, "*.txt"))

    df_list = []
    for file in all_txt_files:
        valid_lines = []
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                fields = line.strip().split("\t")
                if len(fields) == 2:
                    valid_lines.append(line)
                else:
                    print(f"[Malformed Line] {file} (line {i}): {line.strip()}")

        # Convert valid lines to DataFrame
        from io import StringIO
        if valid_lines:  # only parse if non-empty
            df = pd.read_csv(StringIO("".join(valid_lines)), sep="\t", header=None, names=["input_desc", "predicted_desc"])
            df["source_file"] = os.path.basename(file)
            df_list.append(df)

    # Combine all prediction DataFrames
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df[["input_desc", "predicted_desc"]] = combined_df[["input_desc", "predicted_desc"]].apply(lambda x: x.str.lower())
        return combined_df
    else:
        print("No valid prediction data found.")
        return pd.DataFrame(columns=["input_desc", "predicted_desc", "source_file"])

if __name__ == "__main__":
    nhanes_df = load_asa()
    predictions_df = load_predictions("claude_checking/asa24/data/predictions")
    predictions_df = predictions_df.drop_duplicates(subset="input_desc", keep="first")

    # Merge predictions with ground truth
    merged = predictions_df.merge(nhanes_df, on="input_desc", how="left")

    # Evaluate correctness
    merged["correct"] = merged["predicted_desc"] == merged["target_desc"]

    # Accuracy: penalize for incorrect and missing (null target_desc)
    accuracy = merged["correct"].sum() / len(nhanes_df)

    # Summary
    print(f"Total prediction rows unique: {len(predictions_df)}")
    print(f"Total ground truth inputs: {len(nhanes_df)}")
    print(f"Accuracy (penalizing for missing and incorrect): {accuracy:.2%}")

    # Output
    # merged.to_csv("result_nhanes_combined.csv", index=False)

    accuracy_str = f"accuracy: {accuracy:.2%}"

    os.makedirs("claude_checking/asa24/results", exist_ok=True)
    with open("claude_checking/asa24/results/accuracy.txt", "w") as f:
        f.write(accuracy_str)

    merged.to_csv("claude_checking/asa24/results/results_asa24_claude.csv", index=False)
