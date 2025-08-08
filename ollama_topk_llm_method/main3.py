"""
if hallucinates target, will have it rerun same prompt
"""

from util import load_nhanes
from util import embed_match_top_n, compute_top_k_accuracy
from util import generate_prompt, prompt_llm
import pandas as pd
import json
import pickle

def load_asa():
    df = pd.read_excel("data_asa/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
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

system_message = (
    "You are a strict prediction engine. You only return valid JSON objects, "
    "exactly in the format specified. Do not explain. Do not skip. "
    "Never return an empty object. Always return a value from the candidate list."
)

if __name__ == "__main__":
    df = load_asa()
    # REMOVING EXACT MATCHES FROM INPUT AND TARGET
    # df = df[df["input_desc"] != df["target_desc"]]

    input_desc_list = df["input_desc"].to_list()
    
    # targets are loaded in from all of foodb...
    input_desc_list = df["input_desc"].to_list()
    # read in foodb for entire database, as is it is seemingly encoded in latin 1
    target_desc_list = pd.read_csv("data_asa/FooDB_Unique_Descriptions.csv", encoding="latin-1")
    # convert latin 1 to utf-8
    target_desc_list["orig_food_common_name_uncleaned"] = [s.encode("utf-8").decode("utf-8") for s in target_desc_list["orig_food_common_name_uncleaned"]]
    # drop na
    target_desc_list = target_desc_list.dropna(subset=["orig_food_common_name_uncleaned"])
    # take unique only
    target_desc_list = list(set(target_desc_list["orig_food_common_name_uncleaned"].to_list()))
    # lowercase
    target_desc_list = [s.lower() for s in target_desc_list]
    # unique again - probably a simpler way to do this
    target_desc_list = list(set(target_desc_list))

    targets_set = set(target_desc_list)

    top_k = 5
    top_k_results = embed_match_top_n(input_desc_list, target_desc_list, n=top_k)

    acc = compute_top_k_accuracy(df, top_k_results, k=top_k)
    print(f"Top-{top_k} Accuracy: {acc:.2%}")

    results = []

    total, correct = 0, 0
    target_hallucination_count = 0
    num_empty_json_prediction = 0
    responses_arr = []
    for i, row in enumerate(top_k_results):
        input_description = row["input_desc"]
        target_desc_arr = [item["target"] for item in row["top_matches"]]

        prompt = generate_prompt(input_description, target_desc_arr)

        # with open("prompt_example.txt", "w") as f:
        #     f.write(prompt)

        # THIS RETURNS A JSON OBJECT
        """
        #################
        right now we are going to reprompt if it returns empty json, we will reprompt 5 times.
        ################
        """
        for j in range(10):
            response = prompt_llm(prompt, system_message)
            if response != {}:
                predicted = response.get("predicted_target", "").strip().lower()
                if predicted in targets_set:
                    break
        print(response)

        predicted = ""
        try:
            # we want to use input_desc when checking the accuracy too, because it could hallucinate this
            predicted = response.get("predicted_target", "").strip().lower()
        except json.JSONDecodeError:
            # print(" Failed to parse JSON:", response)
            continue

        ground_truth = df[df["input_desc"] == input_description]["target_desc"].values[0]
        is_correct = predicted == ground_truth

        if predicted not in targets_set and response != {}:
            target_hallucination_count += 1
            # print("NOT IN GROUND TRUTH TARGETS")
        
        if response == {}:
            num_empty_json_prediction += 1

        # print(f"Input:          {input_description}")
        # print(f"Ground Truth:   {ground_truth}")
        # print(f"Predicted:      {predicted}")
        # print(f"Correct:      {is_correct}\n")

        # if input description is predicted incorrectly then this will also result in an incorrect prediction
        input_description_llm = response.get("input_desc", "").strip().lower()

        # preliminary accuracy stuff
        total += 1
        if is_correct and input_description_llm == input_description:
            correct += 1

        responses_arr.append(response)

        if i > 0 and (i + 1) % 10 == 0:
            print()
            print(f"LLM Top-{top_k} Accuracy: {correct}/{total} = {correct / total:.2%}")
            print(f"Number of times hallucinated a target: {target_hallucination_count}")
            print(f"Number of times empty json: {num_empty_json_prediction}")
            print()

    with open("asa_responses.pkl", "wb") as f:
        pickle.dump(responses_arr, f)
