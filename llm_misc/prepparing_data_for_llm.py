import sys
import os
import random
import pandas as pd

# adding parent directory to sys.path in order to import util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util import load_asa, load_nhanes

if __name__ == "__main__":
    os.makedirs("data/llm/asa24_experiment_1/",  exist_ok=True)
    os.makedirs("data/llm/asa24_experiment_2/",  exist_ok=True)
    os.makedirs("data/llm/asa24_experiment_3/",  exist_ok=True)
    os.makedirs("data/llm/asa24_experiment_4/",  exist_ok=True)
    os.makedirs("data/llm/nhanes_experiment_1/", exist_ok=True)
    os.makedirs("data/llm/nhanes_experiment_2/", exist_ok=True)
    os.makedirs("data/llm/nhanes_experiment_3/", exist_ok=True)
    os.makedirs("data/llm/nhanes_experiment_4/", exist_ok=True)

    random.seed(8)

    # == asa24 experiment 1 ==
    df = load_asa()
    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/asa24_experiment_1/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/asa24_experiment_1/target_desc_list.csv", index=False)

    #  == asa24 experiment 2 ==
    df = load_asa()
    input_desc_list = df["input_desc"].to_list()

    # targets are coming from all of foodb... dropna, select unique, lowercase all
    target_desc_list = pd.read_csv("data/FooDB_Unique_Descriptions.csv", encoding="latin-1")
    target_desc_list["orig_food_common_name_uncleaned"] = [s.encode("utf-8").decode("utf-8") for s in target_desc_list["orig_food_common_name_uncleaned"]]
    target_desc_list = target_desc_list.dropna(subset=["orig_food_common_name_uncleaned"])
    target_desc_list = list(set(target_desc_list["orig_food_common_name_uncleaned"].to_list()))
    target_desc_list = [s.lower() for s in target_desc_list]
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/asa24_experiment_2/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/asa24_experiment_2/target_desc_list.csv", index=False)

    # == asa24 experiment 3 ==
    df = load_asa()
    df = df[df["input_desc"] != df["target_desc"]]
    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/asa24_experiment_3/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/asa24_experiment_3/target_desc_list.csv", index=False)

    #  == asa24 experiment 4 ==
    df = load_asa()
    df = df[df["input_desc"] != df["target_desc"]]
    input_desc_list = df["input_desc"].to_list()

    # targets are coming from all of foodb... dropna, select unique, lowercase all
    target_desc_list = pd.read_csv("data/FooDB_Unique_Descriptions.csv", encoding="latin-1")
    target_desc_list["orig_food_common_name_uncleaned"] = [s.encode("utf-8").decode("utf-8") for s in target_desc_list["orig_food_common_name_uncleaned"]]
    target_desc_list = target_desc_list.dropna(subset=["orig_food_common_name_uncleaned"])
    target_desc_list = list(set(target_desc_list["orig_food_common_name_uncleaned"].to_list()))
    target_desc_list = [s.lower() for s in target_desc_list]
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/asa24_experiment_4/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/asa24_experiment_4/target_desc_list.csv", index=False)

    # == nhanes experiment 1 ==
    df = load_nhanes()

    df = df[df["label"] == 1]

    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/nhanes_experiment_1/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/nhanes_experiment_1/target_desc_list.csv", index=False)

    # == nhanes experiment 2 ==
    df = load_nhanes()

    input_desc_list, target_desc_list = df["input_desc"].to_list(), df["target_desc"].to_list()
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/nhanes_experiment_2/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/nhanes_experiment_2/target_desc_list.csv", index=False)

    # == nhanes experiment 3 ==
    df = load_nhanes()

    df = df[df["label"] == 1]

    input_desc_list = df["input_desc"].to_list()
    target_desc_list = pd.read_csv("data/dfg2_food_descriptions.csv")["simple_name"]
    target_desc_list = [s.lower() for s in target_desc_list]
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/nhanes_experiment_3/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/nhanes_experiment_3/target_desc_list.csv", index=False)

    # == nhanes experiment 4 ==
    df = load_nhanes()

    input_desc_list = df["input_desc"].to_list()
    target_desc_list = pd.read_csv("data/dfg2_food_descriptions.csv")["simple_name"]
    target_desc_list = [s.lower() for s in target_desc_list]
    target_desc_list = list(set(target_desc_list))
    random.shuffle(target_desc_list)

    df_input = pd.DataFrame({"input_desc": input_desc_list})
    df_input.to_csv("data/llm/nhanes_experiment_4/input_desc_list.csv", index=False)

    df_target = pd.DataFrame({"target_desc": target_desc_list})
    df_target.to_csv("data/llm/nhanes_experiment_4/target_desc_list.csv", index=False)