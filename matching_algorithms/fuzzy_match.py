from rapidfuzz import fuzz, process
import pandas as pd

def fuzzy_match(input_desc_list, target_desc_list):
    assert type(input_desc_list) == list
    assert type(target_desc_list) == list
    # assert len(set(input_desc_list)) == len(input_desc_list), "input_desc_list must not contain duplicates"

    results = []
    for input_desc in input_desc_list:
        best_match, score, best_match_index = process.extractOne(input_desc, target_desc_list, scorer=fuzz.ratio)
        results.append((best_match, score))

    results = pd.DataFrame(results,  columns=["match_fuzzy", "score_fuzzy"])
        
    return results

if __name__ == "__main__":
    arr1 = ["apple pie", "banana bread", "cheddar"]
    arr2 = ["apple", "banana muffin", "cheddar cheese", "banana bre"]

    print(fuzzy_match(arr1, arr2))
