from rapidfuzz import fuzz, process
import pandas as pd

def fuzzy_match(list1, list2):
    """
    Perform fuzzy string matching between list1 and list2.
    Returns a list of tuples: (item_from_list1, best_match_from_list2, score)
    """
    results = []
    for i, item in enumerate(list1):
        best_match, score, best_match_index = process.extractOne(item, list2, scorer=fuzz.ratio)
        results.append((i, best_match, score, best_match_index))

    results = pd.DataFrame(results,  columns=["index", "match_fuzzy", "score_fuzzy", "index_match_fuzzy"])
        
    return results

if __name__ == "__main__":
    list1 = ["apple pie", "banana bread", "cheddar"]
    list2 = ["apple", "banana muffin", "cheddar cheese", "banana bre"]

    print(fuzzy_match(list1, list2))
