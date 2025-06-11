from .fuzzy_match import fuzzy_match
from. tfidf_match import tfidf_match
import pandas as pd

__all__ = ["fuzzy_match", "tfidf_match", "match"]

def match(list1, list2, method="fuzzy"):
    """
    Dispatch function for string matching.

    Parameters:
    - list1: List[str] - source list
    - list2: List[str] - target list
    - method: str - one of 'fuzzy', 'tfidf', 'bert'
    - kwargs: additional keyword args passed to the matching function

    Returns:
    - List of matched pairs or similarity scores
    """
    if method == "fuzzy":
        return fuzzy_match(list1, list2)
    elif method == "tfidf":
        return tfidf_match(list1, list2)
    # elif method == "bert":
    #     return bert_match(list1, list2)
    else:
        raise ValueError(f"Unknown matching method: '{method}'. Choose from 'fuzzy', 'tfidf', or llm")
