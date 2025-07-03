from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def tfidf_match(input_desc_list, target_desc_list):
    assert isinstance(input_desc_list, list)
    assert isinstance(target_desc_list, list)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True
    ).fit(target_desc_list)

    tfidf_input = vectorizer.transform(input_desc_list)   # queries
    tfidf_target = vectorizer.transform(target_desc_list) # corpus

    similarity_matrix = cosine_similarity(tfidf_input, tfidf_target)

    results = []
    for i, row in enumerate(similarity_matrix):
        best_index = np.argmax(row)
        best_score = row[best_index]
        best_match = target_desc_list[best_index]
        results.append((best_match, best_score))

    results = pd.DataFrame(results, columns=["match_tfidf", "score_tfidf"])
    return results