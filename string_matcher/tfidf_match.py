from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

"""
Perform TF-IDF based string matching from list1 to list2.
Returns list of (item_from_list1, best_match_from_list2, similarity_score).


# what the other intern did was list2 + list1[i] is the corpus so all of list 2 and the ith
# entry you are on is the corpus


# other method:
When This Is Fine
If you're matching short lists
If each query should be evaluated in its own context
###### if list2 is reallllllly big then this will be incredibly slow because you are remaking 
########### the "model" everytime you check for a match from list1... 
################ you make a new model len(list1) times
##
It may have actually just been wrong and doing this instead:

It finds the best match for one query from a list
It does not compare every string in strings to every other string in strings
##





# if this is the case then use only list2 as the corpus
Imagine you're Google:

The corpus (list2) is the web (pre-indexed)
The query (list1) is user input

Google doesn’t re-learn its TF-IDF weighting from your search — it uses only the web content to decide what's rare or important.
"""

def tfidf_match(list1, list2):
    vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            lowercase=True).fit(list2)
    
    tfidf_1 = vectorizer.transform(list1) # query
    tfidf_2 = vectorizer.transform(list2) # corpus

    similarity_matrix = cosine_similarity(tfidf_1, tfidf_2)

    results = []
    for i, row in enumerate(similarity_matrix):
        best_index = np.argmax(row)
        best_score = row[best_index]
        results.append((i, list2[best_index], best_score, best_index))

    results = pd.DataFrame(results, columns=["index", "match_tfidf", "score_tfidf", "index_match_tfidf"])

    return results

if __name__ == "__main__":
    list1 = ["apple pie", "banana bread", "cheddar"]
    list2 = ["cheddar cheese", "apple", "banana muffin", "banana bre"]

    results = tfidf_match(list1, list2)

    df = pd.DataFrame(results)
    print(df)
