import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# def top_k_accuracy(sim_matrix, true_targets, candidate_targets, k=1):
#     top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
#     top_k_predictions = [[candidate_targets[i] for i in row] for row in top_k_indices]

#     correct = sum(true in preds for true, preds in zip(true_targets, top_k_predictions))

#     return correct / len(true_targets)

def embed_match(input_list, target_list, top_k=1):
    assert type(input_list) == list
    assert type(target_list) == list

    candidate_targets = list(set(target_list))

    model = SentenceTransformer("thenlper/gte-large") 

    input_vecs  = model.encode(input_list, normalize_embeddings=True)
    target_vecs = model.encode(candidate_targets, normalize_embeddings=True)

    sim_matrix = cosine_similarity(input_vecs, target_vecs)

    results = []
    for i, row in enumerate(sim_matrix):
        best_index = np.argmax(row)
        best_score = row[best_index]
        best_match = candidate_targets[best_index]
        results.append((i, best_match, best_score)) # , best_score, best_index

    df_results = pd.DataFrame(results, columns=["index", "match_embed", "score_embedding"]) # , "score_embedding", "match_index"
    
    return df_results

if __name__ == "__main__":
    list1 = ["apple pie", "banana bread", "cheddar"]
    list2 = ["cheddar cheese", "apple", "banana muffin", "banana bre"]

    results = embed_match(list1, list2)

    df = pd.DataFrame(results)
    print(df)