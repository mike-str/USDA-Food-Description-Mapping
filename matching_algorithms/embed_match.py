from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def embed_match(input_list, target_list):
    assert isinstance(input_list, list)
    assert isinstance(target_list, list)

    model = SentenceTransformer("thenlper/gte-large")

    input_vecs = model.encode(input_list, normalize_embeddings=True)
    target_vecs = model.encode(target_list, normalize_embeddings=True)

    sim_matrix = cosine_similarity(input_vecs, target_vecs)

    results = []
    for i, row in enumerate(sim_matrix):
        best_index = row.argmax()
        best_score = row[best_index]
        best_match = target_list[best_index]
        results.append((best_match, best_score))

    df_results = pd.DataFrame(results, columns=["match_embed", "score_embed"])
    return df_results
