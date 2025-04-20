# src/evaluate.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

def evaluate_embeddings(embeddings, df):
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {score:.3f}")

    def category_consistency(index, top_n=5):
        sim = cosine_similarity(embeddings[index].reshape(1, -1), embeddings).flatten()
        sim[index] = -1
        top = sim.argsort()[::-1][:top_n]
        return sum(df.loc[t, 'categoryName'] == df.loc[index, 'categoryName'] for t in top) / top_n

    avg_consistency = np.mean([category_consistency(i) for i in np.random.choice(len(df), 10)])
    print(f"Category Consistency (Top-5): {avg_consistency:.2f}")
