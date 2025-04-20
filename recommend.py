# src/recommend.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/products_with_embeddings.csv")
embeddings = np.load("models/product_embeddings.npy")

def recommend(product_index, top_n=5):
    sim = cosine_similarity(embeddings[product_index].reshape(1, -1), embeddings).flatten()
    sim[product_index] = -1
    top_idx = sim.argsort()[::-1][:top_n]
    return df.iloc[top_idx][['title', 'categoryName']], sim[top_idx]
