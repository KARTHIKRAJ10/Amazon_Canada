# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess(file_path):
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)

    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['categoryName'])

    scaler = MinMaxScaler()
    df[['stars', 'reviews', 'price', 'listPrice', 'boughtInLastMonth']] = scaler.fit_transform(
        df[['stars', 'reviews', 'price', 'listPrice', 'boughtInLastMonth']]
    )

    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['title']).toarray()

    features = np.concatenate([
        tfidf_matrix,
        df[['category_encoded', 'stars', 'reviews', 'price', 'listPrice', 'boughtInLastMonth']].values
    ], axis=1)

    return df, features
