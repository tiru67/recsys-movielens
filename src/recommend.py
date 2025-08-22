# src/recommend.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_PATH = Path("data/ml-100k/u.data")
ITEM_PATH = Path("data/ml-100k/u.item")
MODEL_PATH = Path("models/svd_model.joblib")

def load_data():
    ratings = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        ITEM_PATH,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"]
    )
    return ratings, movies

def build_interaction_matrix(ratings):
    return ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

def recommend_for_user(user_id: int, top_n: int = 10):
    ratings, movies = load_data()
    user_item_matrix = build_interaction_matrix(ratings)
    svd = joblib.load(MODEL_PATH)

    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in dataset")

    # User vector
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

    # Predict ratings
    user_latent = svd.transform(user_vector)
    reconstructed = svd.inverse_transform(user_latent).flatten()

    # Mask out already-rated movies
    already_rated = user_item_matrix.loc[user_id] > 0
    reconstructed[already_rated.values] = -np.inf

    # Get top N recommendations
    top_indices = np.argsort(reconstructed)[::-1][:top_n]
    movie_ids = user_item_matrix.columns[top_indices].tolist()

    recs = movies[movies["item_id"].isin(movie_ids)]
    return recs

if __name__ == "__main__":
    user_id = 196  # Example: pick a user from your dataset
    recommendations = recommend_for_user(user_id, top_n=5)
    print(f"Top recommendations for user {user_id}:\n")
    print(recommendations)
