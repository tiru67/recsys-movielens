# src/serve_api.py
from fastapi import FastAPI, Query
import joblib, pandas as pd, numpy as np
from pathlib import Path

from src.recommend import load_data, build_interaction_matrix

MODEL_PATH = Path("models/svd_model.joblib")

app = FastAPI(title="Movie Recommender API")

@app.get("/recommend")
def recommend(user_id: int = Query(..., description="User ID"), top_n: int = 5):
    ratings, movies = load_data()
    user_item_matrix = build_interaction_matrix(ratings)
    svd = joblib.load(MODEL_PATH)

    if user_id not in user_item_matrix.index:
        return {"error": f"User {user_id} not found"}

    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    user_latent = svd.transform(user_vector)
    reconstructed = svd.inverse_transform(user_latent).flatten()

    already_rated = user_item_matrix.loc[user_id] > 0
    reconstructed[already_rated.values] = -np.inf

    top_indices = np.argsort(reconstructed)[::-1][:top_n]
    movie_ids = user_item_matrix.columns[top_indices].tolist()
    recs = movies[movies["item_id"].isin(movie_ids)]

    return recs.to_dict(orient="records")
