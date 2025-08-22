# src/app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.write("Personalized movie recommendations powered by TruncatedSVD (MovieLens 100k).")

MODEL_PATH = "models/svd_model.joblib"
MATRIX_PATH = "models/user_item_matrix.joblib"
MOVIE_META_PATH = "models/movies.csv"

@st.cache_resource
def load_resources():
    svd = joblib.load(MODEL_PATH)
    user_item_matrix = joblib.load(MATRIX_PATH)
    movies = pd.read_csv(MOVIE_META_PATH)
    return svd, user_item_matrix, movies

svd, user_item_matrix, movies = load_resources()

def recommend_movies(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in dataset.")

    # Compute latent factors
    user_factors = svd.transform(user_item_matrix)
    item_factors = svd.components_.T
    pred_matrix = np.dot(user_factors, item_factors.T)

    user_idx = user_item_matrix.index.get_loc(user_id)
    preds = pred_matrix[user_idx]

    # Remove movies the user already rated
    rated_items = user_item_matrix.loc[user_id]
    unrated_mask = rated_items == 0
    preds_unrated = preds * unrated_mask.values

    # Top-N recommendations
    top_items = np.argsort(preds_unrated)[::-1][:top_n]
    top_movie_ids = user_item_matrix.columns[top_items]

    recs = movies[movies["item_id"].isin(top_movie_ids)].copy()
    recs["pred_rating"] = preds_unrated[top_items]
    return recs.sort_values(by="pred_rating", ascending=False)

# Streamlit UI
user_id = st.number_input("Enter a User ID (1–943):", min_value=1, max_value=943, value=196)
top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    try:
        recs = recommend_movies(int(user_id), top_n)
        st.success(f"Top {top_n} Recommendations for User {user_id}")
        st.table(recs)
    except Exception as e:
        st.error(f"Error: {e}")
