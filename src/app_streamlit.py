import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import joblib
import os

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.write("Personalized movie recommendations powered by SVD (MovieLens 100k).")

# --------------------------
# Load & Train (cached)
# --------------------------
@st.cache_resource
def load_data_and_train():
    # Load ratings
    df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None,
                         names=["item_id", "title"], usecols=[0,1])

    # Surprise dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2)

    # Train or load model
    model_path = "models/svd_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = SVD()
        model.fit(trainset)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)

    return df, movies, model

ratings, movies, model = load_data_and_train()

# --------------------------
# Recommendation Function
# --------------------------
def recommend_movies(user_id, top_n=5):
    # All movies
    all_items = movies["item_id"].unique()

    # Movies already rated
    rated_items = ratings[ratings["user_id"] == user_id]["item_id"].tolist()

    # Predict ratings for unrated movies
    predictions = []
    for iid in all_items:
        if iid not in rated_items:
            pred = model.predict(user_id, iid).est
            predictions.append((iid, pred))

    # Top N
    top_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    top_df = pd.DataFrame(top_preds, columns=["item_id", "pred_rating"]).merge(movies, on="item_id")
    return top_df[["item_id", "title", "pred_rating"]]

# --------------------------
# Streamlit UI
# --------------------------
user_id = st.number_input("Enter a User ID (1–943):", min_value=1, max_value=943, value=196)
top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    try:
        recs = recommend_movies(int(user_id), top_n)
        st.success(f"Top {top_n} Recommendations for User {user_id}")
        st.table(recs)
    except Exception as e:
        st.error(f"Error: {e}")
