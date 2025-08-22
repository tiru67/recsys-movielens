# src/train.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import joblib

DATA_PATH = Path("data/ml-100k/u.data")
MODEL_PATH = Path("models/svd_model.joblib")

def load_data():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return df

def build_interaction_matrix(df):
    user_item_matrix = df.pivot(
        index="user_id", columns="item_id", values="rating"
    ).fillna(0)
    return user_item_matrix

def train_svd(user_item_matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(user_item_matrix)
    return svd

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Building interaction matrix...")
    mat = build_interaction_matrix(df)

    print("Training SVD model...")
    svd = train_svd(mat, n_components=50)

    # ✅ Ensure models/ folder exists before saving
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svd, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH.resolve()}")
