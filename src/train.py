# src/train.py
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import joblib

DATA_PATH = Path("data/ml-100k/u.data")
MOVIES_PATH = Path("data/ml-100k/u.item")
MODEL_PATH = Path("models/svd_model.joblib")
MATRIX_PATH = Path("models/user_item_matrix.joblib")
MOVIE_META_PATH = Path("models/movies.csv")

def load_data():
    ratings = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        MOVIES_PATH,
        sep="|",
        encoding="latin-1",
        header=None,
        names=["item_id", "title"],
        usecols=[0, 1]
    )
    return ratings, movies

def build_interaction_matrix(df):
    return df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

def train_svd(user_item_matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(user_item_matrix)
    return svd

if __name__ == "__main__":
    print("Loading data...")
    ratings, movies = load_data()

    print("Building interaction matrix...")
    mat = build_interaction_matrix(ratings)

    print("Training SVD model...")
    svd = train_svd(mat, n_components=50)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svd, MODEL_PATH)
    joblib.dump(mat, MATRIX_PATH)
    movies.to_csv(MOVIE_META_PATH, index=False)

    print(f"✅ Model saved to {MODEL_PATH.resolve()}")
    print(f"✅ Matrix saved to {MATRIX_PATH.resolve()}")
    print(f"✅ Movies metadata saved to {MOVIE_META_PATH.resolve()}")
