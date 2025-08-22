# src/data_prep.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/ml-100k/u.data")

def load_movielens():
    # MovieLens 100k format: user_id, item_id, rating, timestamp
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return df

if __name__ == "__main__":
    df = load_movielens()
    print(df.head())
    print(f"Dataset shape: {df.shape}")

