import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

users_test = pd.read_csv(DATA_DIR / "users_test.csv")
interactions_test = pd.read_csv(DATA_DIR / "interactions_test.csv")

from popular_recommender import get_top_products_by_segment

def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k

def evaluate_cold_start(k=10):

    precisions = []

    for user_id in users_test["user_id"].unique():

        relevant_items = interactions_test[
            interactions_test["user_id"] == user_id
        ]["item_id"].tolist()

        if len(relevant_items) == 0:
            continue

        recs_df = get_top_products_by_segment(
            segment=users_test[
                users_test["user_id"] == user_id
            ]["segment"].iloc[0],
            top_k=k
        )

        recommended_items = recs_df["item_id"].tolist()

        precisions.append(
            precision_at_k(recommended_items, relevant_items, k)
        )

    return np.mean(precisions)

if __name__ == "__main__":

    print("Cold Start Evaluation")
    print("Popular Precision@10:", evaluate_cold_start())