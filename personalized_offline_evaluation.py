import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ----------------------------------
# Paths
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ----------------------------------
# Load Data
# ----------------------------------
users = pd.read_csv(DATA_DIR / "users.csv")
items = pd.read_csv(DATA_DIR / "items.csv")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")

# ----------------------------------
# Train-Test Split (per user)
# ----------------------------------
train_rows = []
test_rows = []

for user_id, user_data in interactions.groupby("user_id"):
    if len(user_data) < 5:
        train_rows.append(user_data)
        continue

    train_u, test_u = train_test_split(
        user_data,
        test_size=0.2,
        random_state=42
    )

    train_rows.append(train_u)
    test_rows.append(test_u)

interactions_train = pd.concat(train_rows)
interactions_test = pd.concat(test_rows)

# ----------------------------------
# Replace global interactions
# ----------------------------------
import personalized_recommender
import popular_recommender

personalized_recommender.interactions = interactions_train
popular_recommender.interactions = interactions_train

# Rebuild popularity tables
personalized_recommender.POPULARITY_TABLE = (
    personalized_recommender.build_popularity_table()
)
popular_recommender.POPULARITY_TABLE = (
    popular_recommender.build_popularity_table()
)

# ----------------------------------
# Import Recommenders
# ----------------------------------
from personalized_recommender import recommend_hybrid
from popular_recommender import get_top_products_by_segment

# ----------------------------------
# Metrics
# ----------------------------------
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k

def evaluate_model(recommend_fn, k=10):

    precisions = []

    for user_id in interactions_test["user_id"].unique():

        relevant_items = interactions_test[
            interactions_test["user_id"] == user_id
        ]["item_id"].tolist()

        if len(relevant_items) == 0:
            continue

        recs_df = recommend_fn(user_id, top_k=k)

        if recs_df.empty:
            continue

        recommended_items = recs_df["item_id"].tolist()

        precisions.append(
            precision_at_k(recommended_items, relevant_items, k)
        )

    return np.mean(precisions)

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":

    print("Offline Personalization Evaluation")

    p_score = evaluate_model(recommend_hybrid)
    pop_score = evaluate_model(get_top_products_by_segment)

    print("Personalized Precision@10:", p_score)
    print("Popular Precision@10:", pop_score)
    print("Lift:", (p_score - pop_score) / pop_score)