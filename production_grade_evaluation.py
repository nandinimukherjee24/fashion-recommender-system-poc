import pandas as pd
import numpy as np
from scipy.stats import norm

from personalized_recommender import recommend_hybrid, users, interactions
from personalized_recommender import POPULARITY_TABLE

# ---------------------------
# CONFIG
# ---------------------------
N_USERS = 2000
TOP_K = 10
SEED = 42
np.random.seed(SEED)

# ---------------------------
# Popular Recommender
# ---------------------------
def recommend_popular(user_id, top_k=10):
    user_segment = users.loc[
        users["user_id"] == user_id, "segment"
    ].values[0]

    segment_items = POPULARITY_TABLE[
        POPULARITY_TABLE["segment"] == user_segment
    ]

    ranked = segment_items.sort_values(
        "popularity_score", ascending=False
    )

    return ranked.head(top_k)[
        ["item_id", "brand", "category"]
    ]

# ---------------------------
# Position Weight
# ---------------------------
def position_weight(rank):
    return 1 / np.log2(rank + 2)

# ---------------------------
# Click Simulation
# ---------------------------
def simulate_click(row, user_history, rank):
    base_ctr = 0.05
    pos_weight = position_weight(rank)

    prob = base_ctr * pos_weight

    if not user_history.empty:
        if row["brand"] in user_history["brand"].values:
            prob += 0.04
        if row["category"] in user_history["category"].values:
            prob += 0.02

    return np.random.rand() < prob

# ---------------------------
# Conversion Simulation
# ---------------------------
def simulate_conversion(row, user_history):
    base_cvr = 0.18
    prob = base_cvr

    if not user_history.empty:
        if row["brand"] in user_history["brand"].values:
            prob += 0.04
        if row["category"] in user_history["category"].values:
            prob += 0.03

    return np.random.rand() < prob

# ---------------------------
# Run Experiment
# ---------------------------
def run_experiment():

    sample_users = users.sample(N_USERS)
    results = {}

    for seg in users["segment"].unique():
        results[seg] = {
            "popular": {"clicks":0,"conv":0,"eng":0,"impr":0},
            "personalized":{"clicks":0,"conv":0,"eng":0,"impr":0}
        }

    for _, user_row in sample_users.iterrows():

        user_id = user_row["user_id"]
        seg = user_row["segment"]

        user_history = interactions[
            interactions["user_id"] == user_id
        ].merge(
            POPULARITY_TABLE[["item_id","brand","category"]],
            on="item_id",
            how="left"
        )

        # Popular
        pop_recs = recommend_popular(user_id, TOP_K)
        for rank, (_, row) in enumerate(pop_recs.iterrows()):
            results[seg]["popular"]["impr"] += 1
            if simulate_click(row, user_history, rank):
                results[seg]["popular"]["clicks"] += 1
                results[seg]["popular"]["eng"] += 1
                if simulate_conversion(row, user_history):
                    results[seg]["popular"]["conv"] += 1

        # Personalized
        per_recs = recommend_hybrid(user_id, TOP_K)
        for rank, (_, row) in enumerate(per_recs.iterrows()):
            results[seg]["personalized"]["impr"] += 1
            if simulate_click(row, user_history, rank):
                results[seg]["personalized"]["clicks"] += 1
                results[seg]["personalized"]["eng"] += 1
                if simulate_conversion(row, user_history):
                    results[seg]["personalized"]["conv"] += 1

    return results

# ---------------------------
# Print Results
# ---------------------------
if __name__ == "__main__":

    results = run_experiment()
    import pickle

    with open("segment_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nResults saved to segment_results.pkl")

    print("\n---- SEGMENT LEVEL RESULTS ----")

    for seg, data in results.items():

        pop_ctr = data["popular"]["clicks"] / max(data["popular"]["impr"],1)
        per_ctr = data["personalized"]["clicks"] / max(data["personalized"]["impr"],1)

        pop_cvr = data["popular"]["conv"] / max(data["popular"]["clicks"],1)
        per_cvr = data["personalized"]["conv"] / max(data["personalized"]["clicks"],1)

        print(f"\nSegment: {seg}")
        print(f"Popular CTR: {pop_ctr:.4f}")
        print(f"Personalized CTR: {per_ctr:.4f}")
        print(f"CTR Uplift: {per_ctr - pop_ctr:.4f}")

        print(f"Popular CVR: {pop_cvr:.4f}")
        print(f"Personalized CVR: {per_cvr:.4f}")