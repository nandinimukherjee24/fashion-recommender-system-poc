import pandas as pd
import numpy as np
from pathlib import Path

# Import your recommender functions
from personalized_recommender import recommend_hybrid
from personalized_recommender import POPULARITY_TABLE
from personalized_recommender import users
from personalized_recommender import interactions

# ============================================================
# FAST POPULAR RECOMMENDER
# ============================================================

def recommend_popular(user_id, top_k=10):

    user_row = users[users["user_id"] == user_id]
    if user_row.empty:
        return pd.DataFrame()

    segment = user_row.iloc[0]["segment"]

    segment_items = POPULARITY_TABLE[
        POPULARITY_TABLE["segment"] == segment
    ].copy()

    ranked = segment_items.sort_values(
        "popularity_score",
        ascending=False
    )

    ranked["final_score"] = ranked["popularity_score"]

    return ranked.head(top_k)[
        ["item_id", "brand", "category", "final_score"]
    ]

# ============================================================
# FAST PRECISION@K
# ============================================================

def precision_at_k(model_func, users_list, k=10):

    # Precompute purchases ONCE
    purchases = interactions[
        interactions["event_type"] == "purchase"
    ]

    user_purchase_map = (
        purchases.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    precisions = []

    for i, user_id in enumerate(users_list):

        if i % 200 == 0:
            print(f"Processed {i} users")

        recs = model_func(user_id, top_k=k)

        if recs.empty:
            continue

        actual_items = user_purchase_map.get(user_id)
        if not actual_items:
            continue

        hits = len(set(recs["item_id"]) & actual_items)
        precisions.append(hits / k)

    return np.mean(precisions) if precisions else 0

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print(" FAST TRAINING EVALUATION")
    print("==============================")

    # Use subset for speed (first 2000 users)
    user_sample = users["user_id"].unique()[:1000]

    # ---------------------------
    # Sample Recommendations
    # ---------------------------

    sample_user = user_sample[0]

    print(f"\nPersonalized Recommendations for {sample_user}")
    print(
        recommend_hybrid(sample_user)
        .to_string(index=False)
    )

    print(f"\nPopular Recommendations for {sample_user}")
    print(
        recommend_popular(sample_user)
        .to_string(index=False)
    )

    # ---------------------------
    # Metrics
    # ---------------------------

    print("\nCalculating Precision@10...")

    p_precision = precision_at_k(
        recommend_hybrid,
        user_sample
    )

    pop_precision = precision_at_k(
        recommend_popular,
        user_sample
    )

    print("\nRESULTS")
    print("Personalized Precision@10:", round(p_precision, 6))
    print("Popular Precision@10:", round(pop_precision, 6))

    if pop_precision > 0:
        lift = (p_precision - pop_precision) / pop_precision
        print("Lift vs Popular:", round(lift, 4))
    else:
        print("Lift vs Popular: undefined (popular precision = 0)")