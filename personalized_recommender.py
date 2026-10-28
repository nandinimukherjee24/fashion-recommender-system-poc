import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -----------------------------
# Load Data
# -----------------------------
users = pd.read_csv(DATA_DIR / "users.csv")
items = pd.read_csv(DATA_DIR / "items.csv")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")

# 🔥 CRITICAL FIX: enrich interactions with brand & category
interactions = interactions.merge(
    items[["item_id", "brand", "category"]],
    on="item_id",
    how="left"
)

# -----------------------------
# Popularity Table
# -----------------------------
def build_popularity_table():

    merged = interactions.merge(
        users[["user_id", "segment"]],
        on="user_id",
        how="left"
    )

    agg = (
        merged
        .groupby(["segment", "item_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    agg["views"] = agg.get("view", 0)
    agg["clicks"] = agg.get("click", 0)
    agg["purchases"] = agg.get("purchase", 0)

    agg["smoothed_ctr"] = (agg["clicks"] + 5) / (agg["views"] + 50)
    agg["smoothed_cvr"] = (agg["purchases"] + 2) / (agg["clicks"] + 20)

    agg["popularity_score"] = (
        0.6 * agg["smoothed_ctr"] +
        0.4 * agg["smoothed_cvr"]
    )

    agg = agg.merge(
        items[["item_id", "brand", "category"]],
        on="item_id",
        how="left"
    )

    return agg


POPULARITY_TABLE = build_popularity_table()

# -----------------------------
# Precompute User Affinity (FAST)
# -----------------------------
def build_all_user_affinities(interactions_df):

    user_affinity_map = {}

    grouped = interactions_df.groupby("user_id")

    for user_id, user_data in grouped:

        brand_counts = user_data["brand"].value_counts().to_dict()
        category_counts = user_data["category"].value_counts().to_dict()

        total_brand = sum(brand_counts.values())
        total_cat = sum(category_counts.values())

        brand_affinity = {
            k: v / total_brand
            for k, v in brand_counts.items()
        } if total_brand > 0 else {}

        category_affinity = {
            k: v / total_cat
            for k, v in category_counts.items()
        } if total_cat > 0 else {}

        user_affinity_map[user_id] = (
            brand_affinity,
            category_affinity
        )

    return user_affinity_map


USER_AFFINITY_MAP = build_all_user_affinities(interactions)

# -----------------------------
# MMR Re-ranking
# -----------------------------
def mmr_rerank(df, top_k=10, lambda_div=0.75):

    selected = []
    candidates = df.copy()

    while len(selected) < top_k and not candidates.empty:

        if not selected:
            selected.append(candidates.iloc[0])
            candidates = candidates.iloc[1:]
            continue

        scores = []

        for _, row in candidates.iterrows():

            relevance = row["final_score"]
            penalty = 0

            for sel in selected:
                if row["brand"] == sel["brand"]:
                    penalty += 1
                if row["category"] == sel["category"]:
                    penalty += 0.5

            score = lambda_div * relevance - (1 - lambda_div) * penalty
            scores.append(score)

        candidates["mmr_score"] = scores
        best_idx = candidates["mmr_score"].idxmax()
        selected.append(candidates.loc[best_idx])
        candidates = candidates.drop(best_idx)

    return pd.DataFrame(selected)[
        ["item_id", "brand", "category", "final_score"]
    ]

# -----------------------------
# Hybrid Recommender
# -----------------------------
def recommend_hybrid(user_id, top_k=10):

    user_row = users[users["user_id"] == user_id]

    if user_row.empty:
        return pd.DataFrame()

    segment = user_row.iloc[0]["segment"]

    segment_items = POPULARITY_TABLE[
        POPULARITY_TABLE["segment"] == segment
    ].copy()

    affinity = USER_AFFINITY_MAP.get(user_id)

    # Cold start fallback
    if affinity is None:
        ranked = segment_items.sort_values(
            "popularity_score",
            ascending=False
        )
        ranked["final_score"] = ranked["popularity_score"]
        return ranked.head(top_k)[
            ["item_id", "brand", "category", "final_score"]
        ]

    brand_affinity, category_affinity = affinity

    segment_items["brand_score"] = segment_items["brand"].map(
        brand_affinity
    ).fillna(0)

    segment_items["category_score"] = segment_items["category"].map(
        category_affinity
    ).fillna(0)

    segment_items["affinity_score"] = (
        0.7 * segment_items["brand_score"] +
        0.3 * segment_items["category_score"]
    )

    if segment_items["affinity_score"].max() > 0:
        segment_items["affinity_score"] /= segment_items["affinity_score"].max()

    segment_items["final_score"] = (
        0.7 * segment_items["affinity_score"] +
        0.3 * segment_items["popularity_score"]
    )

    ranked = segment_items.sort_values(
        "final_score",
        ascending=False
    )

    return mmr_rerank(ranked, top_k=top_k)

# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":

    sample_user = users.sample(1)["user_id"].iloc[0]

    print(f"\nRecommendations for {sample_user}")

    recs = recommend_hybrid(sample_user)

    print(
        recs.to_string(index=False)
    )