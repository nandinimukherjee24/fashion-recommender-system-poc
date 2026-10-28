

import pandas as pd
from pathlib import Path
import numpy as np

# -------------------------------
# Resolve Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------
# Load Data
# -------------------------------
users = pd.read_csv(DATA_DIR / "users.csv")
items = pd.read_csv(DATA_DIR / "items.csv")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")

# -------------------------------
# Build Popularity Table
# -------------------------------
def build_popularity_table():

    # Merge user segment into interactions
    merged = interactions.merge(
        users[["user_id", "segment"]],
        on="user_id",
        how="left"
    )

    # Aggregate interactions
    agg = (
        merged
        .groupby(["segment", "item_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all columns exist
    for col in ["view", "click", "purchase"]:
        if col not in agg.columns:
            agg[col] = 0

    # Smoothed CTR (prevents extreme ratios)
    agg["smoothed_ctr"] = (
        (agg["click"] + 5) / (agg["view"] + 20)
    )

    # Smoothed CVR
    agg["smoothed_cvr"] = (
        (agg["purchase"] + 2) / (agg["click"] + 10)
    )

    # Weighted popularity score
    agg["popularity_score"] = (
        0.4 * agg["smoothed_ctr"] +
        0.4 * agg["smoothed_cvr"] +
        0.2 * np.log1p(agg["view"])
    )

    # Attach item metadata
    agg = agg.merge(
        items[["item_id", "brand", "category", "segment"]],
        on="item_id",
        how="left"
    )

    return agg


POPULARITY_TABLE = build_popularity_table()


# -------------------------------
# Recommendation Function
# -------------------------------
def get_top_products_by_segment(user_id=None, segment=None, top_k=10):

    if segment is None:
        if user_id is None:
            raise ValueError("Provide either user_id or segment")

        user_row = users[users["user_id"] == user_id]

        if user_row.empty:
            raise ValueError("User not found")

        segment = user_row["segment"].iloc[0]

    recs = (
        POPULARITY_TABLE[
            POPULARITY_TABLE["segment_x"] == segment
        ]
        .sort_values("popularity_score", ascending=False)
        .head(top_k)
    )

    return recs[[
        "item_id",
        "brand",
        "category",
        "smoothed_ctr",
        "smoothed_cvr",
        "popularity_score"
    ]]


# -------------------------------
# Debug Run
# -------------------------------
if __name__ == "__main__":

    for seg in ["luxury", "fast_fashion", "general_retail"]:
        print(f"\nTop products for {seg.upper()} segment")
        print(get_top_products_by_segment(segment=seg))
