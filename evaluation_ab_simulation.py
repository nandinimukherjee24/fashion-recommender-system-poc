import pandas as pd
import numpy as np
from scipy.stats import norm
from collections import Counter


# -------------------------------------------------
# LOAD DATA (TRAINING ONLY)
# -------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

interactions = pd.read_csv(DATA_DIR / "interactions.csv")
users = pd.read_csv(DATA_DIR / "users.csv")
items = pd.read_csv(DATA_DIR / "items.csv")

interactions = interactions.merge(
    items[["item_id", "brand", "category"]],
    on="item_id",
    how="left"
)

# -------------------------------------------------
# PREPARE GLOBAL POPULARITY
# -------------------------------------------------
popularity = (
    interactions.groupby("item_id")
    .size()
    .reset_index(name="total_interactions")
)

items = items.merge(popularity, on="item_id", how="left")
items["total_interactions"] = items["total_interactions"].fillna(0)

# Normalize popularity
items["popularity_score"] = (
    items["total_interactions"] - items["total_interactions"].min()
) / (
    items["total_interactions"].max() - items["total_interactions"].min()
)

# -------------------------------------------------
# POPULAR RECOMMENDER
# -------------------------------------------------
def get_popular_top_k(k=10):
    return items.sort_values("popularity_score", ascending=False).head(k)


# -------------------------------------------------
# PERSONALIZED RECOMMENDER (HYBRID)
# -------------------------------------------------
def get_personalized_top_k(user_id, k=10):

    user_data = interactions[interactions["user_id"] == user_id]

    if len(user_data) < 3:
        # Cold start fallback
        return get_popular_top_k(k)

    brand_counts = Counter(user_data["brand"])
    category_counts = Counter(user_data["category"])

    total_interactions = len(user_data)

    def compute_affinity(row):
        brand_score = brand_counts[row["brand"]] / total_interactions
        category_score = category_counts[row["category"]] / total_interactions
        return 0.6 * (brand_score + category_score)

    candidate_items = items.copy()

    candidate_items["affinity_score"] = candidate_items.apply(
        compute_affinity, axis=1
    )

    # Hybrid scoring
    candidate_items["final_score"] = (
        0.6 * candidate_items["affinity_score"] +
        0.4 * candidate_items["popularity_score"]
    )

    # Remove already interacted items
    seen_items = set(user_data["item_id"])
    candidate_items = candidate_items[~candidate_items["item_id"].isin(seen_items)]

    return candidate_items.sort_values("final_score", ascending=False).head(k)


# -------------------------------------------------
# CLICK SIMULATION FUNCTION
# -------------------------------------------------
import numpy as np

def simulate_click(row, user_history):
    """
    Realistic click simulation:
    - Base probability
    - Brand affinity boost
    - Category affinity boost
    """

    base_prob = 0.05  # baseline 5%

    if user_history.empty:
        return np.random.rand() < base_prob

    brand_match = row["brand"] in user_history["brand"].values
    category_match = row["category"] in user_history["category"].values

    if brand_match:
        base_prob += 0.08   # +8% lift

    if category_match:
        base_prob += 0.05   # +5% lift

    base_prob = min(base_prob, 0.25)  # cap at 25% realistic CTR

    return np.random.rand() < base_prob

# -------------------------------------------------
# RUN SIMULATION
# -------------------------------------------------
np.random.seed(42)

users_sample = users["user_id"].unique()[:1000]  # simulate on 1000 users

popular_clicks = 0
personalized_clicks = 0
total_impressions = 0

for user_id in users_sample:

    user_history = interactions[interactions["user_id"] == user_id]

    # POPULAR
    popular_recs = get_popular_top_k(10)

    for _, row in popular_recs.iterrows():
        if simulate_click(row, user_history):
            popular_clicks += 1

    # PERSONALIZED
    personalized_recs = get_personalized_top_k(user_id, 10)

    for _, row in personalized_recs.iterrows():
        if simulate_click(row, user_history):
            personalized_clicks += 1

    total_impressions += 10

# -------------------------------------------------
# METRICS
# -------------------------------------------------
popular_ctr = popular_clicks / (len(users_sample) * 10)
personalized_ctr = personalized_clicks / (len(users_sample) * 10)

print("\n----- SIMULATION RESULTS -----")
print(f"Popular CTR: {popular_ctr:.4f}")
print(f"Personalized CTR: {personalized_ctr:.4f}")
print(f"Absolute Uplift: {(personalized_ctr - popular_ctr):.4f}")

# -------------------------------------------------
# STATISTICAL SIGNIFICANCE (Z TEST)
# -------------------------------------------------
p1 = popular_ctr
p2 = personalized_ctr
n1 = len(users_sample) * 10
n2 = len(users_sample) * 10

pooled = (popular_clicks + personalized_clicks) / (n1 + n2)

z = (p2 - p1) / np.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))
p_value = 2 * (1 - norm.cdf(abs(z)))

print(f"Z-score: {z:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("Result is statistically significant")
else:
    print("Result is NOT statistically significant")
