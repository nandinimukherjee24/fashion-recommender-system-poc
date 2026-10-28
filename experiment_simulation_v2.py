import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from personalized_recommender import recommend_hybrid, users, interactions
from personalized_recommender import POPULARITY_TABLE


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
N_USERS = 2000   # reduce slightly for speed (can increase later)
TOP_K = 10
SEED = 42

np.random.seed(SEED)

# -------------------------------------------------
# PRECOMPUTE USER HISTORIES (MAJOR SPEED BOOST)
# -------------------------------------------------
# Merge brand/category ONCE
interaction_enriched = interactions.merge(
    POPULARITY_TABLE[["item_id", "brand", "category"]],
    on="item_id",
    how="left"
)

# Build dictionary: user_id -> user history
USER_HISTORY = {
    uid: df
    for uid, df in interaction_enriched.groupby("user_id")
}


# -------------------------------------------------
# Popular recommender
# -------------------------------------------------
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


# -------------------------------------------------
# CLICK SIMULATION
# -------------------------------------------------
def simulate_click(row, user_history):

    base_ctr = 0.06

    if user_history is None or user_history.empty:
        return np.random.rand() < base_ctr

    brand_set = set(user_history["brand"])
    category_set = set(user_history["category"])

    prob = base_ctr

    if row["brand"] in brand_set:
        prob += 0.05

    if row["category"] in category_set:
        prob += 0.03

    return np.random.rand() < prob


# -------------------------------------------------
# CONVERSION SIMULATION
# -------------------------------------------------
def simulate_conversion(row, user_history):

    base_cvr = 0.18

    if user_history is None or user_history.empty:
        return np.random.rand() < base_cvr

    brand_set = set(user_history["brand"])
    category_set = set(user_history["category"])

    prob = base_cvr

    if row["brand"] in brand_set:
        prob += 0.05

    if row["category"] in category_set:
        prob += 0.04

    return np.random.rand() < prob


# -------------------------------------------------
# RUN EXPERIMENT
# -------------------------------------------------
def run_experiment():
    sample_users = users["user_id"].values[:N_USERS]

    results = {
        "popular": {"clicks": 0, "conversions": 0, "engagements": 0},
        "personalized": {"clicks": 0, "conversions": 0, "engagements": 0},
    }

    total_impressions = N_USERS * TOP_K

    for i, user_id in enumerate(sample_users):

        # Progress print every 300 users
        if i % 300 == 0:
            print(f"Processed {i} users...")

        user_history = USER_HISTORY.get(user_id, None)

        # ---------- POPULAR ----------
        pop_recs = recommend_popular(user_id, TOP_K)
        user_engagement_pop = 0

        for _, row in pop_recs.iterrows():

            if simulate_click(row, user_history):
                results["popular"]["clicks"] += 1
                user_engagement_pop += 1

                if simulate_conversion(row, user_history):
                    results["popular"]["conversions"] += 1

        results["popular"]["engagements"] += user_engagement_pop

        # ---------- PERSONALIZED ----------
        per_recs = recommend_hybrid(user_id, TOP_K)
        user_engagement_per = 0

        for _, row in per_recs.iterrows():

            if simulate_click(row, user_history):
                results["personalized"]["clicks"] += 1
                user_engagement_per += 1

                if simulate_conversion(row, user_history):
                    results["personalized"]["conversions"] += 1

        results["personalized"]["engagements"] += user_engagement_per

    return results, total_impressions


# -------------------------------------------------
# Z TEST (CORRECTED)
# -------------------------------------------------
def z_test(p1, p2, n):

    pooled = (p1*n + p2*n) / (2*n)
    se = np.sqrt(pooled * (1 - pooled) * (2/n))
    z = (p2 - p1) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return z, p_value


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    results, impressions = run_experiment()

    # CTR
    pop_ctr = results["popular"]["clicks"] / impressions
    per_ctr = results["personalized"]["clicks"] / impressions

    # CVR
    pop_cvr = results["popular"]["conversions"] / max(results["popular"]["clicks"], 1)
    per_cvr = results["personalized"]["conversions"] / max(results["personalized"]["clicks"], 1)

    # Basket size
    pop_basket = results["popular"]["engagements"] / N_USERS
    per_basket = results["personalized"]["engagements"] / N_USERS

    # Stats
    z_ctr, p_ctr = z_test(pop_ctr, per_ctr, impressions)

    print("\n------ FULL BUSINESS METRICS ------")
    print(f"Popular CTR: {pop_ctr:.4f}")
    print(f"Personalized CTR: {per_ctr:.4f}")
    print(f"CTR Uplift: {per_ctr - pop_ctr:.4f}")
    print(f"CTR P-value: {p_ctr:.6f}")

    print("\nPopular CVR: {:.4f}".format(pop_cvr))
    print("Personalized CVR: {:.4f}".format(per_cvr))

    print("\nPopular Avg Basket Size: {:.2f}".format(pop_basket))
    print("Personalized Avg Basket Size: {:.2f}".format(per_basket))


    # ---------------- VISUALIZATION ----------------

    plt.figure()
    plt.bar(["Popular", "Personalized"], [pop_ctr, per_ctr])
    plt.title("CTR Comparison")
    plt.ylabel("CTR")
    plt.show()

    plt.figure()
    plt.bar(["Popular", "Personalized"], [pop_cvr, per_cvr])
    plt.title("CVR Comparison")
    plt.ylabel("CVR")
    plt.show()

    plt.figure()
    plt.bar(["Popular", "Personalized"], [pop_basket, per_basket])
    plt.title("Average Basket Size")
    plt.ylabel("Basket Size")
    plt.show()



