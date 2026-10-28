import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon, norm

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ============================================================
# LOAD DATA
# ============================================================
users = pd.read_csv(DATA_DIR / "users.csv")
items = pd.read_csv(DATA_DIR / "items.csv")
interactions = pd.read_csv(DATA_DIR / "interactions.csv")

# ============================================================
# TRAIN / HOLDOUT SPLIT (SEEN USERS)
# ============================================================

train_parts = []
holdout_parts = []

for user_id, user_data in interactions.groupby("user_id"):

    if len(user_data) < 5:
        train_parts.append(user_data)
        continue

    train_u, test_u = train_test_split(
        user_data,
        test_size=0.2,
        random_state=42
    )

    train_parts.append(train_u)
    holdout_parts.append(test_u)

interactions_train = pd.concat(train_parts)
interactions_holdout = pd.concat(holdout_parts)

print("Train users:", interactions_train["user_id"].nunique())
print("Holdout users:", interactions_holdout["user_id"].nunique())

# ============================================================
# IMPORT MODELS
# ============================================================

import personalized_recommender
import popular_recommender

from personalized_recommender import recommend_hybrid
from popular_recommender import get_top_products_by_segment

# Inject training interactions
personalized_recommender.interactions = interactions_train
popular_recommender.interactions = interactions_train

# Build popularity tables
personalized_recommender.POPULARITY_TABLE = \
    personalized_recommender.build_popularity_table()

popular_recommender.POPULARITY_TABLE = \
    popular_recommender.build_popularity_table()

# Initialize user affinity (CRITICAL)
personalized_recommender.initialize_user_affinity(
    interactions_train,
    items
)

# ============================================================
# FAST LOOKUPS
# ============================================================

user_holdout_map = (
    interactions_holdout
    .groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)

price_map = dict(zip(items["item_id"], items["price"]))

# ============================================================
# METRICS
# ============================================================

def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / k

def revenue_precision(rec, rel, k):
    rec_k = rec[:k]
    revenue = sum(price_map.get(i, 0) for i in rec_k if i in rel)
    total = sum(price_map.get(i, 0) for i in rec_k)
    return revenue / total if total > 0 else 0

def bootstrap_ci(values, n_boot=300):
    values = np.array(values)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def estimate_sample_size(p1, p2, alpha=0.05, power=0.8):
    effect = abs(p1 - p2)
    if effect == 0:
        return np.inf
    pooled = (p1 + p2) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta)**2 * 2 * pooled * (1 - pooled)) / (effect**2)
    return int(np.ceil(n))

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model_fn, k=10):

    precisions = []
    revenues = []

    users_list = list(user_holdout_map.items())
    total = len(users_list)

    print(f"\nEvaluating {total} users...")
    start = time.time()

    for idx, (user_id, relevant) in enumerate(users_list, 1):

        recs_df = model_fn(user_id, top_k=k)
        if recs_df is None or recs_df.empty:
            continue

        rec_items = recs_df["item_id"].tolist()

        precisions.append(
            precision_at_k(rec_items, relevant, k)
        )

        revenues.append(
            revenue_precision(rec_items, relevant, k)
        )

        if idx % 5000 == 0:
            print(f"Processed {idx}/{total}")

    print("Done in", round(time.time() - start, 2), "sec")

    return np.array(precisions), np.array(revenues)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n=== OFFLINE PERSONALIZATION EVALUATION ===")

    p_prec, p_rev = evaluate(recommend_hybrid)
    pop_prec, pop_rev = evaluate(get_top_products_by_segment)

    print("\n--- Precision@10 ---")
    print("Personalized:", round(p_prec.mean(), 6))
    print("Popular:", round(pop_prec.mean(), 6))

    lift = (p_prec.mean() - pop_prec.mean()) / pop_prec.mean()
    print("Lift vs Popular:", round(lift, 4))

    ci_low, ci_high = bootstrap_ci(p_prec)
    print("95% CI:", round(ci_low,6), "-", round(ci_high,6))

    stat, pval = wilcoxon(p_prec, pop_prec)
    print("Wilcoxon p-value:", round(pval,6))

    required_n = estimate_sample_size(
        p_prec.mean(),
        pop_prec.mean()
    )
    print("Estimated A/B sample size per variant:", required_n)