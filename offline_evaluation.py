import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, norm

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

users_test = pd.read_csv(DATA_DIR / "users_test.csv")
interactions_test = pd.read_csv(DATA_DIR / "interactions_test.csv")
items = pd.read_csv(DATA_DIR / "items.csv")

# =========================================================
# IMPORT YOUR EXISTING RECOMMENDERS (UNCHANGED)
# =========================================================
from personalized_recommender import recommend_hybrid
from popular_recommender import get_top_products_by_segment

# =========================================================
# POPULAR WRAPPER (handles new users_test cleanly)
# =========================================================
def popular_wrapper(user_id, top_k=10):
    user_row = users_test[users_test["user_id"] == user_id]
    if user_row.empty:
        return pd.DataFrame()
    segment = user_row["segment"].iloc[0]
    return get_top_products_by_segment(segment=segment, top_k=top_k)

# =========================================================
# RANDOM BASELINE
# =========================================================
def random_baseline(user_id, top_k=10):
    sample = items.sample(top_k)
    return sample[["item_id", "brand", "category"]]

# =========================================================
# METRICS
# =========================================================
def precision_at_k(rec, rel, k):
    if len(rec) == 0:
        return 0
    return len(set(rec[:k]) & set(rel)) / k

def recall_at_k(rec, rel, k):
    if len(rel) == 0:
        return 0
    return len(set(rec[:k]) & set(rel)) / len(rel)

def hit_rate_at_k(rec, rel, k):
    return int(len(set(rec[:k]) & set(rel)) > 0)

def revenue_weighted_precision(rec, rel, k):
    rec_k = rec[:k]
    hits = list(set(rec_k) & set(rel))
    if len(hits) == 0:
        return 0
    hit_prices = items[items["item_id"].isin(hits)]["price"].sum()
    total_prices = items[items["item_id"].isin(rec_k)]["price"].sum()
    return hit_prices / total_prices if total_prices > 0 else 0

def intra_list_diversity(rec_items):
    if len(rec_items) <= 1:
        return 0
    df = items[items["item_id"].isin(rec_items)]
    brands = df["brand"].tolist()
    categories = df["category"].tolist()
    diversity = 0
    comparisons = 0
    for i in range(len(brands)):
        for j in range(i+1, len(brands)):
            comparisons += 1
            if brands[i] != brands[j]:
                diversity += 1
            if categories[i] != categories[j]:
                diversity += 1
    return diversity / comparisons if comparisons > 0 else 0

# =========================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================
def bootstrap_ci(data, n_boot=1000):
    data = np.array(data)
    if len(data) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

# =========================================================
# SAMPLE SIZE ESTIMATION (No statsmodels needed)
# =========================================================
def estimate_sample_size(p1, p2, alpha=0.05, power=0.8):
    effect = abs(p1 - p2)
    if effect == 0:
        return np.inf
    pooled = (p1 + p2) / 2
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = ((z_alpha * np.sqrt(2 * pooled * (1 - pooled)) +
          z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2) / (effect ** 2)
    return int(n)

# =========================================================
# CORE EVALUATION FUNCTION
# =========================================================
def evaluate_model(recommend_fn, k=10):

    precisions, recalls, hits = [], [], []
    revenue_precisions = []
    diversity_scores = []

    for user_id in users_test["user_id"].unique():

        relevant = interactions_test[
            interactions_test["user_id"] == user_id
        ]["item_id"].tolist()

        if len(relevant) == 0:
            continue

        rec_df = recommend_fn(user_id, top_k=k)

        if rec_df is None or rec_df.empty:
            continue

        rec_items = rec_df["item_id"].tolist()

        precisions.append(precision_at_k(rec_items, relevant, k))
        recalls.append(recall_at_k(rec_items, relevant, k))
        hits.append(hit_rate_at_k(rec_items, relevant, k))
        revenue_precisions.append(
            revenue_weighted_precision(rec_items, relevant, k)
        )
        diversity_scores.append(
            intra_list_diversity(rec_items)
        )

    return {
        "precision_list": precisions,
        "precision": np.mean(precisions) if precisions else 0,
        "recall": np.mean(recalls) if recalls else 0,
        "hit_rate": np.mean(hits) if hits else 0,
        "revenue_precision": np.mean(revenue_precisions) if revenue_precisions else 0,
        "diversity": np.mean(diversity_scores) if diversity_scores else 0
    }

# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":

    print("\nEvaluating Personalized Model...")
    personalized = evaluate_model(recommend_hybrid)

    print("\nEvaluating Popular Model...")
    popular = evaluate_model(popular_wrapper)

    print("\nEvaluating Random Baseline...")
    random_model = evaluate_model(random_baseline)

    p_prec = np.array(personalized["precision_list"])
    pop_prec = np.array(popular["precision_list"])
    rand_prec = np.array(random_model["precision_list"])

    print("\n--- Overall Metrics ---")
    print("Personalized Precision:", personalized["precision"])
    print("Popular Precision:", popular["precision"])
    print("Random Precision:", random_model["precision"])

    if random_model["precision"] > 0:
        print("Lift vs Random:",
              (personalized["precision"] - random_model["precision"]) /
              random_model["precision"])

    if popular["precision"] > 0:
        print("Lift vs Popular:",
              (personalized["precision"] - popular["precision"]) /
              popular["precision"])

    ci_low, ci_high = bootstrap_ci(p_prec)
    print("\n95% CI for Personalized Precision:")
    print(ci_low, ci_high)

    if len(p_prec) == len(pop_prec) and len(p_prec) > 0:
        stat, pval = wilcoxon(p_prec, pop_prec)
        print("\nWilcoxon test p-value:", pval)

    required_n = estimate_sample_size(
        popular["precision"],
        personalized["precision"]
    )

    print("\nEstimated sample size per variant for A/B test:", required_n)