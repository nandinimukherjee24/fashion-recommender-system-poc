import pandas as pd
import numpy as np
from itertools import combinations
from popular_recommender import get_top_products_by_segment

# Load users
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

users = pd.read_csv(DATA_DIR / "users.csv")

# Sample 100 users
sample_users = users.sample(100, random_state=42)["user_id"].tolist()

all_recommendations = {}

print("Running baseline audit...\n")

# Generate recommendations
for user in sample_users:
    recs = get_top_products_by_segment(user, top_n=10)
    all_recommendations[user] = recs["item_id"].tolist()

print("Sample recs:")
print(recs.head())
print("Length:", len(recs))


# -------------------------------
# 1. Overlap Calculation
# -------------------------------
overlaps = []

for u1, u2 in combinations(sample_users, 2):
    set1 = set(all_recommendations[u1])
    set2 = set(all_recommendations[u2])
    overlap = len(set1.intersection(set2)) / 10
    overlaps.append(overlap)

avg_overlap = np.mean(overlaps)

print("===== OVERLAP METRIC =====")
print("Average Top-10 Overlap:", round(avg_overlap, 3))

# -------------------------------
# 2. Brand & Category Diversity
# -------------------------------
all_items = []

for user in sample_users:
    recs = get_top_products_by_segment(user, top_n=10)
    all_items.append(recs)

all_items_df = pd.concat(all_items)

unique_brands = all_items_df["brand"].nunique()
unique_categories = all_items_df["category"].nunique()

print("\n===== DIVERSITY METRICS =====")
print("Unique Brands in 1000 recs:", unique_brands)
print("Unique Categories in 1000 recs:", unique_categories)

print("\nAudit Complete.")
