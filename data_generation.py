import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ======================================================
# PRODUCTION-GRADE DATA GENERATOR
# Popularity vs Personalization Experiment
# ======================================================

np.random.seed(42)

# ======================================================
# PATH SETUP (SAVE TO RECOMMENDER_POC/data/)
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Saving files to:", DATA_DIR)

# ======================================================
# CONFIGURATION
# ======================================================

SEGMENTS = ["luxury", "fast_fashion", "general_retail"]

BRANDS = {
    "luxury": ["Prada","Gucci","Louis Vuitton","Burberry","Dior",
               "Balenciaga","Versace","Fendi","Armani","Saint Laurent"],

    "fast_fashion": ["Zara","H&M","Uniqlo","Forever 21","Mango",
                     "Pull&Bear","Bershka","Stradivarius","Cotton On","Shein"],

    "general_retail": ["Westside","Pantaloons","Reliance Trends",
                       "Lifestyle","Max Fashion","Shoppers Stop",
                       "FabIndia","Global Desi","Allen Solly","Van Heusen"]
}

CATEGORIES = ["Tops","Dresses","Jeans","Shoes","Bags","Jackets"]

# Required volumes
USERS_TRAIN = 4000
USERS_TEST = 1000
ITEMS_PER_SEGMENT = 5000   # 5000 × 3 = 15000 total
INTERACTIONS_PER_USER = 20

# Base funnel probabilities
BASE_CTR = {
    "luxury": 0.03,
    "fast_fashion": 0.10,
    "general_retail": 0.06
}

BASE_CVR = {
    "luxury": 0.08,
    "fast_fashion": 0.15,
    "general_retail": 0.10
}

# ======================================================
# USER GENERATION
# ======================================================

def generate_users(n_per_segment, prefix):
    users = []
    uid = 1

    for segment in SEGMENTS:
        for _ in range(n_per_segment):
            users.append({
                "user_id": f"{prefix}U{uid}",
                "segment": segment,
                "preferred_brand": np.random.choice(BRANDS[segment]),
                "preferred_category": np.random.choice(CATEGORIES)
            })
            uid += 1

    df = pd.DataFrame(users)
    return df.sample(frac=1).reset_index(drop=True)

# ======================================================
# ITEM GENERATION (Shared Across Train & Test)
# ======================================================

def generate_items():
    items = []
    iid = 1

    for segment in SEGMENTS:
        for _ in range(ITEMS_PER_SEGMENT):
            items.append({
                "item_id": f"I{iid}",
                "segment": segment,
                "brand": np.random.choice(BRANDS[segment]),
                "category": np.random.choice(CATEGORIES),
                "price": np.random.randint(500, 300000)
            })
            iid += 1

    df = pd.DataFrame(items)
    return df.sample(frac=1).reset_index(drop=True)

# ======================================================
# INTERACTION GENERATION (Behaviorally Biased)
# ======================================================

def generate_interactions(users_df, items_df, days_back):

    interactions = []
    interaction_id = 1
    start_date = datetime.now() - timedelta(days=days_back)

    items_by_segment = {
        seg: items_df[items_df["segment"] == seg]
        for seg in SEGMENTS
    }

    for _, user in users_df.iterrows():

        segment = user["segment"]
        user_items = items_by_segment[segment]

        sampled_items = user_items.sample(INTERACTIONS_PER_USER)

        for _, item in sampled_items.iterrows():

            impressions = np.random.randint(1, 10)

            brand_match = item["brand"] == user["preferred_brand"]
            category_match = item["category"] == user["preferred_category"]

            # ---- CLICK PROBABILITY ----
            ctr = BASE_CTR[segment]

            if brand_match and category_match:
                ctr *= 4
            elif brand_match:
                ctr *= 3
            elif category_match:
                ctr *= 1.5

            clicked = np.random.rand() < ctr

            # ---- PURCHASE PROBABILITY ----
            purchased = False
            if clicked:
                cvr = BASE_CVR[segment]

                if brand_match:
                    cvr *= 2
                if category_match:
                    cvr *= 1.3

                purchased = np.random.rand() < cvr

            # ---- EVENT TYPE LOGIC (STRICT FUNNEL) ----
            if purchased:
                event_type = "purchase"
                clicks = 1
            elif clicked:
                event_type = "click"
                clicks = 1
            else:
                event_type = "view"
                clicks = 0

            interactions.append({
                "interaction_id": f"T{interaction_id}",
                "user_id": user["user_id"],
                "item_id": item["item_id"],
                "segment": segment,
                "event_type": event_type,
                "impressions": impressions,
                "clicks": clicks,
                "platform": np.random.choice(["web","app"]),
                "timestamp": start_date + timedelta(days=np.random.randint(0, days_back))
            })

            interaction_id += 1

    df = pd.DataFrame(interactions)
    return df.sample(frac=1).reset_index(drop=True)

# ======================================================
# EXECUTION
# ======================================================

print("Generating items...")
items_df = generate_items()

print("Generating train users...")
train_users_df = generate_users(USERS_TRAIN, "TR")

print("Generating test users...")
test_users_df = generate_users(USERS_TEST, "TE")

print("Generating train interactions...")
train_interactions_df = generate_interactions(train_users_df, items_df, 365)

print("Generating test interactions...")
test_interactions_df = generate_interactions(test_users_df, items_df, 90)

# Save files
train_users_df.to_csv(DATA_DIR / "users.csv", index=False)
test_users_df.to_csv(DATA_DIR / "users_test.csv", index=False)
items_df.to_csv(DATA_DIR / "items.csv", index=False)
train_interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)
test_interactions_df.to_csv(DATA_DIR / "interactions_test.csv", index=False)

print("\nDataset generation complete.")
print("Files saved in:", DATA_DIR)
print("Files:", list(DATA_DIR.iterdir()))

\


