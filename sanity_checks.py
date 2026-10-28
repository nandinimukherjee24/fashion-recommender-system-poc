import pandas as pd
from pathlib import Path

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load datasets
users = pd.read_csv(DATA_DIR / "users.csv")
users_test = pd.read_csv(DATA_DIR / "users_test.csv")

interactions = pd.read_csv(DATA_DIR / "interactions.csv")
interactions_test = pd.read_csv(DATA_DIR / "interactions_test.csv")

print("\n===== USER COUNTS =====")
print("Train Users:", len(users))
print("Test Users:", len(users_test))

print("\n===== INTERACTION COUNTS =====")
print("Train Interactions:", len(interactions))
print("Test Interactions:", len(interactions_test))

print("\n===== INTERACTIONS PER USER (TRAIN) =====")
print(interactions.groupby("user_id").size().describe())

print("\n===== EVENT DISTRIBUTION (TRAIN) =====")
print(interactions["event_type"].value_counts(normalize=True))

print("\n===== SEGMENT DISTRIBUTION (ITEMS) =====")
items = pd.read_csv(DATA_DIR / "items.csv")
print(items["segment"].value_counts())

print("===== USER-ITEM SEGMENT ALIGNMENT CHECK =====")

merged = interactions.merge(
    users[["user_id", "segment"]],
    on="user_id",
    how="left"
)

merged = merged.merge(
    items[["item_id", "segment"]],
    on="item_id",
    how="left",
    suffixes=("_user", "_item")
)

print("Columns after merge:", merged.columns.tolist())

alignment_rate = (
    merged["segment_user"] == merged["segment_item"]
).mean()

print("Segment Alignment Rate:", alignment_rate)
