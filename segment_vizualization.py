import pickle
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# Load Precomputed Results
# ---------------------------------
with open("segment_results.pkl", "rb") as f:
    results = pickle.load(f)

segments = list(results.keys())

pop_ctr = []
per_ctr = []
pop_cvr = []
per_cvr = []

for seg in segments:
    data = results[seg]

    pop_ctr.append(
        data["popular"]["clicks"] / max(data["popular"]["impr"], 1)
    )
    per_ctr.append(
        data["personalized"]["clicks"] / max(data["personalized"]["impr"], 1)
    )

    pop_cvr.append(
        data["popular"]["conv"] / max(data["popular"]["clicks"], 1)
    )
    per_cvr.append(
        data["personalized"]["conv"] / max(data["personalized"]["clicks"], 1)
    )

x = np.arange(len(segments))
width = 0.35

# ---------------------------------
# OVERALL METRICS
# ---------------------------------
total_pop_clicks = sum(results[s]["popular"]["clicks"] for s in segments)
total_pop_impr = sum(results[s]["popular"]["impr"] for s in segments)

total_per_clicks = sum(results[s]["personalized"]["clicks"] for s in segments)
total_per_impr = sum(results[s]["personalized"]["impr"] for s in segments)

overall_pop_ctr = total_pop_clicks / total_pop_impr
overall_per_ctr = total_per_clicks / total_per_impr

plt.figure()
plt.bar(["Popular", "Personalized"], 
        [overall_pop_ctr, overall_per_ctr])
plt.ylabel("CTR")
plt.title("Overall CTR Comparison")
plt.show()

# ---------------------------------
# CTR Plot
# ---------------------------------
plt.figure()
plt.bar(x - width/2, pop_ctr, width, label="Popular")
plt.bar(x + width/2, per_ctr, width, label="Personalized")
plt.xticks(x, segments, rotation=30)
plt.ylabel("CTR")
plt.title("CTR Comparison by Segment")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# CVR Plot
# ---------------------------------
plt.figure()
plt.bar(x - width/2, pop_cvr, width, label="Popular")
plt.bar(x + width/2, per_cvr, width, label="Personalized")
plt.xticks(x, segments, rotation=30)
plt.ylabel("CVR")
plt.title("Conversion Rate by Segment")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------
# CTR Uplift Plot (Very Important for Business)
# ---------------------------------
uplift = np.array(per_ctr) - np.array(pop_ctr)

plt.figure()
plt.bar(segments, uplift)
plt.xticks(rotation=30)
plt.ylabel("Absolute CTR Uplift")
plt.title("Personalization Uplift by Segment")
plt.tight_layout()
plt.show()

# ---------------------------------
# REVENUE SIMULATION
# ---------------------------------

avg_order_value = 2500  # adjust if needed

pop_revenue = sum(results[s]["popular"]["conv"] for s in segments) * avg_order_value
per_revenue = sum(results[s]["personalized"]["conv"] for s in segments) * avg_order_value

plt.figure()
plt.bar(["Popular", "Personalized"], 
        [pop_revenue, per_revenue])
plt.ylabel("Simulated Revenue")
plt.title("Revenue Impact Simulation")
plt.show()

print("\nRevenue Uplift:", per_revenue - pop_revenue)