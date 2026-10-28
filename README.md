# Fashion Recommender System POC

A segment-aware fashion recommendation framework designed to support luxury fashion, fast fashion, and general retail using popularity-based ranking, hybrid personalization, and cold-start handling.

## Overview

This proof of concept explores how a unified recommendation framework can adapt to different retail segments using synthetic user, product, and interaction data.

The system combines two recommendation pipelines:

1. **Popularity-Based Recommendations:** Ranks products using aggregate engagement and purchase signals within each segment.
2. **Personalized Recommendations:** Adjusts popularity scores using a user's preferred brand and category.

The project also implements fallback strategies for users and products with limited interaction history.

## Business Problem

Fashion retailers serve customers with different purchasing patterns. Luxury shoppers may prioritize brands and make fewer purchases, fast-fashion shoppers may respond more strongly to trends, and general retail shoppers may exhibit more balanced behavior.

A single recommendation framework should accommodate these differences without requiring a completely separate algorithm for every segment.

## Project Objectives

- Build a unified framework for three fashion segments.
- Generate synthetic user, item, and interaction data.
- Establish a popularity-based baseline.
- Implement brand- and category-aware personalization.
- Handle new users and new products with limited historical data.
- Evaluate recommendation quality on unseen test users.
- Interpret performance differences and identify opportunities for improvement.

## System Architecture

### 1. Synthetic Data Model

The proposed data model contains three core tables:

- **Users:** User ID, preferred category, preferred brand, average spend, and segment.
- **Items:** Item ID, brand, category, price, and segment.
- **Interactions:** Interaction ID, user ID, item ID, event type, timestamp, platform, impressions, clicks, CTR, and rating.

The intended schema includes approximately 2,000 users, 2,000 items, and 2,000–3,000 interactions. These are design targets; actual generated row counts should be verified against the implementation.

### 2. Popularity-Based Pipeline

The baseline aggregates product engagement within each segment using:

- Total clicks
- Total purchases
- Average click-through rate (CTR)

The proposed score is:

`score = 0.4 × clicks + 0.4 × purchases + 0.2 × average CTR`

Products are ranked within their segment, producing a non-personalized recommendation list.

### 3. Personalized Pipeline

The hybrid model starts with the popularity score and applies preference boosts:

`final_score = base_score + 0.3 × category_match + 0.5 × brand_match`

The category and brand indicators equal 1 when the corresponding product attribute matches the user's preference, and 0 otherwise.

This creates a transparent ranking method that combines aggregate popularity with user-level preferences.

### 4. Cold-Start Handling

**New users:** Use segment-level popularity when user interaction history is unavailable.

**Existing users with no usable history:** Fall back to popular products in their segment and exclude previously seen products when the required history is available.

**New products:** Estimate an initial score from historical brand and category performance within the same segment:

`new_item_score = 0.6 × brand_score + 0.4 × category_score`

These are lightweight metadata-based fallback strategies intended for a proof of concept.

## Evaluation

The reported evaluation uses unseen test users and compares the popularity baseline with the hybrid personalized recommender.

| Metric | Popular baseline | Personalized model |
|---|---:|---:|
| Precision@10 | 0.003982 | 0.002655 |
| Relative lift | — | -33% |

The reported results show that the popularity baseline outperformed the personalized model on the current test set.

This suggests that segment-level aggregation can be a strong baseline when individual user histories are sparse. The personalization layer provides a foundation for further development, but its performance should not be assumed to improve without additional validation.

**Evaluation note:** These figures are reported project results and should be checked against the saved test outputs before being treated as independently reproduced results.

## Key Design Decisions

- Use synthetic data to demonstrate the framework without claiming access to production retail data.
- Keep the core ranking logic consistent across retail segments.
- Apply segment filtering to reduce cross-segment contamination.
- Prefer interpretable scoring rules for an initial proof of concept.
- Include cold-start fallbacks so the system can return recommendations with limited historical data.
- Treat the popularity model as a baseline that future personalization approaches must outperform.

## Limitations

- Synthetic interactions may not reflect actual customer behavior.
- CTR is estimated from simulated exposure and engagement assumptions rather than observed production traffic.
- The current personalization approach uses explicit brand and category preferences rather than collaborative filtering or learned embeddings.
- Precision@10 depends on how relevant items are defined in the test data.
- Further evaluation is needed before drawing conclusions about real-world recommendation quality.

## Future Improvements

- Incorporate recency-weighted popularity scores.
- Explore collaborative filtering and user-item interaction models.
- Improve personalized ranking with price sensitivity and behavioral features.
- Evaluate segment-specific ranking weights.
- Conduct controlled offline experiments and, where feasible, online A/B testing.
- Monitor coverage, diversity, novelty, and segment-level performance alongside precision.

## Tools and Technologies

Python, Pandas, NumPy, and the libraries actually used in the implementation.

## Repository Structure

- `notebooks/fashion_recommender_poc.ipynb` — notebook implementation.
- `data/` — optional synthetic data samples.
- `docs/` — optional supporting documentation.
- `requirements.txt` — dependencies for reproduction.

## Disclaimer

This repository presents a proof of concept based on synthetic data and simplified recommendation logic. It is not a production recommendation system, and its reported metrics should not be interpreted as evidence of performance on live retail traffic.
