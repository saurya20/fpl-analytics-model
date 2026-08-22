# Phase 8 Walkthrough: Historical Football Context Feature Evaluation & Model Comparison

This document provides a comprehensive report of the methodology, verification steps, quantitative results, and final production recommendation for Phase 8.

---

## 1. Objectives

The primary objective of Phase 8 is to scientifically determine whether adding historical football-context features (player workload and fixture congestion) improves the performance of the LightGBM player prediction model.

We compare two models using identical hyperparameters and hyperparameters tuning:
*   **MODEL A (BASELINE)**: Only includes baseline player features (rolling minutes, rolling points, prices, transfers, fixture difficulty, and team strengths).
*   **MODEL B (CONTEXT-ENRICHED)**: Includes all baseline features plus 21 additional workload and fixture-context features (Premier League minutes/matches rolling window, external cup match details, days since last match, team matches in last/next 7/14 days, and fixture congestion score).

The comparison evaluates:
1.  **Points Prediction** (regression on `target_points`)
2.  **Minutes Prediction** (regression on `target_minutes`)
3.  **Playing 60+ Minutes Probability** (binary classification on `target_60_plus_minutes`)

---

## 2. Data Sources & Coverage

### Included Seasons and Competitions
*   **Premier League (PL)**: Fully covered from **2018-19 to 2025-26** seasons. Workload and congestion stats are dynamically compiled from historical FPL parquet files.
*   **External Competitions**: The pipeline was tested with a raw JSON log containing subset records for Bukayo Saka in Champions League fixtures.

### Excluded Competitions & Limitations
Champions League, Europa League, Conference League, FA Cup, EFL Cup, and International matches are documented as **excluded** for all other players. Complete historical player-match logs for these competitions are not present offline in the repository and cannot be scraped dynamically without hitting strict FBref rate limits (20 requests/minute).

---

## 3. Player Identity Mapping & Leakage Prevention

### Identity Mapping
We build a player mapping table (`player_id_mapping.parquet`) to link FPL `player_code` to FBref `fbref_id`.
*   **Confidence levels**: Only mappings marked as `exact` (highest confidence) or verified overrides are utilized.
*   **Collision Warnings**: If a mapping is ambiguous or an `fbref_id` maps to multiple FPL codes, the mapping pipeline halts and prints:
    `"Player identity collision detected: ..."` to avoid silent mapping errors.

### Point-in-Time & Leakage Prevention Constraints
To prevent chronological data leakage:
1.  **Cutoff Respect**: Only matches occurring **before** the gameweek deadline cutoff time are included in the rolling window stats.
2.  **Target GW Exclusion**: Target gameweek kickoff stats are strictly excluded.
3.  **File Preservation**: The original `features_df.parquet` was kept byte-for-byte unchanged (verified via SHA-256 hash checking) to ensure validation integrity.
4.  **No Random Splitting**: expanding-window splits are partitioned strictly along chronological boundaries (seasons/gameweeks).

---

## 4. Quantitative Results & Evaluation

The backtest pipeline was run chronologically across three expanding splits:
*   **Validation Season**: 2023-24 (trained on 2016-17 to 2022-23)
*   **Test Season**: 2024-25 (trained on 2016-17 to 2023-24)
*   **Out-of-sample (OOS) Season**: 2025-26 (trained on 2016-17 to 2024-25)

### Points Prediction Performance (Regression)

| Season | Model | MAE | RMSE | $R^2$ | Spearman Rank | Top 10% Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2023-24 (Val)** | **Model A (Baseline)** | **0.898713** | 1.883101 | 0.352550 | **0.704127** | 0.147368 |
| | **Model B (Context)** | 0.899622 | **1.882726** | **0.352809** | 0.704115 | **0.157895** |
| **2024-25 (Test)**| **Model A (Baseline)** | 1.000939 | 1.983441 | 0.346809 | **0.721602** | **0.194737** |
| | **Model B (Context)** | **1.000401** | **1.982750** | **0.347264** | 0.721367 | 0.181579 |
| **2025-26 (OOS)** | **Model A (Baseline)** | 0.939671 | 1.888632 | 0.365909 | 0.736633 | **0.118421** |
| | **Model B (Context)** | **0.939459** | **1.888318** | **0.366120** | **0.736917** | **0.118421** |

### Playing 60+ Minutes Performance (Classification)

| Season | Model | ROC-AUC | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **2023-24 (Val)** | **Model A (Baseline)** | 0.954575 | 0.897901 | 0.801546 |
| | **Model B (Context)** | **0.954794** | **0.899013** | **0.803617** |
| **2024-25 (Test)**| **Model A (Baseline)** | 0.946389 | 0.882170 | **0.791538** |
| | **Model B (Context)** | **0.946903** | **0.882460** | 0.791409 |
| **2025-26 (OOS)** | **Model A (Baseline)** | 0.954828 | 0.896527 | 0.801931 |
| | **Model B (Context)** | **0.955029** | **0.897099** | **0.802910** |

---

## 5. Statistical Significance Testing

To verify whether the differences in weekly Spearman rank correlations are statistically significant, we conducted a Wilcoxon signed-rank test on weekly points Spearman rank correlation differences between Model B and Model A:

*   **Mean Spearman Difference**: $+0.000012$
*   **Median Spearman Difference**: $-0.000144$
*   **Weekly Win Rate (Model B)**: $47.37\%$
*   **Wilcoxon $p$-value**: **$0.956032$**

**Conclusion**: The $p$-value is significantly higher than the $0.05$ threshold, confirming that the performance difference between Model A and Model B is **not statistically significant** and represents minor statistical noise.

---

## 6. Ablation Analysis (Validation Season 2023-24)

To understand the individual contributions of workload and fixture congestion features, we evaluated models containing sub-selections of these features:

| Model Group | Spearman Correlation | Top 10% Overlap |
| :--- | :--- | :--- |
| **Baseline (Model A)** | 0.704127 | 0.147368 |
| **Baseline + Workload** | 0.704301 | **0.157895** |
| **Baseline + Fixture Congestion** | **0.704384** | 0.152632 |
| **Baseline + Both (Model B)** | 0.704115 | **0.157895** |

Adding Workload or Fixture Congestion features in isolation yields a very minor improvement in validation metrics, but combining them in Model B dilutes this effect, leaving performance virtually identical to the Baseline.

---

## 7. Feature Importance Analysis

Below are the top 15 features in Model B ranked by LightGBM Gain Importance:

1.  `transfers_in_ratio` (Gain: $1.93 \times 10^6$)
2.  `total_competitive_minutes_last_21d` (Gain: $1.57 \times 10^6$) — *New workload feature*
3.  `minutes_last_1` (Gain: $5.51 \times 10^6$)
4.  `current_price` (Gain: $4.19 \times 10^5$)
5.  `pl_minutes_last_14d` (Gain: $3.97 \times 10^5$) — *New workload feature*
6.  `transfers_balance_ratio` (Gain: $2.76 \times 10^5$)
7.  `is_double_gw` (Gain: $2.14 \times 10^5$)
8.  `selected_ratio` (Gain: $2.05 \times 10^5$)
9.  `minutes_last_3` (Gain: $1.78 \times 10^5$)
10. `transfers_out_ratio` (Gain: $1.25 \times 10^5$)

### Workload & Fixture Congestion Features Ranking
*   `total_competitive_minutes_last_21d` is highly ranked because it captures domestic Premier League workload.
*   All `external_` features (e.g. `external_minutes_last_14d`, `external_goals_last_14d`) have **exactly 0 gain/split importance** because cup match data is heavily missing (100% missingness offline) for the general player population.

---

## 8. Production Recommendation

We **strongly recommend retaining MODEL A (BASELINE)** and rejecting the replacement of the production model with Model B.

### Rationale
1.  **No Material Points Performance Improvement**: Validation, Test, and Out-of-sample points spearman rank correlations are virtually unchanged.
2.  **No Statistical Significance**: The Wilcoxon signed-rank test $p$-value is $0.956$, showing the differences are statistical noise.
3.  **Cup Match Missingness**: Offline cup match data is unavailable for the general player pool, causing `external_` features to be 100% default-filled in practice.
4.  **Operational Complexity**: Adding Model B would introduce complex runtime dependencies to link external player log mappings in the production prediction pipeline, which is not justified given the lack of predictive gains.
