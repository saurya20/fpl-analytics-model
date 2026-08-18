# FPL Analytics & Points Prediction System

A scientifically rigorous, point-in-time machine learning system designed to predict expected Fantasy Premier League (FPL) points, optimize squad selections, and provide decision-support for transfers.

---

## 1. Project Objective
The objective is to build a serious FPL analytics model that:
* Estimates expected player points for upcoming gameweeks.
* Predicts expected minutes and playing probabilities (P(60+ minutes)).
* Evaluates value-for-money metrics (Points per £M).
* quantifies prediction uncertainty.
* Guides transfer decisions and squad line-ups using a mathematical backtested baseline.

---

## 2. Data Source & Setup
We use historical FPL data from the **Vaastav Fantasy Premier League GitHub Repository** as our primary raw data source.
* **Vaastav Path**: `../Fantasy-Premier-League/data`
* **Historical Coverage**: 10 completed seasons (2016-17 through 2025-26).
* **Granularity**: Player × Gameweek × Fixture level.

---

## 3. Data-Update Process
* **Current Process (Phase 2)**: **MANUAL**. Data updates are performed manually by syncing the local Vaastav repository and re-running the dataset builder.
* **Future Process (Phase 7)**: **AUTOMATED**. An automatic data pipeline will pull directly from the official FPL API and scrape supplementary data streams (e.g. understat) after every matchday.

---

## 4. Dataset Structure
The dataset builder outputs four primary processed datasets in `data/processed/`:
1. `player_gw.parquet`: Historical player performances for every gameweek-fixture (~253k rows).
2. `players.parquet`: Seasonal player metadata (price, position, name, stable identifier code).
3. `fixtures.parquet`: Match fixture metadata (event, kickoff time, difficulty, teams).
4. `teams.parquet`: Season-level team records and baseline strengths.

---

## 5. Feature Engineering
We enforce **strict chronological windowing** to prevent data leakage. Features for GW N are computed using only information from gameweeks `< N`.

* **Eligible Prediction Population**: Active players only. A player enters the prediction pool starting from their first appearance in FPL in that season.
* **Player Form**: Rolling sums/means of points (last 1, 3, 5, 10 GWs), and regularized points per 90 (points_per_90_last_3/5/10) with minutes played thresholds.
* **Minutes/Starts**: Rolling sums of minutes and starts (proxy starts used for pre-2022-23).
* **Expected Stats (xG, xA, xGI, xGC)**: Preserved as `NaN` for seasons prior to 2022-23 (where the metrics are unavailable), allowing tree models to split on missingness. Imputed with position-season medians for linear models.
* **Season Experience**: Cumulative minutes, starts, and gameweeks played in the current season.
* **Double Gameweeks (DGW)**: Fixtures are aggregated. We construct DGW indicators, min/max/mean difficulties, and summed opponent rolling stats.
* **Point-in-Time Team Strength**: Rolling goals scored, conceded, and clean sheets (last 3, 5 games) of both the player's team and the opponent.

---

## 6. Prediction Targets
Separate prediction models are built for different targets:
* `target_points` (Regression): Expected points in GW N.
* `target_minutes` (Regression): Expected minutes in GW N.
* `target_60_plus_minutes` (Classification): Probability of playing 60+ minutes.

---

## 7. Model Methodology
We compare three rule-based baselines and three ML models:
1. **Position Average Baseline**: Predicts position-season average points.
2. **Last 5-Game Average Baseline**: Predicts average points of the last 5 GWs.
3. **Weighted Recent Form Baseline**: Predicts a weighted average (0.5 * GW-1 + 0.3 * GW-2/3 + 0.2 * GW-4/5).
4. **Ridge Regression**: Linear regularized model (with median imputation and missingness indicators).
5. **Random Forest Regressor**: Non-linear tree baseline.
6. **LightGBM / XGBoost Regressor**: Gradient boosted trees handling missing values natively.

---

## 8. Backtesting Methodology
We implement a rigorous 4-phase chronological backtesting pipeline:
* **Phase 1 (Development / CV)**: seasons 2016-17 to 2022-23.
* **Phase 2 (Validation / Selection)**: 2023-24. We perform weekly expanding window backtests (retraining the model each week on historical data and predicting the next gameweek).
* **Phase 3 (Holdout Testing)**: 2024-25. Used to evaluate final models.
* **Phase 4 (Out-of-Sample)**: 2025-26. Used strictly as a final test of generalization.

**Evaluation Metrics**: MAE, RMSE, R², Weekly Spearman Rank Correlation (player-ranking quality), and Top N overlap (10, 20, 50).

---

## 9. How to Run the Pipeline

### Setup Environment
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Build Processed Datasets
Sync the Vaastav repo, then run:
```bash
python src/data/build_dataset.py
```

### Step 2: Build Features
Generates the player grid, computes rolling features, merges point-in-time statistics, and saves the final parquet:
```bash
python src/features/build_features.py
```

### Step 3: Run Experiments & Backtesting
Runs holdout and expanding-window weekly backtests, saving metrics and predictions under `data/results/`:
```bash
python scratch/run_experiments.py
```

---

## 10. Future Roadmap
* **Phase 3**: Sophisticated ML models (hyperparameter tuning, transfer learning).
* **Phase 4**: Squad Optimization (Integer Linear Programming to pick 15-player squads under budget, position, and club constraints).
* **Phase 5**: Transfer Optimization (multi-period expected points maximization incorporating hit penalties).
* **Phase 6**: Chip Strategy Optimization (Wildcard, Free Hit, Bench Boost, Triple Captain).
* **Phase 7**: Automatic Data updates (FPL API + scraper integration).
