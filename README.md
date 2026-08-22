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

### Step 4: Run Production Player Predictions & Forecasts
Runs the production inference CLI to get player predicted points, minutes, playing probability, value rankings, and static-horizon multi-gameweek forecasts:
```bash
# General usage for predicting a specific GW
python -m src.models.predict --season 2024-25 --gw 15 --top 20

# Filter by player position
python -m src.models.predict --season 2024-25 --gw 15 --top 20 --position MID

# Multi-gameweek static-horizon forecast (e.g. 3 GWs)
python -m src.models.predict --season 2024-25 --gw 15 --horizon 3 --top 10

# Force retrain production models
python -m src.models.predict --season 2024-25 --gw 15 --force-retrain
```

### Step 5: Run Model Hyperparameter Tuning
Runs the Optuna hyperparameter optimization script to search for the best LightGBM points regression configuration:
```bash
python -m src.models.tune --trials 30 --metric spearman
```

---

## 10. Phase 3A — Player Prediction & Decision Support
Phase 3A converts the validated modeling structure into a reusable, production-ready prediction and decision support pipeline:

### Workflow Schema:
`Raw FPL data` (Manual Sync)  
→ `build_dataset.py` (Deduplicated parquet structures)  
→ `build_features.py` (Point-in-time rolling statistics and active grids)  
→ `predict.py` (Auto-train on strict cutoff metadata / joblib serialization)  
→ `LightGBM Models` (Points, minutes, and playing probability estimators)  
→ `Static-Horizon Forecasting` (Holds player form constant while updating future fixture context)  
→ `Rankings & Derived Metrics` (Points/£M, position filtering, uncertainty distribution probabilities)  
→ `data/results/*.csv` (Prediction outputs)

*Note: Data ingestion is currently **MANUALLY** updated (Phase 2 syncing process). Automated FPL API ingestion, notifications, and squad optimization are slated for future phases.*

---

## 11. Phase 3B — Model Optimization & Hyperparameter Tuning
Phase 3B explores systematic hyperparameter optimization using Optuna to maximize the model's ranking ability on strictly chronological validation data:

### Validation Splits (Tuning Search vs Evaluation):
* **Tuning Train**: `2016-17` to `2022-23`
* **Tuning Validation**: `2023-24` (Optuna optimizes Spearman Rank Correlation on this season using static holdout)
* **Tuning Test**: `2024-25` (Untouched during search)
* **Out-of-Sample**: `2025-26` (Untouched during search)

### Best Hyperparameters:
```yaml
best_params:
  colsample_bytree: 0.7423
  learning_rate: 0.0254
  max_depth: 9
  min_child_samples: 63
  n_estimators: 250
  num_leaves: 118
  reg_alpha: 0.0063
  reg_lambda: 0.000005
  subsample: 0.7105
```

### Baseline vs Tuned Comparative Metrics (Weekly Expanding Backtest):
* **2023-24 (Validation)**:
  * Baseline: Spearman = `0.6996`, MAE = `0.9023`, Top-20 Acc = `19.3%`
  * Tuned: Spearman = **`0.7047`**, MAE = **`0.8964`**, Top-20 Acc = **`21.1%`**
* **2024-25 (Test)**:
  * Baseline: Spearman = `0.7185`, MAE = `1.0061`, Top-20 Acc = `22.9%`
  * Tuned: Spearman = **`0.7228`**, MAE = **`0.9999`**, Top-20 Acc = **`23.2%`**
* **2025-26 (Out-of-Sample)**:
  * Baseline: Spearman = `0.7332`, MAE = `0.9391`, Top-20 Acc = `18.6%`
  * Tuned: Spearman = **`0.7370`**, MAE = **`0.9385`**, Top-20 Acc = `18.2%`

### Top Feature Importances:
1. `num__minutes_last_1` (**30.51%**): Dominant indicator of playing this week.
2. `num__transfers_in_ratio` (**17.25%**): Market transfers ratio acts as a collective wisdom indicator.
3. `num__ict_index_last_3` (**7.80%**): Short-term combined attacking rating.
4. `num__current_price` (**5.47%**): Market-tier proxy.

*Note: Verification checks confirmed zero target or raw-GW performance leakage.*

---

## 12. Phase 4 — Squad Optimization
Phase 4 implements a mathematical squad optimizer using Integer Linear Programming (ILP) with the PuLP and COIN-OR CBC solver. It reads predictions generated by Phase 3A/3B and selects the mathematically optimal squad satisfying all official FPL constraints:

### Step 6: Run FPL Squad Optimizer
To run the squad optimizer for a specific season and gameweek:
```bash
python -m src.optimization.squad_optimizer --season 2024-25 --gw 15
```

### Constraints Satisfied Simultaneously:
1. **Squad Structure**: Exactly 15 players (2 GKs, 5 DEFs, 5 MIDs, 3 FWDs).
2. **Budget Constraint**: Total squad cost $\le$ £100.0m.
3. **Club Constraint**: Max 3 players from any Premier League team.
4. **Starting XI Structure**: Exactly 11 starters containing exactly 1 GK, $\ge$ 3 DEFs, $\ge$ 1 MID, and $\ge$ 1 FWD.
5. **Captaincy Rules**: Exactly 1 captain (double expected points) and 1 vice-captain, both of whom must be starting players.
6. **Bench Rules**: 4 substitutes ordered by expected value (outfield substitutes sorted descending by expected points).
7. **Scientific Safety**: Strict leakage guards prevent actual future performance metrics from entering the LP solver.

---

## 13. Phase 5 — Weekly Transfer Optimization
Phase 5 implements a weekly transfer optimizer using Integer Linear Programming (ILP) to determine the best single-gameweek transfers (Hold vs Transfer strategies) given an owned current squad. It automatically deducts configurable hit penalties (default 4.0 points per transfer above the free transfers allowance) and compares the net expected points.

### Run FPL Transfer Optimizer
To run the transfer optimizer for a target gameweek:
```bash
python -m src.optimization.transfer_optimizer --season 2024-25 --gw 15 --squad data/input/current_squad.json --free-transfers 1
```

---

## 14. Phase 6 — Current Ingestion & In-Season Updates
Phase 6 adds a live data acquisition layer. It fetches bootstrap static player metadata, fixture lists, and bulk completed gameweek live statistics directly from the official FPL API. It builds virtual current season tables and compiles them with historical parquets to construct next-GW features in `current_features.parquet` without mutating the historical training dataset `features_df.parquet`.

### Run In-Season Ingestion & Features Update
```bash
python -m src.data.update_current --season 2024-25 --gw 15
```

### Complete In-Season Weekly Decision Workflow:
```bash
# 1. Update current player/fixture stats to current_features.parquet
python -m src.data.update_current --season 2024-25 --gw 15

# 2. Run points predictions using trained LightGBM models
python -m src.models.predict --season 2024-25 --gw 15

# 3. Optimize transfer strategy given predictions and current squad
python -m src.optimization.transfer_optimizer --season 2024-25 --gw 15 --squad data/input/current_squad.json --free-transfers 1
```



