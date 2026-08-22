# Phase 6 Walkthrough - Current Data Pipeline & Inference Layer

This document details the design, API data-acquisition layer, player mapping scheme, and the step-by-step execution workflow of the FPL Analytics In-Season Current Ingestion pipeline.

---

## 1. Why Historical FPL Data Alone is Insufficient
* **Static Nature of Historical Data**: Historical training sets are compiled post-season. During an active season, player prices change daily, ownership ratios shift weekly, and players are traded between clubs.
* **Mid-week Cup Rotations & Injuries**: FPL historical data only captures Premier League matches. It does not log mid-week Champions League, Europa League, FA Cup, or international matches, which are critical for detecting rotation risks and player fatigue.
* **Point-in-Time Decisions**: For upcoming GW $N$, we need to calculate rolling form using GW $N-1$ and earlier, while merging the known fixture difficulties of GW $N$, without exposing the model to the actual results of GW $N$.

---

## 2. Separation of Training vs Inference Data
To protect the integrity of our models and prevent accidental training data corruption, we maintain a strict architectural separation:

```
[Historical Training Set]  ──> data/processed/features_df.parquet     (UNTOUCHED by weekly updates)
[In-Season Inference Set]  ──> data/processed/current_features.parquet  (Overwritten weekly with GW N state)
```

At prediction time, `predict.py` queries `current_features.parquet` first. If it contains the target season and gameweek, it loads it; otherwise, it falls back to `features_df.parquet` (which preserves the weekly backtesting framework).

---

## 3. Data Sources & API Endpoints

### A. FPL API
To update player statuses and calculate rolling form, we fetch raw JSONs from official FPL endpoints:
* **Bootstrap Static** (`/api/bootstrap-static/`): Fetches live player names, codes, current teams, prices, transfer counts, selected ratios, and injury/suspension statuses.
* **Fixtures** (`/api/fixtures/`): Fetches all 380 match fixtures, completed scores, and fixture difficulties.
* **Live Gameweek Stats** (`/api/event/{gw_id}/live/`): Fetches bulk player actual statistics (minutes, points, goals, assists, saves, bonus, expected stats) for all players in a single API call for a completed gameweek.

### B. External Football Data
* **Source**: Scraped match logs from `https://fbref.com` (using standard Python libraries like `soccerdata`).
* **Enrichment File**: `data/raw/matches/external_cup_matches.json`.
* **Details Tracked**: Minutes, starts, goals, assists, and xG/xA for Champions League, Europa League, FA Cup, League Cup, and international matches.
* *Note*: This is stored separately as an enrichment database. The existing production model does not consume these cup features until a future historical retraining phase is executed.

---

## 4. Player Identity Mapping
FPL players are mapped to external IDs using `src/data/player_mapping.py`:
1. **Name Normalization**: Strips accents, converts to lowercase, removes punctuation/hyphens.
2. **FPL Club Filter**: Verifies that players are on the same team to prevent duplicate name matches.
3. **Manual Overrides**: Resolves ambiguous matches using `config/player_mapping_overrides.json`.
4. **Parquet Export**: Outputs the persistent mapping table to `data/processed/player_id_mapping.parquet`.

---

## 5. Point-in-Time Leakage Protection
For a target gameweek $N$:
* **Completed Stats**: Only FPL live data and fixtures from gameweeks $\le N-1$ are compiled.
* **Fixture Details**: Fixture opponent, home/away, and difficulties for GW $N$ are joined (as they are known before kickoff).
* **Leakage Guard**: All raw GW $N$ actuals (minutes, points, goals) are dropped, and mock placeholder target variables are generated.

---

## 6. Weekly Execution Workflow

### STEP 1: Update current weekly data and compute inference features
```bash
python -m src.data.update_current --season 2024-25 --gw 15
```
*Downloads FPL API data up to GW 14, merges with historical parquets, computes rolling averages, and saves the GW 15 inference features to `current_features.parquet`.*

### STEP 2: Generate next-GW predictions using the existing trained LightGBM
```bash
python -m src.models.predict --season 2024-25 --gw 15
```
*Loads `current_features.parquet`, generates predictions and uncertainty metrics, and saves results to `data/results/predictions_2024-25_gw15.csv`.*

### STEP 3: Run squad optimization
```bash
python -m src.optimization.squad_optimizer --season 2024-25 --gw 15
```

### STEP 4: Run transfer optimization
```bash
python -m src.optimization.transfer_optimizer --season 2024-25 --gw 15 --squad data/input/current_squad.json --free-transfers 1
```

---

## 7. Limitations & Future Retraining
* **Un-integrated Cup Features**: Champions League and cup minutes are collected, but not fed into the active LightGBM.
* **Future Training Plan**: Once multiple seasons of cup match minutes are compiled, a new training run will be scheduled to add `minutes_all_comp_last_5` and similar features to the historical training parquet (`features_df.parquet`) and retrain the model.
