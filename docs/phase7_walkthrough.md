# Phase 7 Walkthrough - Current Football Context & Player Workload Features

This document explains the design, calculated rolling workload/congestion metrics, mapping rules, and execution steps for Phase 7.

---

## 1. Why FPL-Only Data is Insufficient for Current Decisions
* **The Mid-week Cup Gap**: FPL only monitors Premier League games. If a player plays 90 minutes in a Champions League match on Wednesday, travel and fatigue are not represented in FPL data, leading to incorrect assumptions about their starting status on Saturday.
* **Fixture Congestion**: Teams competing in Europe play every 3-4 days. This optimizer captures team-level match counts and rest days to reflect player rotation risks.

---

## 2. In-Season Workload & Fixture Congestion Features

We computed **21 new features** categorized into two groups:

### A. Player Workload Features (15 columns)
* `pl_minutes_last_14d`
* `external_minutes_last_7d`, `external_minutes_last_14d`, `external_minutes_last_21d`
* `total_minutes_last_14d`
* `external_appearances_last_7d`, `external_appearances_last_14d`, `external_appearances_last_21d`
* `external_starts_last_14d`
* `external_goals_last_14d`
* `external_assists_last_14d`
* `total_competitive_minutes_last_7d`, `total_competitive_minutes_last_14d`, `total_competitive_minutes_last_21d`
* `days_since_player_last_match`

### B. Fixture Congestion & Rest Features (6 columns)
* `team_matches_last_7d`
* `team_matches_last_14d`
* `team_matches_next_7d`
* `days_since_team_last_match`
* `days_until_next_match`
* `fixture_congestion_score`

---

## 3. Player ID Mapping & overrides
1. **Normalization**: Strips accents, maps lowercase strings, maps names and Premier League teams.
2. **Mapping Parquet**: Generates `data/processed/player_id_mapping.parquet` mapping FPL codes to FBref IDs.
3. **Override Config**: Checks `config/player_mapping_overrides.json` to resolve ambiguous matches. Unmatched FPL codes default safely to unmatched placeholder IDs (`ext_{fpl_code}`) to prevent silent errors.

---

## 4. Point-in-Time Cutoff & Leakage Guards
For an upcoming GW $N$ prediction:
* **Allowed**: Completed matches (PL + external) kickoff dates strictly before the FPL gameweek deadline cutoff $D_N$. GW $N$ fixture schedule and opponent difficulties (known before deadline).
* **Forbidden**: GW $N$ actual performance stats, points, minutes, goals, assists, and any match kickoff dates $\ge D_N$ (e.g. Champions League matches scheduled after FPL deadline).
* **Leakage Verification**: The test suite guarantees no future match info enters the features.

---

## 5. Model Schema Compatibility & Retraining Decision
* **Compatibility**: Appending these columns to `current_features.parquet` is safe because `predict.py` extracts only the 103 active columns defined in `config/config.yaml` categories, ignoring workload features at inference.
* **Why the model is not retrained**: To prevent breaking prediction or mutating `features_df.parquet`, we must collect historical versions of these cup/workload metrics across all prior seasons before retraining. The next phase will execute a chronological validation comparison.

---

## 6. Weekly Execution Workflow

### Step 1: Fetch live FPL stats, external cup data, and compile features
```bash
python -m src.data.update_current --season 2024-25 --gw 15
```
*Saves combined next-GW features to `current_features.parquet` and standalone workloads to `current_player_workload.parquet`.*

### Step 2: Generate player predictions
```bash
python -m src.models.predict --season 2024-25 --gw 15
```
*Outputs predictions CSV to `data/results/predictions_2024-25_gw15.csv`.*

### Step 3: Optimize transfers
```bash
python -m src.optimization.transfer_optimizer --season 2024-25 --gw 15 --squad data/input/current_squad.json --free-transfers 1
```
*Generates Hold vs Transfer recommended strategy.*
