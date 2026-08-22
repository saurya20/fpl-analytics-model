# Tasks - Phase 2 Implementation (Completed)

- `[x]` Step 1: Deduplicate raw player-GW-fixture rows in `build_dataset.py`
- `[x]` Step 2: Run dataset builder to regenerate clean parquet files
- `[x]` Step 3: Implement `src/utils/helpers.py` (position standardization & identity check)
- `[x]` Step 4: Run the identity consistency report
- `[x]` Step 5: Implement feature engineering modules
  - `[x]` `src/features/player_features.py`
  - `[x]` `src/features/team_features.py`
  - `[x]` `src/features/fixture_features.py`
  - `[x]` `src/features/build_features.py`
- `[x]` Step 6: Create configuration `config/config.yaml`
- `[x]` Step 7: Build features dataset and run validation script (leak check)
- `[x]` Step 8: Implement model modules
  - `[x]` `src/models/baseline.py`
  - `[x]` `src/models/train.py`
  - `[x]` `src/models/predict.py`
- `[x]` Step 9: Implement evaluation backtest module (`src/evaluation/backtest.py`)
- `[x]` Step 10: Run the baseline and ML model backtests, saving results to `data/results/`
- `[x]` Step 11: Create documentation in `README.md` and walkthrough report

# Tasks - Phase 3A: Production Player Prediction & CLI (Completed)

- `[x]` Step 12: Implement model serialization, auto-training, and metadata saving/loading in `src/models/predict.py`
- `[x]` Step 13: Implement points, minutes, and playing probability predictions in `src/models/predict.py`
- `[x]` Step 14: Implement static-horizon forecasting in `src/models/predict.py`
- `[x]` Step 15: Implement command-line interface (CLI) in `src/models/predict.py`
- `[x]` Step 16: Create test suite `tests/test_predict.py` (with strict cutoff, leakage checks, etc.)
- `[x]` Step 17: Run all tests (both old tests and new predict tests)
- `[x]` Step 18: Execute prediction CLI for a historical gameweek and verify results (numerically and DGW correctness)
- `[x]` Step 19: Demonstrate strict cutoff exclusion of target/future gameweeks using test CLI run
- `[x]` Step 20: Update README.md with Phase 3A documentation and compile walkthrough

# Tasks - Phase 3B: Model Optimization & Hyperparameter Tuning (Completed)

- `[x]` Step 21: Install `optuna` in the virtual environment
- `[x]` Step 22: Implement Optuna candidate search and static holdout evaluation in `src/models/tune.py`
- `[x]` Step 23: Update `backtest.py` to support custom hyperparameters (`custom_params`)
- `[x]` Step 24: Save best tuned configuration to `config/tuned_lightgbm.yaml` and logs to `tuning_results.csv`
- `[x]` Step 25: Implement fresh baseline vs tuned expanding chronological backtests on three seasons (2023-26)
- `[x]` Step 26: Implement Feature Importance analysis (Gain and Split) and save to `feature_importance.csv`
- `[x]` Step 27: Create test suite `tests/test_tuning.py` (split boundary and leakage checks)
- `[x]` Step 28: Run entire test suite (12 tests passed successfully)
- `[x]` Step 29: Compile `docs/phase3b_walkthrough.md` report and update README.md

# Tasks - Phase 4: Squad Optimization (Completed)

- `[x]` Step 30: Install `pulp` library in the virtual environment
- `[x]` Step 31: Add `team` name mappings inside `predict.py` using `teams.parquet`
- `[x]` Step 32: Create `src/optimization/constraints.py` (squad size, position splits, budget limit, club limit, starting XI limits, captaincy rules)
- `[x]` Step 33: Create `src/optimization/objective.py` (starter points + captain bonus + epsilon vice-captain tie-breaker)
- `[x]` Step 34: Create `src/optimization/squad_optimizer.py` (formulating LP, quiet CBC solver, bench outfield ordering, CLI options, saving results)
- `[x]` Step 35: Create test suite `tests/test_optimizer.py` verifying all 25 constraints and BGW/DGW handling
- `[x]` Step 36: Run full test suite (20 tests passed successfully)
# Tasks - Phase 5: Weekly Transfer Optimization (Completed)

- `[x]` Step 38: Create `src/optimization/transfer_optimizer.py` (Hold vs Transfer strategies, FPL budget selling limits, transfer penalty logic)
- `[x]` Step 39: Create test suite `tests/test_transfer_optimizer.py` verifying all 23 transfer constraints and logic checks
- `[x]` Step 40: Run full test suite (26 tests passed successfully)
- `[x]` Step 41: Compile walkthrough documentation `docs/phase5_walkthrough.md` and update `README.md`

# Tasks - Phase 6: Current Ingestion & In-Season Updates (Completed)

- `[x]` Step 42: Create identity mapping script `src/data/player_mapping.py`
- `[x]` Step 43: Create in-season ingestion and update script `src/data/update_current.py`
- `[x]` Step 44: Update prediction loader `predict.py` to seamlessly combine historical training data and current inference features without mutating `features_df.parquet`
- `[x]` Step 45: Create test suite `tests/test_update_current.py` verifying parquets isolation, schema compatibility, and point-in-time constraints
- `[x]` Step 46: Run full test suite (27 tests passed successfully)
- `[x]` Step 47: Create walkthrough documentation `docs/current_data_pipeline_walkthrough.md` and audit report `docs/current_data_audit.md`

# Tasks - Phase 8: Historical Football Context Feature Evaluation & Model Comparison (In Progress)

- `[x]` Step 48: Create configuration `config/context_model_evaluation.yaml`
- `[x]` Step 49: Implement vectorized feature generation script `src/features/generate_historical_context.py` to build `historical_context_features.parquet`
- `[x]` Step 50: Implement evaluation backtest script `src/evaluation/evaluate_context_model.py` comparing Baseline A vs Context B
- `[ ]` Step 51: Run backtests and compile comparison CSVs (comparison, weekly, importance)
- `[ ]` Step 52: Create test suite `tests/test_context_evaluation.py` and run tests
- `[ ]` Step 53: Write final walkthrough documentation `docs/phase8_walkthrough.md`


