import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.train import build_pipeline
from src.models.baseline import LastNGamesAverageRegressor, WeightedRecentFormRegressor, PositionSeasonAverageRegressor
from src.models.predict import generate_predictions_with_uncertainty, rank_players_by_value

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def calculate_evaluation_metrics(df: pd.DataFrame, pred_col: str, target_col: str) -> Dict[str, Any]:
    """
    Calculate FPL evaluation metrics:
    - MAE, RMSE, R2
    - Average Weekly Spearman Rank Correlation (crucial)
    - Top N Overlap Accuracy (N=10, 20, 50)
    """
    y_true = df[target_col].values
    y_pred = df[pred_col].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 1. Average Weekly Spearman Rank Correlation
    # We group by (season, gw) and calculate Spearman rank correlation
    weekly_corrs = []
    
    # Also calculate Top N accuracies
    top_10_overlaps = []
    top_20_overlaps = []
    top_50_overlaps = []
    
    for (season, gw), group in df.groupby(["season", "gw"]):
        if len(group) < 10:
            continue
            
        t_true = group[target_col].values
        t_pred = group[pred_col].values
        
        # Spearman correlation (handle constant values)
        if np.all(t_pred == t_pred[0]) or np.all(t_true == t_true[0]):
            weekly_corrs.append(0.0)
        else:
            corr, _ = spearmanr(t_pred, t_true)
            if not np.isnan(corr):
                weekly_corrs.append(corr)
                
        # Top N Overlaps
        # Rank players by actual and predicted points
        group_sorted_pred = group.sort_values(pred_col, ascending=False)
        group_sorted_true = group.sort_values(target_col, ascending=False)
        
        def get_overlap(n: int) -> float:
            pred_top_n = set(group_sorted_pred.head(n)["code"].values)
            true_top_n = set(group_sorted_true.head(n)["code"].values)
            return len(pred_top_n.intersection(true_top_n)) / float(n)
            
        top_10_overlaps.append(get_overlap(10))
        top_20_overlaps.append(get_overlap(20))
        top_50_overlaps.append(get_overlap(50))
        
    avg_spearman = np.mean(weekly_corrs) if weekly_corrs else 0.0
    avg_top_10 = np.mean(top_10_overlaps) if top_10_overlaps else 0.0
    avg_top_20 = np.mean(top_20_overlaps) if top_20_overlaps else 0.0
    avg_top_50 = np.mean(top_50_overlaps) if top_50_overlaps else 0.0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman_rank_corr": float(avg_spearman),
        "top_10_accuracy": float(avg_top_10),
        "top_20_accuracy": float(avg_top_20),
        "top_50_accuracy": float(avg_top_50)
    }


def run_season_holdout_backtest(
    features_df: pd.DataFrame,
    train_seasons: List[str],
    test_season: str,
    model_name: str,
    experiment_name: str,
    custom_params: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train a model on training seasons and test on a single holdout season.
    Very fast, useful for validation and quick baseline checks.
    """
    config = load_config()
    
    # Get all features
    numeric_features = []
    for cat in ["player_base_rolling", "player_expected", "player_derived", "player_value_ownership", "player_experience", "fixture_aggregates", "team_rolling", "team_static"]:
        numeric_features.extend(config["features"][cat])
        
    categorical_features = ["position"]
    target_col = config["targets"]["points"]
    
    # Split train and test
    train_data = features_df[features_df["season"].isin(train_seasons)]
    test_data = features_df[features_df["season"] == test_season]
    
    # Fill target NaNs (asserted 0 NaNs but safe)
    train_data = train_data.dropna(subset=[target_col])
    test_data = test_data.dropna(subset=[target_col])
    
    X_train = train_data[numeric_features + categorical_features]
    y_train = train_data[target_col].values
    
    X_test = test_data[numeric_features + categorical_features]
    y_test = test_data[target_col].values
    
    print(f"Training {model_name} | Train seasons: {train_seasons} ({len(train_data):,} rows)")
    print(f"Testing on holdout season: {test_season} ({len(test_data):,} rows)")
    
    # Train model
    if model_name in ["last_5_avg", "weighted_form", "position_avg"]:
        # Baselines
        if model_name == "last_5_avg":
            model = LastNGamesAverageRegressor(n=5)
        elif model_name == "weighted_form":
            model = WeightedRecentFormRegressor()
        else:
            model = PositionSeasonAverageRegressor()
        model.fit(X_train, y_train)
    else:
        # ML pipeline
        model = build_pipeline(model_name, numeric_features, categorical_features, task_type="regression")
        if custom_params and model_name == "lightgbm":
            model.set_params(**{f"model__{k}": v for k, v in custom_params.items()})
        model.fit(X_train, y_train)
        
    # Predict
    # For baseline models, use a dummy RMSE of 2.5. For ML pipelines, we can compute train RMSE or default.
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    
    # Generate predictions
    predictions = generate_predictions_with_uncertainty(model, X_test, validation_rmse=train_rmse)
    
    # Combine with metadata and actuals
    results_df = test_data[["season", "gw", "code", "name", "position", "current_price", target_col]].copy()
    results_df["predicted_points"] = predictions["predicted_points"].values
    results_df["uncertainty_std"] = predictions["uncertainty_std"].values
    results_df["prob_5_plus"] = predictions["prob_5_plus"].values
    results_df["prob_10_plus"] = predictions["prob_10_plus"].values
    
    # Rank
    results_df = rank_players_by_value(results_df, predictions)
    
    # Evaluate
    metrics = calculate_evaluation_metrics(results_df, "predicted_points", target_col)
    metrics["model_name"] = model_name
    metrics["train_rmse"] = float(train_rmse)
    
    return metrics, results_df


def run_weekly_expanding_backtest(
    features_df: pd.DataFrame,
    train_seasons: List[str],
    test_season: str,
    model_name: str,
    experiment_name: str,
    retrain_interval: int = 1,
    custom_params: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Expanded/Rolling Window Backtester:
    Simulates the FPL season gameweek by gameweek.
    For each GW N in the test season:
      1. Filter training data strictly to gameweeks < N (prior seasons + current season up to N-1).
      2. Retrain the model (if retrain_interval allows).
      3. Generate predictions for GW N.
      4. Save and repeat.
    """
    config = load_config()
    
    numeric_features = []
    for cat in ["player_base_rolling", "player_expected", "player_derived", "player_value_ownership", "player_experience", "fixture_aggregates", "team_rolling", "team_static"]:
        numeric_features.extend(config["features"][cat])
        
    categorical_features = ["position"]
    target_col = config["targets"]["points"]
    
    test_season_data = features_df[features_df["season"] == test_season]
    
    gws = sorted(test_season_data["gw"].unique())
    print(f"\nRunning Weekly Expanding Backtest on {test_season} ({len(gws)} gameweeks) using {model_name}...")
    
    all_gw_preds = []
    current_model = None
    last_train_gw = -1
    validation_rmse = 2.5 # Initial default
    
    for gw in gws:
        t0 = time.time()
        
        # 1. Train set: previous seasons + current season's gameweeks strictly < gw
        # This guarantees 100% leak-free point-in-time training!
        prior_seasons_data = features_df[features_df["season"].isin(train_seasons)]
        current_season_past_data = test_season_data[test_season_data["gw"] < gw]
        
        train_data = pd.concat([prior_seasons_data, current_season_past_data], ignore_index=True)
        train_data = train_data.dropna(subset=[target_col])
        
        # 2. Test set: current gameweek
        gw_test_data = test_season_data[test_season_data["gw"] == gw].copy()
        if gw_test_data.empty:
            continue
            
        X_test_gw = gw_test_data[numeric_features + categorical_features]
        
        # Check if we need to retrain
        is_baseline = model_name in ["last_5_avg", "weighted_form", "position_avg"]
        should_retrain = (
            current_model is None or 
            is_baseline or 
            (gw - last_train_gw) >= retrain_interval
        )
        
        if should_retrain:
            X_train = train_data[numeric_features + categorical_features]
            y_train = train_data[target_col].values
            
            # Train
            if is_baseline:
                if model_name == "last_5_avg":
                    current_model = LastNGamesAverageRegressor(n=5)
                elif model_name == "weighted_form":
                    current_model = WeightedRecentFormRegressor()
                else:
                    current_model = PositionSeasonAverageRegressor()
            else:
                current_model = build_pipeline(model_name, numeric_features, categorical_features, task_type="regression")
                if custom_params and model_name == "lightgbm":
                    current_model.set_params(**{f"model__{k}": v for k, v in custom_params.items()})
                
            current_model.fit(X_train, y_train)
            last_train_gw = gw
            
            # Calculate validation RMSE on the train set (representing recent performance)
            train_preds = current_model.predict(X_train)
            validation_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            
        # 3. Predict GW N
        predictions = generate_predictions_with_uncertainty(current_model, X_test_gw, validation_rmse=validation_rmse)
        
        gw_test_data["predicted_points"] = predictions["predicted_points"].values
        gw_test_data["uncertainty_std"] = predictions["uncertainty_std"].values
        gw_test_data["prob_5_plus"] = predictions["prob_5_plus"].values
        gw_test_data["prob_10_plus"] = predictions["prob_10_plus"].values
        
        all_gw_preds.append(gw_test_data)
        
        print(f"  GW {gw:02d} processed in {time.time() - t0:.2f}s | Train size: {len(train_data):,} | Test size: {len(gw_test_data):,}")
        
    # Combine all predictions
    results_df = pd.concat(all_gw_preds, ignore_index=True)
    
    # Calculate value rankings
    results_df = rank_players_by_value(
        results_df,
        pd.DataFrame({
            "predicted_points": results_df["predicted_points"],
            "uncertainty_std": results_df["uncertainty_std"],
            "prob_5_plus": results_df["prob_5_plus"],
            "prob_10_plus": results_df["prob_10_plus"]
        })
    )
    
    # Evaluate
    metrics = calculate_evaluation_metrics(results_df, "predicted_points", target_col)
    metrics["model_name"] = model_name
    metrics["backtest_type"] = "weekly_expanding"
    
    return metrics, results_df


def save_experiment_results(
    metrics: Dict[str, Any],
    predictions: pd.DataFrame,
    model_name: str,
    test_season: str,
    experiment_name: str
):
    """
    Save the backtest results and metrics to data/results/ with a timestamp.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{test_season}_{model_name}_{experiment_name}_{timestamp}"
    
    # Save metrics
    metrics_path = RESULTS_DIR / f"{filename_base}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save predictions
    preds_path = RESULTS_DIR / f"{filename_base}_predictions.parquet"
    # Keep only relevant columns for saving space
    save_cols = [
        "season", "gw", "code", "name", "position", "current_price",
        "target_points", "predicted_points", "predicted_points_per_million",
        "uncertainty_std", "prob_5_plus", "prob_10_plus", "rank_overall", "rank_position"
    ]
    predictions[save_cols].to_parquet(preds_path, index=False)
    
    print(f"\nSaved Experiment Results:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Predictions: {preds_path}")
