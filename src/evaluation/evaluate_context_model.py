import pandas as pd
import numpy as np
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import spearmanr, wilcoxon

import sys
import gc
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train import build_pipeline
from src.models.predict import load_tuned_hyperparameters, generate_predictions_with_uncertainty

# Constants
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
EVAL_CONFIG_PATH = PROJECT_ROOT / "config" / "context_model_evaluation.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def calculate_spearman_and_overlaps(df: pd.DataFrame, pred_col: str, target_col: str) -> Tuple[float, float, float]:
    """Calculate average weekly Spearman correlation, Top-10, and Top-20 overlaps."""
    weekly_corrs = []
    top_10_overlaps = []
    top_20_overlaps = []
    
    for (season, gw), group in df.groupby(["season", "gw"]):
        if len(group) < 10:
            continue
        t_true = group[target_col].values
        t_pred = group[pred_col].values
        
        # Spearman Rank Correlation
        if np.all(t_pred == t_pred[0]) or np.all(t_true == t_true[0]):
            weekly_corrs.append(0.0)
        else:
            corr, _ = spearmanr(t_pred, t_true)
            if not np.isnan(corr):
                weekly_corrs.append(corr)
                
        # Overlaps
        group_sorted_pred = group.sort_values(pred_col, ascending=False)
        group_sorted_true = group.sort_values(target_col, ascending=False)
        
        pred_top_10 = set(group_sorted_pred.head(10)["code"].values)
        true_top_10 = set(group_sorted_true.head(10)["code"].values)
        top_10_overlaps.append(len(pred_top_10.intersection(true_top_10)) / 10.0)
        
        pred_top_20 = set(group_sorted_pred.head(20)["code"].values)
        true_top_20 = set(group_sorted_true.head(20)["code"].values)
        top_20_overlaps.append(len(pred_top_20.intersection(true_top_20)) / 20.0)
        
    return (
        float(np.mean(weekly_corrs)) if weekly_corrs else 0.0,
        float(np.mean(top_10_overlaps)) if top_10_overlaps else 0.0,
        float(np.mean(top_20_overlaps)) if top_20_overlaps else 0.0
    )

def run_chronological_backtest(
    df: pd.DataFrame,
    train_seasons: List[str],
    test_season: str,
    feature_cols: List[str],
    target_col: str,
    task_type: str,
    hyperparams: Dict[str, Any],
    retrain_interval: int = 4
) -> pd.DataFrame:
    """Run chronological expanding backtest for a target season with configurable retraining interval to save memory."""
    # Memory optimization: Keep only columns needed for this specific training and predict task
    required_cols = list(set(["season", "gw", "code", target_col] + feature_cols))
    df = df[[c for c in required_cols if c in df.columns]].copy()
    
    test_data = df[df["season"] == test_season]
    gws = sorted(test_data["gw"].unique())
    
    categorical_features = ["position"]
    numeric_features = [c for c in feature_cols if c not in categorical_features]
    
    all_gw_preds = []
    current_model = None
    last_train_gw = -1
    
    # Pre-split training data from prior seasons
    prior_data = df[df["season"].isin(train_seasons)]
    
    for gw in gws:
        gw_test = test_data[test_data["gw"] == gw].copy()
        if gw_test.empty:
            continue
            
        X_test = gw_test[feature_cols]
        
        # Check if we should retrain
        should_retrain = (
            current_model is None or 
            (gw - last_train_gw) >= retrain_interval
        )
        
        if should_retrain:
            # Free old model memory explicitly before constructing training datasets
            if current_model is not None:
                del current_model
                current_model = None
                gc.collect()
                
            current_past = test_data[test_data["gw"] < gw]
            train_sub = pd.concat([prior_data, current_past], ignore_index=True)
            train_sub = train_sub.dropna(subset=[target_col])
            
            if train_sub.empty:
                continue
                
            X_train = train_sub[feature_cols]
            y_train = train_sub[target_col].values
            del train_sub
            gc.collect()
            
            # Build and fit pipeline
            current_model = build_pipeline("lightgbm", numeric_features, categorical_features, task_type=task_type)
            if hyperparams:
                applied_params = {f"model__{k}": v for k, v in hyperparams.items()}
                current_model.set_params(**applied_params)
                
            current_model.fit(X_train, y_train)
            last_train_gw = gw
            
            # Free up memory explicitly
            gc.collect()
            
        # Predict
        if task_type == "regression":
            preds = current_model.predict(X_test)
            gw_test["predicted"] = np.clip(preds, 0.0, None)
        else:
            # Classification
            if hasattr(current_model, "predict_proba"):
                probs = current_model.predict_proba(X_test)[:, 1]
            else:
                probs = np.clip(current_model.predict(X_test), 0.0, 1.0)
            gw_test["predicted"] = probs
            
        all_gw_preds.append(gw_test)
        
    return pd.concat(all_gw_preds, ignore_index=True) if all_gw_preds else pd.DataFrame()

def main():
    print("=" * 60)
    print("PHASE 8: HISTORICAL EVALUATION & MODEL COMPARISON")
    print("=" * 60)
    
    # 1. Load Configurations
    config = load_yaml(CONFIG_PATH)
    eval_config = load_yaml(EVAL_CONFIG_PATH)
    
    # Load hyperparams
    tuned_params = load_tuned_hyperparameters()
    print(f"Loaded Tuned Hyperparameters: {tuned_params}")
    
    # Identify required features first
    context_cols = []
    for grp in eval_config["feature_groups"]["context"]:
        context_cols.extend(config["features"].get(grp, []))
        
    baseline_cols = []
    for grp in eval_config["feature_groups"]["baseline"]:
        baseline_cols.extend(config["features"].get(grp, []))
    baseline_cols.append("position") # categorical
    
    # Load Datasets
    print("\nLoading datasets...")
    baseline_cols = list(set(baseline_cols))
    
    # 1. Determine columns to load first using pyarrow schema reader (no row allocations)
    import pyarrow.parquet as pq
    features_cols_to_keep = ["season", "gw", "code", "name", "target_points", "target_minutes", "target_60_plus_minutes"] + baseline_cols
    schema_cols = pq.read_schema(PROCESSED_DIR / "features_df.parquet").names
    features_cols_to_keep = [col for col in features_cols_to_keep if col in schema_cols]
    
    context_cols_to_keep = ["season", "gw", "code", "name"] + context_cols
    schema_context_cols = pq.read_schema(PROCESSED_DIR / "historical_context_features.parquet").names
    context_cols_to_keep = [col for col in context_cols_to_keep if col in schema_context_cols]
    
    # Load optimized columns
    features_df = pd.read_parquet(PROCESSED_DIR / "features_df.parquet", columns=features_cols_to_keep)
    context_features_df = pd.read_parquet(PROCESSED_DIR / "historical_context_features.parquet", columns=context_cols_to_keep)
    
    # Filter to active seasons to save rows
    active_seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    features_df = features_df[features_df["season"].isin(active_seasons)].copy()
    context_features_df = context_features_df[context_features_df["season"].isin(active_seasons)].copy()
    
    # Downcast dtypes to float32/int32 to halve memory footprint
    for col in features_df.columns:
        if features_df[col].dtype == np.float64:
            features_df[col] = features_df[col].astype(np.float32)
        elif features_df[col].dtype == np.int64:
            features_df[col] = features_df[col].astype(np.int32)
            
    for col in context_features_df.columns:
        if context_features_df[col].dtype == np.float64:
            context_features_df[col] = context_features_df[col].astype(np.float32)
        elif context_features_df[col].dtype == np.int64:
            context_features_df[col] = context_features_df[col].astype(np.int32)
            
    # Merge context features into baseline features (highly optimized memory footprint)
    df = pd.merge(features_df, context_features_df, on=["season", "gw", "code", "name"], how="left")
    
    # Clean up intermediate dataframes from memory immediately
    del features_df
    del context_features_df
    import gc
    gc.collect()
    
    # Fill any NaNs in workload features with defaults
    for col in context_cols:
        if "days" in col:
            df[col] = df[col].fillna(np.float32(99.0))
        else:
            df[col] = df[col].fillna(np.float32(0.0))
            
    print(f"Merged Dataset Shape: {df.shape}")
    
    # 2. Prevent Dataset Mismatch Verification
    total_rows = len(df)
    seasons = sorted(df["season"].unique())
    gws = sorted(df["gw"].unique())
    players = df["code"].nunique()
    
    print("\n" + "-"*40)
    print("DATASET COVERAGE VERIFICATION")
    print("-"*40)
    print(f"Total player-GW rows: {total_rows:,}")
    print(f"Number of seasons: {len(seasons)} ({seasons})")
    print(f"Number of gameweeks: {len(gws)}")
    print(f"Number of unique players: {players:,}")
    
    print("\nFeature Missingness (relative to non-empty seasons):")
    non_empty_seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    active_mask = df["season"].isin(non_empty_seasons)
    for col in context_cols:
        pct_missing = (df[active_mask][col] == (99.0 if "days" in col else 0.0)).mean() * 100
        print(f"  {col}: {pct_missing:.2f}% default/missing value")
        
    # Enriched Feature Columns list
    enriched_cols = baseline_cols + context_cols
    
    # 3. Model Comparison Chronological Splits
    splits = eval_config["splits"]
    train_seasons = splits["train_seasons"]
    
    # We evaluate across Validation, Test, and Out-of-sample seasons
    target_seasons = [
        ("Validation", splits["validation_season"], train_seasons),
        ("Test", splits["test_season"], train_seasons + [splits["validation_season"]]),
        ("Out-of-sample", splits["oos_season"], train_seasons + [splits["validation_season"], splits["test_season"]])
    ]
    
    results = []
    weekly_spearman_details = []
    val_pts_metrics_a = (0.0, 0.0, 0.0)
    val_pts_metrics_b = (0.0, 0.0, 0.0)
    
    for split_name, season, train_list in target_seasons:
        print(f"\nEvaluating target season: {season} ({split_name})")
        print(f"  Train seasons: {train_list}")
        
        # A. Expected Points Model (Regression)
        print(f"  Running Backtests for points...")
        preds_a_pts = run_chronological_backtest(df, train_list, season, baseline_cols, "target_points", "regression", tuned_params)
        preds_b_pts = run_chronological_backtest(df, train_list, season, enriched_cols, "target_points", "regression", tuned_params)
        
        # Compute points metrics
        pts_metrics_a = calculate_spearman_and_overlaps(preds_a_pts, "predicted", "target_points")
        pts_metrics_b = calculate_spearman_and_overlaps(preds_b_pts, "predicted", "target_points")
        
        if season == "2023-24":
            val_pts_metrics_a = pts_metrics_a
            val_pts_metrics_b = pts_metrics_b
            
        mae_a = mean_absolute_error(preds_a_pts["target_points"], preds_a_pts["predicted"])
        rmse_a = np.sqrt(mean_squared_error(preds_a_pts["target_points"], preds_a_pts["predicted"]))
        r2_a = r2_score(preds_a_pts["target_points"], preds_a_pts["predicted"])
        
        mae_b = mean_absolute_error(preds_b_pts["target_points"], preds_b_pts["predicted"])
        rmse_b = np.sqrt(mean_squared_error(preds_b_pts["target_points"], preds_b_pts["predicted"]))
        r2_b = r2_score(preds_b_pts["target_points"], preds_b_pts["predicted"])
        
        # Save points metrics
        results.append({
            "season": season, "split": split_name, "model": "Model A (Baseline)", "target": "points",
            "mae": mae_a, "rmse": rmse_a, "r2": r2_a, "spearman": pts_metrics_a[0], "top_10": pts_metrics_a[1], "top_20": pts_metrics_a[2]
        })
        results.append({
            "season": season, "split": split_name, "model": "Model B (Context)", "target": "points",
            "mae": mae_b, "rmse": rmse_b, "r2": r2_b, "spearman": pts_metrics_b[0], "top_10": pts_metrics_b[1], "top_20": pts_metrics_b[2]
        })
        
        # Compile weekly Spearman correlation detail for Wilcoxon significance test
        # We group by GW to compare weekly wins
        gw_a_corrs = {}
        for (s, gw), grp in preds_a_pts.groupby(["season", "gw"]):
            if len(grp) >= 10:
                corr, _ = spearmanr(grp["predicted"], grp["target_points"])
                gw_a_corrs[gw] = corr
                
        for (s, gw), grp in preds_b_pts.groupby(["season", "gw"]):
            if len(grp) >= 10:
                corr, _ = spearmanr(grp["predicted"], grp["target_points"])
                weekly_spearman_details.append({
                    "season": season,
                    "gw": gw,
                    "model_a_spearman": gw_a_corrs.get(gw, 0.0),
                    "model_b_spearman": corr,
                    "spearman_diff": corr - gw_a_corrs.get(gw, 0.0)
                })
        
        # B. Expected Minutes Model (Regression)
        print(f"  Running Backtests for minutes...")
        preds_a_min = run_chronological_backtest(df, train_list, season, baseline_cols, "target_minutes", "regression", tuned_params)
        preds_b_min = run_chronological_backtest(df, train_list, season, enriched_cols, "target_minutes", "regression", tuned_params)
        
        mae_min_a = mean_absolute_error(preds_a_min["target_minutes"], preds_a_min["predicted"])
        rmse_min_a = np.sqrt(mean_squared_error(preds_a_min["target_minutes"], preds_a_min["predicted"]))
        r2_min_a = r2_score(preds_a_min["target_minutes"], preds_a_min["predicted"])
        
        mae_min_b = mean_absolute_error(preds_b_min["target_minutes"], preds_b_min["predicted"])
        rmse_min_b = np.sqrt(mean_squared_error(preds_b_min["target_minutes"], preds_b_min["predicted"]))
        r2_min_b = r2_score(preds_b_min["target_minutes"], preds_b_min["predicted"])
        
        results.append({
            "season": season, "split": split_name, "model": "Model A (Baseline)", "target": "minutes",
            "mae": mae_min_a, "rmse": rmse_min_a, "r2": r2_min_a, "spearman": 0.0, "top_10": 0.0, "top_20": 0.0
        })
        results.append({
            "season": season, "split": split_name, "model": "Model B (Context)", "target": "minutes",
            "mae": mae_min_b, "rmse": rmse_min_b, "r2": r2_min_b, "spearman": 0.0, "top_10": 0.0, "top_20": 0.0
        })
        
        # C. Playing Probability Model (Classification)
        print(f"  Running Backtests for playing 60+ classification...")
        preds_a_clf = run_chronological_backtest(df, train_list, season, baseline_cols, "target_60_plus_minutes", "classification", tuned_params)
        preds_b_clf = run_chronological_backtest(df, train_list, season, enriched_cols, "target_60_plus_minutes", "classification", tuned_params)
        
        # Calculate Classification Metrics
        y_true = preds_a_clf["target_60_plus_minutes"].values
        y_prob_a = preds_a_clf["predicted"].values
        y_pred_a = (y_prob_a >= 0.5).astype(int)
        
        y_prob_b = preds_b_clf["predicted"].values
        y_pred_b = (y_prob_b >= 0.5).astype(int)
        
        results.append({
            "season": season, "split": split_name, "model": "Model A (Baseline)", "target": "playing_60",
            "roc_auc": roc_auc_score(y_true, y_prob_a),
            "accuracy": accuracy_score(y_true, y_pred_a),
            "precision": precision_score(y_true, y_pred_a),
            "recall": recall_score(y_true, y_pred_a),
            "f1": f1_score(y_true, y_pred_a)
        })
        results.append({
            "season": season, "split": split_name, "model": "Model B (Context)", "target": "playing_60",
            "roc_auc": roc_auc_score(y_true, y_prob_b),
            "accuracy": accuracy_score(y_true, y_pred_b),
            "precision": precision_score(y_true, y_pred_b),
            "recall": recall_score(y_true, y_pred_b),
            "f1": f1_score(y_true, y_pred_b)
        })

    # Save Comparison DataFrames
    comparison_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULTS_DIR / "context_model_comparison.csv", index=False)
    
    weekly_df = pd.DataFrame(weekly_spearman_details)
    weekly_df.to_csv(RESULTS_DIR / "context_weekly_comparison.csv", index=False)
    
    print("\n" + "-"*40)
    print("BACKTEST COMPARISON SUMMARY (Points)")
    print("-"*40)
    print(comparison_df[comparison_df["target"] == "points"][["season", "model", "mae", "rmse", "r2", "spearman", "top_10"]].to_string(index=False))
    
    print("\n" + "-"*40)
    print("BACKTEST COMPARISON SUMMARY (Playing 60+)")
    print("-"*40)
    print(comparison_df[comparison_df["target"] == "playing_60"][["season", "model", "roc_auc", "accuracy", "f1"]].to_string(index=False))
    
    # 4. Statistical Significance Wilcoxon signed-rank test
    print("\n" + "-"*40)
    print("STATISTICAL SIGNIFICANCE (Points weekly Spearman)")
    print("-"*40)
    
    diffs = weekly_df["spearman_diff"].values
    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)
    win_rate = (diffs > 0).mean() * 100
    std_diff = np.std(diffs)
    
    print(f"Mean Spearman Difference:  {mean_diff:+.6f}")
    print(f"Median Spearman Difference: {median_diff:+.6f}")
    print(f"Weekly Win Rate (Model B): {win_rate:.2f}%")
    print(f"Std Dev of Differences:     {std_diff:.6f}")
    
    # Wilcoxon signed-rank test (exclude zero differences)
    non_zero_diffs = diffs[diffs != 0]
    if len(non_zero_diffs) >= 5:
        stat, p_value = wilcoxon(non_zero_diffs)
        print(f"Wilcoxon signed-rank test p-value: {p_value:.6f}")
        if p_value < 0.05:
            print("  -> Statistically Significant improvement (p < 0.05)!")
        else:
            print("  -> Difference is NOT statistically significant (p >= 0.05).")
    else:
        print("  -> Too few non-zero difference weeks for Wilcoxon test.")
        
    # 5. Ablation Analysis on Validation Season 2023-24
    print("\n" + "-"*40)
    print("ABLATION ANALYSIS (Validation 2023-24)")
    print("-"*40)
    
    # Feature group variables
    workload_only_cols = baseline_cols + config["features"]["player_workload"]
    congestion_only_cols = baseline_cols + config["features"]["fixture_congestion"]
    
    print("Running workload-only backtest...")
    preds_workload = run_chronological_backtest(df, train_seasons, "2023-24", workload_only_cols, "target_points", "regression", tuned_params)
    print("Running congestion-only backtest...")
    preds_congestion = run_chronological_backtest(df, train_seasons, "2023-24", congestion_only_cols, "target_points", "regression", tuned_params)
    
    w_metrics = calculate_spearman_and_overlaps(preds_workload, "predicted", "target_points")
    c_metrics = calculate_spearman_and_overlaps(preds_congestion, "predicted", "target_points")
    
    # Display comparison
    ablation_results = [
        {"Model Group": "Baseline (Model A)", "Spearman": val_pts_metrics_a[0], "Top-10": val_pts_metrics_a[1]},
        {"Model Group": "Baseline + Workload", "Spearman": w_metrics[0], "Top-10": w_metrics[1]},
        {"Model Group": "Baseline + Congestion", "Spearman": c_metrics[0], "Top-10": c_metrics[1]},
        {"Model Group": "Baseline + Both (Model B)", "Spearman": val_pts_metrics_b[0], "Top-10": val_pts_metrics_b[1]}
    ]
    print(pd.DataFrame(ablation_results).to_string(index=False))
    
    # 6. Feature Importance Calculation
    print("\n" + "-"*40)
    print("FEATURE IMPORTANCE (Model B)")
    print("-"*40)
    
    # Train final Model B on all available data up to target GW 15 2024-25 to compute feature importance
    final_train_data = df[df["season"] < "2024-25"].dropna(subset=["target_points"]).copy()
    X_train_final = final_train_data[enriched_cols]
    y_train_final = final_train_data["target_points"].values
    
    categorical_features = ["position"]
    numeric_features = [c for c in enriched_cols if c not in categorical_features]
    
    final_pipeline = build_pipeline("lightgbm", numeric_features, categorical_features, task_type="regression")
    if tuned_params:
        applied_params = {f"model__{k}": v for k, v in tuned_params.items()}
        final_pipeline.set_params(**applied_params)
        
    final_pipeline.fit(X_train_final, y_train_final)
    
    # Get feature names after preprocessing
    preprocessor = final_pipeline.named_steps["preprocessor"]
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in preprocessor.get_feature_names_out()]
    
    # Get LightGBM model
    lgb_model = final_pipeline.named_steps["model"]
    importances_gain = lgb_model.booster_.feature_importance(importance_type="gain")
    importances_split = lgb_model.booster_.feature_importance(importance_type="split")
    
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": importances_gain,
        "importance_split": importances_split
    }).sort_values("importance_gain", ascending=False)
    
    # Save importance
    fi_df.to_csv(RESULTS_DIR / "context_feature_importance.csv", index=False)
    
    print("\nTop 15 features by Gain importance:")
    print(fi_df.head(15).to_string(index=False))
    
    # Top context features
    print("\nWorkload/Congestion features ranking:")
    context_fi = fi_df[fi_df["feature"].isin(context_cols)]
    print(context_fi.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("CONTEXT MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
