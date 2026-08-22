import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.tune import run_static_validation
from src.models.train import build_pipeline
from src.evaluation.backtest import load_config
from src.models.predict import enforce_leakage_guard

class TestModelTuning(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        features_path = Path(cls.config["paths"]["processed_dir"]) / "features_df.parquet"
        cls.features_df = pd.read_parquet(features_path)
        
        # Candidate hyperparameter sample
        cls.sample_params = {
            "learning_rate": 0.05,
            "n_estimators": 50,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        }
        
    def test_static_validation_success_and_metrics(self):
        """Test 1: Model trains successfully and returns valid validation metrics."""
        train_seasons = self.config["splits"]["dev_seasons"]
        val_season = self.config["splits"]["validation_season"]
        
        metrics = run_static_validation(
            self.features_df, 
            train_seasons, 
            val_season, 
            self.sample_params, 
            self.config
        )
        
        # Verify metric keys are present
        self.assertIn("spearman_rank_corr", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        
        # Verify no NaN or Inf in metric values
        for val in metrics.values():
            self.assertTrue(np.isfinite(val))
            
        print(f"    Static validation score check: Spearman = {metrics['spearman_rank_corr']:.4f}")

    def test_chronological_split_boundaries(self):
        """Test 2: Ensure training features chronologically precede validation features."""
        train_seasons = self.config["splits"]["dev_seasons"]
        val_season = self.config["splits"]["validation_season"]
        test_season = self.config["splits"]["test_season"]
        oos_season = self.config["splits"]["out_of_sample_season"]
        
        # Train data
        train_data = self.features_df[self.features_df["season"].isin(train_seasons)]
        # Val data
        val_data = self.features_df[self.features_df["season"] == val_season]
        
        # Verify all train seasons are strictly less than validation season
        for s in train_data["season"].unique():
            self.assertTrue(s < val_season, f"Train season {s} is not before validation season {val_season}")
            
        # Verify test/oos seasons are strictly greater than validation season
        self.assertTrue(val_season < test_season)
        self.assertTrue(test_season < oos_season)
        
    def test_test_data_isolation(self):
        """Test 3: Verify test and OOS seasons are completely excluded from tuning dataset."""
        train_seasons = self.config["splits"]["dev_seasons"]
        val_season = self.config["splits"]["validation_season"]
        test_season = self.config["splits"]["test_season"]
        oos_season = self.config["splits"]["out_of_sample_season"]
        
        # Check active tuning dataset features
        train_val_data = self.features_df[self.features_df["season"].isin(train_seasons + [val_season])]
        
        # Ensure absolutely no test or OOS season exists in the tuning pool
        self.assertEqual(train_val_data[train_val_data["season"] == test_season].shape[0], 0)
        self.assertEqual(train_val_data[train_val_data["season"] == oos_season].shape[0], 0)

    def test_leakage_guard_check(self):
        """Test 4: Verify leakage checks trigger on target or raw points columns."""
        # Create a mock df containing raw total_points
        mock_df = pd.DataFrame({
            "total_points_last_5": [4.0],
            "total_points": [5.0]  # Leak column!
        })
        
        with self.assertRaises(ValueError):
            enforce_leakage_guard(mock_df, self.config)
            
    def test_tuned_model_serialization(self):
        """Test 5: Tuned pipeline can fit, serialize to temp file, and deserialize successfully."""
        train_seasons = self.config["splits"]["dev_seasons"]
        target_col = self.config["targets"]["points"]
        
        numeric_features = []
        for cat in ["player_base_rolling", "player_expected", "player_derived", "player_value_ownership", "player_experience", "fixture_aggregates", "team_rolling", "team_static"]:
            numeric_features.extend(self.config["features"][cat])
        categorical_features = ["position"]
        feature_cols = numeric_features + categorical_features
        
        train_data = self.features_df[self.features_df["season"].isin(train_seasons)].dropna(subset=[target_col]).head(1000)
        
        pipeline = build_pipeline("lightgbm", numeric_features, categorical_features, task_type="regression")
        pipeline.set_params(**{f"model__{k}": v for k, v in self.sample_params.items()})
        
        pipeline.fit(train_data[feature_cols], train_data[target_col].values)
        
        # Save to a temp joblib file
        temp_path = Path("tmp_tuned_model.joblib")
        try:
            joblib.dump(pipeline, temp_path)
            
            # Load and verify
            loaded = joblib.load(temp_path)
            self.assertIsNotNone(loaded)
            
            preds = loaded.predict(train_data[feature_cols])
            self.assertEqual(len(preds), len(train_data))
        finally:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
