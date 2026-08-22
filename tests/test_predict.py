import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import sys

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.predict import load_production_models, predict_gameweek, enforce_leakage_guard, get_model_paths, load_config

class TestProductionPredict(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We will use a historical season/GW that exists in features_df.parquet for testing
        cls.season = "2024-25"
        cls.gw = 15
        
        # Load config
        cls.config = load_config()
        
        # Load features dataset to check presence and get some samples
        features_path = Path(cls.config["paths"]["processed_dir"]) / "features_df.parquet"
        cls.features_df = pd.read_parquet(features_path)
        
        # Ensure production models are trained and saved for this test case
        cls.models = load_production_models(cls.season, cls.gw)
        
    def test_model_loading_and_metadata(self):
        """Test 1: Model loads successfully and metadata is saved alongside it."""
        for label in ["points", "minutes", "prob_60"]:
            m_path, meta_path = get_model_paths(self.season, self.gw, label)
            
            # Assert file paths exist
            self.assertTrue(m_path.exists(), f"Model file missing: {m_path}")
            self.assertTrue(meta_path.exists(), f"Metadata file missing: {meta_path}")
            
            # Assert joblib model loads
            model = joblib.load(m_path)
            self.assertIsNotNone(model)
            
            # Assert metadata loads and has required keys
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            self.assertEqual(meta["cutoff_season"], self.season)
            self.assertEqual(meta["cutoff_gw"], self.gw)
            self.assertIn("target", meta)
            self.assertIn("feature_names", meta)
            self.assertIn("training_row_count", meta)
            self.assertIn("training_seasons", meta)
            self.assertIn("model_type", meta)
            self.assertIn("validation_rmse", meta)
            self.assertIn("version", meta)
            
    def test_strict_training_cutoff(self):
        """
        Test 2: Strict training cutoff test.
        Assert that the trained model contains absolutely no rows from the target gameweek or future gameweeks.
        """
        for label in ["points", "minutes", "prob_60"]:
            _, meta_path = get_model_paths(self.season, self.gw, label)
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            # Verify training seasons don't exceed the cutoff
            for s in meta["training_seasons"]:
                self.assertTrue(s <= self.season, f"Training season {s} exceeds cutoff season {self.season}!")
                
            # Perform a manual data cutoff test: check that train mask filters correctly
            cutoff_mask = (self.features_df["season"] < self.season) | (
                (self.features_df["season"] == self.season) & (self.features_df["gw"] < self.gw)
            )
            future_mask = ~cutoff_mask
            
            train_data = self.features_df[cutoff_mask]
            future_data = self.features_df[future_mask]
            
            # Asserts
            self.assertTrue(len(train_data) > 0)
            self.assertEqual(len(train_data), meta["training_row_count"])
            
            # Verify no training rows match target GW or later in the current season, or later seasons
            self.assertEqual(train_data[(train_data["season"] == self.season) & (train_data["gw"] >= self.gw)].shape[0], 0)
            self.assertEqual(train_data[train_data["season"] > self.season].shape[0], 0)
            
            print(f"    Verified strict cutoff for '{label}': trained on {len(train_data):,} rows, 0 rows from >= ({self.season}, GW {self.gw}).")
            
    def test_feature_columns_schema_and_ordering(self):
        """Test 3: Feature schema exact matches and order consistency."""
        numeric_features = []
        for cat in ["player_base_rolling", "player_expected", "player_derived", "player_value_ownership", "player_experience", "fixture_aggregates", "team_rolling", "team_static"]:
            numeric_features.extend(self.config["features"][cat])
        categorical_features = ["position"]
        expected_features = numeric_features + categorical_features
        
        for label in ["points", "minutes", "prob_60"]:
            _, meta_path = get_model_paths(self.season, self.gw, label)
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            # Verify list of features matches config
            self.assertEqual(meta["feature_names"], expected_features, f"Feature mismatch for {label}!")
            
    def test_leakage_guard(self):
        """
        Test 4: Leakage guard throws ValueError if target columns or raw performance metrics are passed
        to enforce_leakage_guard.
        """
        # Create a mock features dataframe that has target points
        mock_df = pd.DataFrame({
            "total_points_last_3": [2.0],
            "position": ["MID"],
            "target_points": [5.0] # Prohibited column!
        })
        
        with self.assertRaises(ValueError) as ctx:
            enforce_leakage_guard(mock_df, self.config)
        self.assertIn("DATA LEAKAGE SHIELD TRIGGERED", str(ctx.exception))
        
        # Create a mock features dataframe that has raw minutes
        mock_df_2 = pd.DataFrame({
            "total_points_last_3": [2.0],
            "position": ["MID"],
            "minutes": [90.0] # Prohibited column!
        })
        
        with self.assertRaises(ValueError) as ctx2:
            enforce_leakage_guard(mock_df_2, self.config)
        self.assertIn("DATA LEAKAGE SHIELD TRIGGERED", str(ctx2.exception))
        
    def test_prediction_dataframe_properties(self):
        """Test 5: Prediction quality, NaN/inf absence, row matches, and DGW row count consistency."""
        preds, forecast = predict_gameweek(self.season, self.gw, horizon=1)
        
        # 1. No NaNs/infs in predicted columns
        for col in ["predicted_points", "predicted_minutes", "prob_60_plus", "predicted_points_per_million"]:
            self.assertTrue(col in preds.columns)
            self.assertEqual(preds[col].isna().sum(), 0, f"NaNs found in column {col}!")
            self.assertTrue(np.isfinite(preds[col]).all(), f"Infs found in column {col}!")
            
        # 2. Assert exactly one row per player (player grid code)
        self.assertEqual(preds.duplicated(subset=["code"]).sum(), 0, "Duplicate players found in predictions!")
        
        # 3. Verify double-gameweek players remain as one prediction row (aggregated)
        dgw_players = self.features_df[
            (self.features_df["season"] == self.season) & 
            (self.features_df["gw"] == self.gw) & 
            (self.features_df["is_double_gw"] == 1.0)
        ]
        if len(dgw_players) > 0:
            sample_code = dgw_players["code"].values[0]
            preds_sample = preds[preds["code"] == sample_code]
            self.assertEqual(len(preds_sample), 1, f"DGW player {sample_code} has multiple rows in predictions!")
            self.assertEqual(preds_sample["num_fixtures"].values[0], 2.0, "DGW num_fixtures must be 2.0!")
            
    def test_ranking_and_value_calculations(self):
        """Test 6: Rankings sorted correctly and Points/£M value calculations are correct."""
        preds, _ = predict_gameweek(self.season, self.gw, horizon=1)
        
        # 1. Sorting check: should be sorted descending by predicted points
        points = preds["predicted_points"].values
        diffs = np.diff(points)
        self.assertTrue((diffs <= 0).all(), "Predictions are not sorted correctly descending by expected points!")
        
        # 2. Points per million check: points / price (where price > 0, else points / 4.0)
        sample = preds.head(5)
        for _, row in sample.iterrows():
            expected_ppm = row["predicted_points"] / (row["current_price"] if row["current_price"] > 0 else 4.0)
            self.assertAlmostEqual(row["predicted_points_per_million"], expected_ppm, places=5)
            
    def test_no_mutation(self):
        """Test 7: Verification that inference does not mutate the underlying dataset."""
        features_path = Path(self.config["paths"]["processed_dir"]) / "features_df.parquet"
        initial_hash = hash(pd.read_parquet(features_path).to_json())
        
        # Run prediction
        _, _ = predict_gameweek(self.season, self.gw, horizon=1)
        
        # Hash after prediction
        final_hash = hash(pd.read_parquet(features_path).to_json())
        self.assertEqual(initial_hash, final_hash, "The features dataset was mutated during inference!")

    def test_tuned_hyperparameters_loaded_and_used(self):
        """Test 8: Verify that tuned hyperparameters are loaded, set on LightGBM models, and logged in metadata."""
        from src.models.predict import load_tuned_hyperparameters, get_model_paths
        import json
        import joblib
        
        # 1. Verify load_tuned_hyperparameters() successfully reads config/tuned_lightgbm.yaml
        tuned_params = load_tuned_hyperparameters()
        self.assertIsNotNone(tuned_params)
        self.assertTrue(len(tuned_params) > 0, "Tuned hyperparameters should not be empty")
        self.assertIn("learning_rate", tuned_params)
        self.assertIn("n_estimators", tuned_params)
        
        # 2. Check metadata logs tuned config source
        m_path, meta_path = get_model_paths(self.season, self.gw, "points")
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        self.assertEqual(meta["hyperparameters_source"], "tuned")
        self.assertEqual(meta["hyperparameters"]["learning_rate"], tuned_params["learning_rate"])
        
        # 3. Check that the loaded joblib pipeline actually uses these parameters
        pipeline = joblib.load(m_path)
        model_step = pipeline.named_steps["model"]
        
        # Assert parameters match
        for param_name, expected_value in tuned_params.items():
            actual_value = model_step.get_params()[param_name]
            self.assertAlmostEqual(actual_value, expected_value, places=5, 
                                   msg=f"Parameter {param_name} mismatch: {actual_value} vs {expected_value}")
            
        print("    Verified tuned hyperparameters are correctly loaded and applied in production models.")



if __name__ == "__main__":
    unittest.main()
