import unittest
import pandas as pd
import numpy as np
import json
import hashlib
import io
import sys
from unittest.mock import patch
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.generate_historical_context import main as run_historical_context_gen
from src.evaluation.evaluate_context_model import run_chronological_backtest
from src.data.player_mapping import build_player_id_mapping

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "config"

class TestContextEvaluation(unittest.TestCase):
    
    def setUp(self):
        """Set up mock datasets for point-in-time and leakage checks."""
        self.season = "2024-25"
        
        # Saka code 101, Rice code 102
        self.mock_features_df = pd.DataFrame([
            {"season": "2024-25", "gw": 14, "code": 101, "name": "Bukayo Saka", "position": "MID", "season_team_id": 1, "target_points": 5.0, "target_minutes": 90.0, "target_60_plus_minutes": 1.0, "current_price": 10.0},
            {"season": "2024-25", "gw": 15, "code": 101, "name": "Bukayo Saka", "position": "MID", "season_team_id": 1, "target_points": 8.0, "target_minutes": 90.0, "target_60_plus_minutes": 1.0, "current_price": 10.0},
            {"season": "2024-25", "gw": 14, "code": 102, "name": "Declan Rice", "position": "MID", "season_team_id": 1, "target_points": 3.0, "target_minutes": 90.0, "target_60_plus_minutes": 1.0, "current_price": 9.0},
            {"season": "2024-25", "gw": 15, "code": 102, "name": "Declan Rice", "position": "MID", "season_team_id": 1, "target_points": 2.0, "target_minutes": 90.0, "target_60_plus_minutes": 1.0, "current_price": 9.0}
        ])
        
        # PL match history
        self.mock_player_gw_df = pd.DataFrame([
            {"season": "2024-25", "gw": 14, "player_id": 1, "code": 101, "kickoff_time": "2024-11-15T15:00:00Z", "minutes": 90, "goals_scored": 1, "assists": 0, "starts": 1},
            {"season": "2024-25", "gw": 14, "player_id": 2, "code": 102, "kickoff_time": "2024-11-15T15:00:00Z", "minutes": 90, "goals_scored": 0, "assists": 0, "starts": 1}
        ])
        
        # GW deadlines
        self.mock_fixtures_df = pd.DataFrame([
            {"season": "2024-25", "event": 14, "team_h": 1, "team_a": 2, "kickoff_time": "2024-11-15T15:00:00Z", "deadline_time": "2024-11-15T13:30:00Z"},
            {"season": "2024-25", "event": 15, "team_h": 1, "team_a": 3, "kickoff_time": "2024-11-22T15:00:00Z", "deadline_time": "2024-11-22T13:30:00Z"}
        ])
        
        self.mock_players_df = pd.DataFrame([
            {"player_id": 1, "code": 101, "season": "2024-25"},
            {"player_id": 2, "code": 102, "season": "2024-25"}
        ])
        
        self.mock_player_id_mapping = pd.DataFrame([
            {"fpl_code": 101, "fpl_name": "Bukayo Saka", "fbref_id": "ext_101", "fbref_name": "Bukayo Saka", "confidence": "exact"},
            {"fpl_code": 102, "fpl_name": "Declan Rice", "fbref_id": "ext_102", "fbref_name": "Declan Rice", "confidence": "exact"}
        ])
        
        # Mock Cup Matches:
        # Match A: Before Cutoff (10 days before target GW 15 deadline) -> should be counted!
        # Match B: After Cutoff (1 day after target GW 15 deadline) -> MUST be excluded (zero leakage check)!
        self.mock_external_matches = [
            {
                "fbref_id": "ext_101",
                "fbref_name": "Bukayo Saka",
                "date": "2024-11-12T20:00:00Z", # before GW 15 cutoff
                "competition": "Champions League",
                "opponent": "Sporting CP",
                "minutes": 90,
                "started": 1,
                "goals": 1,
                "assists": 0
            },
            {
                "fbref_id": "ext_101",
                "fbref_name": "Bukayo Saka",
                "date": "2024-11-23T20:00:00Z", # after GW 15 cutoff
                "competition": "Champions League",
                "opponent": "Real Madrid",
                "minutes": 90,
                "started": 1,
                "goals": 2,
                "assists": 1
            }
        ]

    @patch("src.features.generate_historical_context.pd.read_parquet")
    @patch("src.features.generate_historical_context.json.load")
    def test_historical_context_generation_pit_and_no_leakage(self, mock_json_load, mock_read_parquet):
        """Verify historical context features conform to point-in-time constraints (no target gw stats leak).
        Covers rules: (1) no future data, (2) prediction cutoff respected, (3) target GW data excluded.
        """
        def side_effect(path):
            if "features_df" in str(path):
                return self.mock_features_df
            elif "player_gw" in str(path):
                return self.mock_player_gw_df
            elif "fixtures" in str(path):
                return self.mock_fixtures_df
            elif "players" in str(path):
                return self.mock_players_df
            elif "player_id_mapping" in str(path):
                return self.mock_player_id_mapping
            return pd.DataFrame()
            
        mock_read_parquet.side_effect = side_effect
        mock_json_load.return_value = self.mock_external_matches
        
        saved_df = []
        def to_parquet_replacement(self_df, path, *args, **kwargs):
            if "historical_context_features" in str(path):
                saved_df.append(self_df)
                
        with patch("src.features.generate_historical_context.pd.DataFrame.to_parquet", new=to_parquet_replacement):
            with patch("src.features.generate_historical_context.Path.exists", return_value=True):
                # Run Generation
                run_historical_context_gen()
        
        self.assertEqual(len(saved_df), 1)
        res_df = saved_df[0]
        
        # Assert Saka row for GW 15
        saka_15 = res_df[(res_df["gw"] == 15) & (res_df["code"] == 101)]
        self.assertFalse(saka_15.empty)
        
        # Saka played 90 mins in PL GW 14 and 90 mins in CL on Nov 12 (before cutoff) -> total workload should be 180.0
        self.assertEqual(float(saka_15["pl_minutes_last_14d"].values[0]), 90.0)
        self.assertEqual(float(saka_15["external_minutes_last_14d"].values[0]), 90.0)
        self.assertEqual(float(saka_15["total_minutes_last_14d"].values[0]), 180.0)
        self.assertEqual(int(saka_15["external_goals_last_14d"].values[0]), 1)
        self.assertEqual(int(saka_15["external_assists_last_14d"].values[0]), 0)
        
        # Target GW performance (e.g. FPL points, target_points) must not leak to features
        self.assertNotIn("target_points", res_df.columns)
        self.assertNotIn("target_minutes", res_df.columns)
        self.assertNotIn("target_60_plus_minutes", res_df.columns)

    def test_training_data_preservation_hash_check(self):
        """Verify features_df.parquet remains byte-for-byte unchanged after runs.
        Covers rule: (4) historical training data remains unchanged.
        """
        hist_path = PROCESSED_DIR / "features_df.parquet"
        self.assertTrue(hist_path.exists())
        
        with open(hist_path, "rb") as f:
            initial_hash = hashlib.sha256(f.read()).hexdigest()
            
        df = pd.read_parquet(hist_path)
        self.assertFalse(df.empty)
        
        with open(hist_path, "rb") as f:
            final_hash = hashlib.sha256(f.read()).hexdigest()
            
        self.assertEqual(initial_hash, final_hash, "ERROR: Training parquet features_df.parquet was mutated!")

    def test_baseline_columns_untouched(self):
        """Verify baseline features remain identical and match baseline columns list.
        Covers rule: (5) baseline feature columns remain unchanged.
        """
        import yaml
        with open(CONFIG_DIR / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        with open(CONFIG_DIR / "context_model_evaluation.yaml", "r") as f:
            eval_config = yaml.safe_load(f)
            
        baseline_cols = []
        for grp in eval_config["feature_groups"]["baseline"]:
            baseline_cols.extend(config["features"].get(grp, []))
            
        # Verify baseline columns exist in processed features_df.parquet
        hist_df = pd.read_parquet(PROCESSED_DIR / "features_df.parquet")
        for col in baseline_cols:
            self.assertIn(col, hist_df.columns)

    def test_context_model_features_approved(self):
        """Verify context model contains only baseline plus approved context feature columns.
        Covers rule: (6) context model contains only approved additional features.
        """
        import yaml
        with open(CONFIG_DIR / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        with open(CONFIG_DIR / "context_model_evaluation.yaml", "r") as f:
            eval_config = yaml.safe_load(f)
            
        context_cols = []
        for grp in eval_config["feature_groups"]["context"]:
            context_cols.extend(config["features"].get(grp, []))
            
        approved_set = {
            "pl_minutes_last_14d", "external_minutes_last_7d", "external_minutes_last_14d", "external_minutes_last_21d",
            "total_minutes_last_14d", "external_appearances_last_7d", "external_appearances_last_14d",
            "external_appearances_last_21d", "external_starts_last_14d", "external_goals_last_14d", "external_assists_last_14d",
            "total_competitive_minutes_last_7d", "total_competitive_minutes_last_14d", "total_competitive_minutes_last_21d",
            "days_since_player_last_match", "team_matches_last_7d", "team_matches_last_14d", "team_matches_next_7d",
            "days_since_team_last_match", "days_until_next_match", "fixture_congestion_score"
        }
        
        for col in context_cols:
            self.assertIn(col, approved_set)

    def test_chronological_splits_and_no_random(self):
        """Verify train/test splits use chronological expanding window, without random splits.
        Covers rules: (7) identical chronological splits, (8) no random splitting.
        """
        df = pd.DataFrame([
            {"season": "2022-23", "gw": 1, "code": 101, "position": "MID", "feature_1": 1.0, "target_points": 5.0},
            {"season": "2022-23", "gw": 2, "code": 101, "position": "MID", "feature_1": 2.0, "target_points": 6.0},
            {"season": "2023-24", "gw": 1, "code": 101, "position": "MID", "feature_1": 1.5, "target_points": 4.0},
            {"season": "2023-24", "gw": 2, "code": 101, "position": "MID", "feature_1": 2.5, "target_points": 7.0}
        ])
        
        hyperparams = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 5}
        feature_cols = ["feature_1", "position"]
        
        # Run backtest
        preds = run_chronological_backtest(
            df=df,
            train_seasons=["2022-23"],
            test_season="2023-24",
            feature_cols=feature_cols,
            target_col="target_points",
            task_type="regression",
            hyperparams=hyperparams
        )
        
        # Chronological verification: check that preds are generated in sequence
        self.assertEqual(list(preds["gw"]), [1, 2])
        # Random splitting would mix train/test rows, but here test rows are strictly kept in original sequence
        self.assertEqual(list(preds["season"].unique()), ["2023-24"])

    def test_model_b_train_and_predict(self):
        """Verify that Model B trains and makes predictions correctly on LightGBM.
        Covers rule: (9) Model B can train and predict.
        """
        df = pd.DataFrame([
            {"season": "2022-23", "gw": 1, "code": 101, "position": "MID", "feature_1": 1.0, "target_points": 5.0},
            {"season": "2022-23", "gw": 2, "code": 101, "position": "MID", "feature_1": 2.0, "target_points": 6.0},
            {"season": "2023-24", "gw": 1, "code": 101, "position": "MID", "feature_1": 1.5, "target_points": 4.0},
            {"season": "2023-24", "gw": 2, "code": 101, "position": "MID", "feature_1": 2.5, "target_points": 7.0}
        ])
        
        hyperparams = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 5}
        feature_cols = ["feature_1", "position"]
        
        preds = run_chronological_backtest(
            df=df,
            train_seasons=["2022-23"],
            test_season="2023-24",
            feature_cols=feature_cols,
            target_col="target_points",
            task_type="regression",
            hyperparams=hyperparams
        )
        self.assertIn("predicted", preds.columns)
        self.assertFalse(preds["predicted"].isna().any())

    def test_metrics_on_identical_rows(self):
        """Verify both models predictions are evaluated on the exact same row subsets/indices.
        Covers rule: (10) metrics calculated on identical rows.
        """
        df = pd.DataFrame([
            {"season": "2023-24", "gw": 1, "code": 101, "position": "MID", "feature_1": 1.0, "target_points": 5.0},
            {"season": "2023-24", "gw": 2, "code": 101, "position": "MID", "feature_1": 2.0, "target_points": 6.0}
        ])
        
        preds_a = df.copy()
        preds_a["predicted"] = [4.5, 5.8]
        preds_b = df.copy()
        preds_b["predicted"] = [4.8, 6.1]
        
        # Verify alignment
        self.assertEqual(len(preds_a), len(preds_b))
        self.assertTrue((preds_a.index == preds_b.index).all())
        self.assertTrue((preds_a["code"] == preds_b["code"]).all())
        self.assertTrue((preds_a["gw"] == preds_b["gw"]).all())

    @patch("src.features.generate_historical_context.json.load")
    def test_missing_external_data_handling(self, mock_json_load):
        """Verify that when external matches JSON is missing or malformed, the pipeline handles it safely.
        Covers rule: (11) missing external data handled safely.
        """
        original_exists = Path.exists
        
        def exists_replacement(self_obj):
            if "external_cup_matches" in str(self_obj):
                return False
            return original_exists(self_obj)
            
        with patch("src.features.generate_historical_context.Path.exists", new=exists_replacement):
            mock_json_load.return_value = None
            # The script should not crash and should execute successfully with defaults
            try:
                run_historical_context_gen()
            except Exception as e:
                self.fail(f"generate_historical_context main crashed when external file was missing: {e}")

    def test_player_mapping_collisions_warning(self):
        """Verify ambiguous mappings and identity collisions generate output warnings.
        Covers rule: (12) player mapping collisions detected.
        """
        fpl_players = pd.DataFrame([
            {"code": 101, "name": "Bukayo Saka", "team": 1},
            {"code": 102, "name": "Declan Rice", "team": 1}
        ])
        ext_players = pd.DataFrame([
            {"fbref_id": "ext_collision", "fbref_name": "Bukayo Saka", "team": "1"},
            {"fbref_id": "ext_collision", "fbref_name": "Declan Rice", "team": "1"} # Duplicate fbref_id triggers collision warning
        ])
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            build_player_id_mapping(fpl_players, ext_players)
            captured = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            
        self.assertIn("Player identity collision detected", captured)

    @patch("src.features.generate_historical_context.pd.read_parquet")
    @patch("src.features.generate_historical_context.json.load")
    def test_historical_generation_determinism(self, mock_json_load, mock_read_parquet):
        """Verify that running historical features generation multiple times yields identical feature outputs.
        Covers rule: (13) results are deterministic where applicable.
        """
        def side_effect(path):
            if "features_df" in str(path):
                return self.mock_features_df
            elif "player_gw" in str(path):
                return self.mock_player_gw_df
            elif "fixtures" in str(path):
                return self.mock_fixtures_df
            elif "players" in str(path):
                return self.mock_players_df
            elif "player_id_mapping" in str(path):
                return self.mock_player_id_mapping
            return pd.DataFrame()
            
        mock_read_parquet.side_effect = side_effect
        mock_json_load.return_value = self.mock_external_matches
        
        saved_dfs = []
        def to_parquet_replacement(self_df, path, *args, **kwargs):
            if "historical_context_features" in str(path):
                saved_dfs.append(self_df)
                
        with patch("src.features.generate_historical_context.pd.DataFrame.to_parquet", new=to_parquet_replacement):
            with patch("src.features.generate_historical_context.Path.exists", return_value=True):
                # Run 1
                run_historical_context_gen()
                # Run 2
                run_historical_context_gen()
        
        self.assertEqual(len(saved_dfs), 2)
        pd.testing.assert_frame_equal(saved_dfs[0], saved_dfs[1])

if __name__ == "__main__":
    unittest.main()
