import unittest
import pandas as pd
import numpy as np
import json
import hashlib
from unittest.mock import patch
from pathlib import Path
from src.data.update_current import run_update_current_pipeline, get_last_completed_gw
from src.data.player_mapping import build_player_id_mapping

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

class TestUpdateCurrent(unittest.TestCase):
    
    def setUp(self):
        """Set up mock FPL API data."""
        self.season = "2024-25"
        self.gw = 15
        
        # 1. Mock Bootstrap Static Data
        self.mock_bootstrap = {
            "events": [
                {"id": g, "finished": True, "is_current": False, "is_next": False}
                for g in range(1, 15)
            ] + [{"id": 15, "finished": False, "is_current": True, "is_next": False}],
            "teams": [
                {
                    "id": t, "code": 1000 + t, "name": f"Team {t}", "short_name": f"T{t}",
                    "strength": 3, "strength_attack_home": 1000, "strength_attack_away": 1000,
                    "strength_defence_home": 1000, "strength_defence_away": 1000
                }
                for t in range(1, 4)
            ],
            "elements": [
                # 3 players (1 GK, 1 DEF, 1 MID)
                {
                    "id": 1, "code": 101, "first_name": "Bukayo", "second_name": "Saka",
                    "web_name": "Saka", "element_type": 3, "team": 1, "now_cost": 105,
                    "selected_by_percent": "25.0", "transfers_in_event": 50, "transfers_out_event": 10
                },
                {
                    "id": 2, "code": 102, "first_name": "Erling", "second_name": "Haaland",
                    "web_name": "Haaland", "element_type": 4, "team": 2, "now_cost": 150,
                    "selected_by_percent": "50.0", "transfers_in_event": 100, "transfers_out_event": 20
                },
                {
                    "id": 3, "code": 103, "first_name": "Gabriel", "second_name": "Magalhaes",
                    "web_name": "Gabriel", "element_type": 2, "team": 1, "now_cost": 60,
                    "selected_by_percent": "15.0", "transfers_in_event": 20, "transfers_out_event": 5
                }
            ]
        }
        
        # 2. Mock Fixtures list (GW 1 to 15)
        self.mock_fixtures = []
        fixture_id = 1
        for g in range(1, 16):
            # 2 fixtures per GW
            self.mock_fixtures.append({
                "id": fixture_id, "event": g, "team_h": 1, "team_a": 2,
                "team_h_score": 2, "team_a_score": 1, "finished": g < 15,
                "minutes": 90, "team_h_difficulty": 3, "team_a_difficulty": 4
            })
            fixture_id += 1
            self.mock_fixtures.append({
                "id": fixture_id, "event": g, "team_h": 3, "team_a": 1,
                "team_h_score": 0, "team_a_score": 0, "finished": g < 15,
                "minutes": 90, "team_h_difficulty": 4, "team_a_difficulty": 3
            })
            fixture_id += 1
            
        # 3. Mock Live Gameweek stats (for each finished GW)
        self.mock_live_gw = {}
        for g in range(1, 15):
            self.mock_live_gw[g] = {
                "elements": [
                    {
                        "id": pid,
                        "stats": {"transfers_in": 10, "transfers_out": 2},
                        "explain": [
                            {
                                "fixture": g * 2 - 1, # Mock fixture ID matching team
                                "stats": [
                                    {"identifier": "minutes", "points": 2, "value": 90},
                                    {"identifier": "total_points", "points": 5, "value": 5},
                                    {"identifier": "goals_scored", "points": 4, "value": 1},
                                    {"identifier": "expected_goals", "points": 0, "value": "0.45"},
                                    {"identifier": "expected_assists", "points": 0, "value": "0.12"}
                                ]
                            }
                        ]
                    }
                    for pid in [1, 2, 3]
                ]
            }

    @patch("src.data.update_current.fetch_json")
    def test_update_current_pipeline_execution(self, mock_fetch):
        """Test FPL API Ingestion, Current features creation, compatibility, and training preservation."""
        # Setup mocks
        def side_effect(url):
            if "bootstrap-static" in url:
                return self.mock_bootstrap
            elif "fixtures" in url:
                return self.mock_fixtures
            elif "/live/" in url:
                # Extract gameweek
                import re
                match = re.search(r"event/(\d+)/live", url)
                gw = int(match.group(1))
                return self.mock_live_gw[gw]
            return {}
            
        mock_fetch.side_effect = side_effect
        
        # 1. Capture hash of historical features_df.parquet before run
        hist_path = PROCESSED_DIR / "features_df.parquet"
        self.assertTrue(hist_path.exists(), "Historical training dataset features_df.parquet must exist")
        with open(hist_path, "rb") as f:
            initial_hash = hashlib.sha256(f.read()).hexdigest()
            
        # 2. Run update pipeline for GW 15
        run_update_current_pipeline(self.season, self.gw)
        
        # 3. Assert current_features.parquet exists
        curr_path = PROCESSED_DIR / "current_features.parquet"
        self.assertTrue(curr_path.exists(), "Inference features file current_features.parquet was not created")
        
        # 4. Check that features_df.parquet was NOT mutated
        with open(hist_path, "rb") as f:
            final_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(initial_hash, final_hash, "ERROR: The historical training dataset features_df.parquet was mutated!")
        
        # 5. Load generated current features and assert schema compatibility
        curr_df = pd.read_parquet(curr_path)
        hist_df = pd.read_parquet(hist_path)
        
        # Verify columns match
        # (excluding targets which are mock placeholders in current_features)
        feature_cols = [c for c in hist_df.columns if c not in ["target_points", "target_minutes", "target_60_plus_minutes"]]
        for col in feature_cols:
            self.assertIn(col, curr_df.columns, f"Required column {col} missing in inference features")
            
        # Verify rows only contain season/gw cutoff target
        self.assertTrue((curr_df["season"] == self.season).all())
        self.assertTrue((curr_df["gw"] == self.gw).all())
        
        # Verify target GW (GW 15) has NO leakage: actual current GW stats must be empty or missing
        self.assertNotIn("total_points", curr_df.columns)
        self.assertNotIn("minutes", curr_df.columns)
        
        # 6. Verify Player ID Mapping creation
        mapping_path = PROCESSED_DIR / "player_id_mapping.parquet"
        fpl_sub = curr_df[["code", "name"]].drop_duplicates()
        build_player_id_mapping(fpl_sub)
        self.assertTrue(mapping_path.exists(), "Player ID mapping table was not created")
        mapping_df = pd.read_parquet(mapping_path)
        self.assertEqual(len(mapping_df), len(fpl_sub))
        self.assertIn("fpl_code", mapping_df.columns)
        self.assertIn("fbref_id", mapping_df.columns)

if __name__ == "__main__":
    unittest.main()
