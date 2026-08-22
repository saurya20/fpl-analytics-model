import unittest
import pandas as pd
import numpy as np
import json
import hashlib
from unittest.mock import patch
from pathlib import Path
from src.data.update_current import run_update_current_pipeline
from src.features.current_workload import compute_workload_features
from src.data.player_mapping import build_player_id_mapping

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

class TestCurrentWorkload(unittest.TestCase):
    
    def setUp(self):
        """Set up FPL actuals and workload mocks."""
        self.season = "2024-25"
        self.gw = 15
        
        # Target GW 15 deadline: 2024-11-23T11:00:00Z
        self.mock_bootstrap = {
            "events": [
                {"id": g, "finished": True, "deadline_time": f"2024-11-{10+g}T11:00:00Z"}
                for g in range(1, 15)
            ] + [{"id": 15, "finished": False, "deadline_time": "2024-11-23T11:00:00Z"}],
            "teams": [
                {"id": 1, "code": 1001, "name": "Team 1", "short_name": "T1",
                 "strength": 3, "strength_attack_home": 1000, "strength_attack_away": 1000,
                 "strength_defence_home": 1000, "strength_defence_away": 1000}
            ],
            "elements": [
                {"id": 1, "code": 101, "first_name": "Bukayo", "second_name": "Saka",
                 "web_name": "Saka", "element_type": 3, "team": 1, "now_cost": 105,
                 "selected_by_percent": "25.0", "transfers_in_event": 50, "transfers_out_event": 10}
            ]
        }
        
        self.mock_fixtures = [
            # Past fixtures (event < 15, finished = True)
            {"id": 1, "event": 14, "team_h": 1, "team_a": 2, "team_h_score": 1, "team_a_score": 0,
             "finished": True, "minutes": 90, "kickoff_time": "2024-11-22T12:00:00Z"},
            # Upcoming GW 15 fixture (event = 15, kickoff after deadline)
            {"id": 2, "event": 15, "team_h": 1, "team_a": 3, "team_h_score": None, "team_a_score": None,
             "finished": False, "minutes": 0, "kickoff_time": "2024-11-23T15:00:00Z"},
            # Future fixture (event = 16)
            {"id": 3, "event": 16, "team_h": 2, "team_a": 1, "team_h_score": None, "team_a_score": None,
             "finished": False, "minutes": 0, "kickoff_time": "2024-11-30T15:00:00Z"}
        ]
        
        self.mock_live_gw = {}
        for g in range(1, 15):
            self.mock_live_gw[g] = {
                "elements": [
                    {
                        "id": 1,
                        "stats": {"transfers_in": 10, "transfers_out": 2},
                        "explain": [
                            {
                                "fixture": 1,
                                "stats": [
                                    {"identifier": "minutes", "points": 2, "value": 90},
                                    {"identifier": "total_points", "points": 2, "value": 2}
                                ]
                            }
                        ]
                    }
                ]
            }
            
        # Mock Cup Matches:
        # Match A: Before Cutoff (10 days before) -> should be counted!
        # Match B: After Cutoff (1 day after target GW deadline) -> MUST be excluded (zero leakage check)!
        self.mock_external_matches = [
            {
                "fbref_id": "ext_101",
                "fbref_name": "Bukayo Saka",
                "date": "2024-11-13T20:00:00Z", # 10 days before cutoff
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
                "date": "2024-11-24T20:00:00Z", # 1 day AFTER target GW 15 deadline!
                "competition": "Champions League",
                "opponent": "Real Madrid",
                "minutes": 90,
                "started": 1,
                "goals": 2,
                "assists": 1
            }
        ]

    @patch("src.data.update_current.fetch_json")
    def test_workload_point_in_time_exclusion(self, mock_fetch):
        """Test Point-in-Time: verify external cup matches after the target deadline are excluded, preventing leakage."""
        def side_effect(url):
            if "bootstrap-static" in url:
                return self.mock_bootstrap
            elif "fixtures" in url:
                return self.mock_fixtures
            elif "/live/" in url:
                import re
                match = re.search(r"event/(\d+)/live", url)
                gw = int(match.group(1))
                return self.mock_live_gw[gw]
            return {}
            
        mock_fetch.side_effect = side_effect
        
        # Capture training data hash before run
        hist_path = PROCESSED_DIR / "features_df.parquet"
        with open(hist_path, "rb") as f:
            initial_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Write player id mapping for test
        fpl_players = pd.DataFrame([{"code": 101, "name": "Bukayo Saka", "team": 1}])
        build_player_id_mapping(fpl_players)
            
        # Write mock cup matches to file
        ext_matches_path = RAW_DIR / "matches" / "external_cup_matches.json"
        ext_matches_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ext_matches_path, "w") as f:
            json.dump(self.mock_external_matches, f, indent=4)
            
        # Run Ingestion
        run_update_current_pipeline(self.season, self.gw)
        
        # Load generated current workload
        workload_path = PROCESSED_DIR / "current_player_workload.parquet"
        self.assertTrue(workload_path.exists())
        workload_df = pd.read_parquet(workload_path)
        
        # Bukayo Saka code is 101
        saka_row = workload_df[workload_df["code"] == 101]
        self.assertFalse(saka_row.empty)
        
        # Assertions on match cutoff
        # Saka played 90 minutes before cutoff (Sporting CP) and 90 minutes after (Real Madrid)
        # Only Sporting CP (90 mins, 1 goal, 0 assists) should enter rolling workload!
        # The 90 mins, 2 goals, 1 assist from Real Madrid (after deadline) MUST be excluded!
        self.assertEqual(float(saka_row["external_minutes_last_14d"].values[0]), 90.0)
        self.assertEqual(int(saka_row["external_goals_last_14d"].values[0]), 1)
        self.assertEqual(int(saka_row["external_assists_last_14d"].values[0]), 0)
        
        # Ensure training features_df was not mutated
        with open(hist_path, "rb") as f:
            final_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(initial_hash, final_hash)
        
        # Verify schema compatibility of current_features.parquet
        curr_path = PROCESSED_DIR / "current_features.parquet"
        curr_df = pd.read_parquet(curr_path)
        self.assertIn("external_minutes_last_14d", curr_df.columns)
        self.assertIn("fixture_congestion_score", curr_df.columns)
        
        # Verify no future target stats leaked
        self.assertNotIn("total_points", curr_df.columns)
        
    @patch("src.features.current_workload.load_json")
    def test_missing_external_data_does_not_crash(self, mock_load):
        """Test that missing or malformed external matches JSON does not crash the pipeline."""
        mock_load.return_value = None # Simulates file missing or empty
        
        players_df = pd.DataFrame([{"player_id": 1, "code": 101, "name": "Bukayo Saka", "team": 1}])
        fixtures_df = pd.DataFrame([{"id": 1, "event": 14, "team_h": 1, "team_a": 2, "finished": True, "event": 14}])
        player_gw_df = pd.DataFrame([{"player_id": 1, "gw": 14, "fixture": 1, "minutes": 90}])
        teams_df = pd.DataFrame([{"id": 1, "name": "Arsenal", "season": self.season}])
        
        try:
            workload_df = compute_workload_features(
                season=self.season,
                target_gw=self.gw,
                player_gw_df=player_gw_df,
                players_df=players_df,
                fixtures_df=fixtures_df,
                teams_df=teams_df,
                bootstrap_raw=self.mock_bootstrap
            )
            # Should run successfully with 0.0 defaults
            self.assertEqual(float(workload_df[workload_df["code"] == 101]["external_minutes_last_14d"].values[0]), 0.0)
        except Exception as e:
            self.fail(f"compute_workload_features crashed on missing external data: {e}")
            
    def test_player_mapping_ambiguity_and_collisions(self):
        """Test mapping logic warnings for ambiguous matches and identity collisions."""
        # Create FPL players pool
        fpl_players = pd.DataFrame([
            {"code": 101, "name": "Bukayo Saka", "team": 1},
            {"code": 102, "name": "Declan Rice", "team": 1},
            {"code": 103, "name": "Gabriel Martinelli", "team": 1}
        ])
        
        # Create external players pool with a duplicate ID to force a collision
        # and ambiguous entries to force ambiguity fallback
        ext_players = pd.DataFrame([
            {"fbref_id": "ext_collision", "fbref_name": "Bukayo Saka", "team": "1"},
            {"fbref_id": "ext_collision", "fbref_name": "Declan Rice", "team": "1"}, # Collision on ext_collision ID
            {"fbref_id": "ext_ambig_1", "fbref_name": "Gabriel", "team": "1"},
            {"fbref_id": "ext_ambig_2", "fbref_name": "Gabriel", "team": "1"} # Multiple Gabriel matches for Gabriel Martinelli
        ])
        
        import io
        import sys
        
        # Redirect stdout to capture printed warnings
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            mapping_df = build_player_id_mapping(fpl_players, ext_players)
            captured = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            
        # Assertions on warnings being logged
        self.assertIn("Player identity collision detected", captured)
        self.assertIn("Ambiguous identity matches", captured)
        
        # Assert mapping status correctness
        saka_map = mapping_df[mapping_df["fpl_code"] == 101].iloc[0]
        self.assertEqual(saka_map["fbref_id"], "ext_collision")
        self.assertEqual(saka_map["confidence"], "exact")
        
        martinelli_map = mapping_df[mapping_df["fpl_code"] == 103].iloc[0]
        # Ambiguous match should fall back to unmatched fallback code
        self.assertEqual(martinelli_map["fbref_id"], "ext_103")
        self.assertEqual(martinelli_map["confidence"], "ambiguous_fallback")

if __name__ == "__main__":
    unittest.main()
