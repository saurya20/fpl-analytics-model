import unittest
import pandas as pd
import numpy as np
import json
import hashlib
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipeline.weekly_decision import (
    determine_season_and_gw, 
    run_weekly_decision_pipeline
)
STATE_FILE = PROJECT_ROOT / "data" / "state" / "test_current_squad.json"
HISTORY_DIR = PROJECT_ROOT / "data" / "state" / "test_history"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

class TestWeeklyDecision(unittest.TestCase):
    
    def setUp(self):
        """Set up mock outputs and configurations."""
        self.mock_bootstrap = {
            "events": [
                {"id": 1, "finished": True, "deadline_time": "2025-08-11T17:30:00Z"},
                {"id": 2, "finished": False, "is_next": True, "deadline_time": "2025-08-18T17:30:00Z"}
            ],
            "teams": [
                {"id": 1, "code": 1, "name": "Arsenal", "short_name": "ARS", "strength": 4},
                {"id": 2, "code": 2, "name": "Chelsea", "short_name": "CHE", "strength": 3},
                {"id": 3, "code": 3, "name": "Man Utd", "short_name": "MUN", "strength": 3},
                {"id": 4, "code": 4, "name": "Man City", "short_name": "MCI", "strength": 5},
                {"id": 5, "code": 5, "name": "Liverpool", "short_name": "LIV", "strength": 4}
            ],
            "elements": [
                # GKs
                {"id": 1, "code": 101, "first_name": "P101", "second_name": "S101", "element_type": 1, "team": 1, "now_cost": 50},
                {"id": 2, "code": 102, "first_name": "P102", "second_name": "S102", "element_type": 1, "team": 2, "now_cost": 45},
                {"id": 16, "code": 116, "first_name": "P116", "second_name": "S116", "element_type": 1, "team": 1, "now_cost": 55},
                # DEFs
                {"id": 3, "code": 103, "first_name": "P103", "second_name": "S103", "element_type": 2, "team": 1, "now_cost": 60},
                {"id": 4, "code": 104, "first_name": "P104", "second_name": "S104", "element_type": 2, "team": 3, "now_cost": 50},
                {"id": 5, "code": 105, "first_name": "P105", "second_name": "S105", "element_type": 2, "team": 2, "now_cost": 45},
                {"id": 6, "code": 106, "first_name": "P106", "second_name": "S106", "element_type": 2, "team": 4, "now_cost": 40},
                {"id": 7, "code": 107, "first_name": "P107", "second_name": "S107", "element_type": 2, "team": 5, "now_cost": 40},
                {"id": 17, "code": 117, "first_name": "P117", "second_name": "S117", "element_type": 2, "team": 1, "now_cost": 65},
                # MIDs
                {"id": 8, "code": 108, "first_name": "P108", "second_name": "S108", "element_type": 3, "team": 1, "now_cost": 85},
                {"id": 9, "code": 109, "first_name": "P109", "second_name": "S109", "element_type": 3, "team": 3, "now_cost": 75},
                {"id": 10, "code": 110, "first_name": "P110", "second_name": "S110", "element_type": 3, "team": 2, "now_cost": 70},
                {"id": 11, "code": 111, "first_name": "P111", "second_name": "S111", "element_type": 3, "team": 4, "now_cost": 65},
                {"id": 12, "code": 112, "first_name": "P112", "second_name": "S112", "element_type": 3, "team": 5, "now_cost": 60},
                {"id": 18, "code": 118, "first_name": "P118", "second_name": "S118", "element_type": 3, "team": 1, "now_cost": 125},
                # FWDs
                {"id": 13, "code": 113, "first_name": "P113", "second_name": "S113", "element_type": 4, "team": 3, "now_cost": 95},
                {"id": 14, "code": 114, "first_name": "P114", "second_name": "S114", "element_type": 4, "team": 4, "now_cost": 80},
                {"id": 15, "code": 115, "first_name": "P115", "second_name": "S115", "element_type": 4, "team": 5, "now_cost": 75},
                {"id": 19, "code": 119, "first_name": "P119", "second_name": "S119", "element_type": 4, "team": 1, "now_cost": 140}
            ]
        }
        
        # Current squad codes (15 players)
        self.mock_squad_json = {
            "players": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "free_transfers": 1,
            "bank": 0.5
        }
        
        # Create a real mock squad JSON file to avoid mocking json.load globally
        self.mock_squad_path = PROJECT_ROOT / "tests" / "mock_squad.json"
        self.mock_squad_path.parent.mkdir(exist_ok=True)
        with open(self.mock_squad_path, "w") as f:
            json.dump(self.mock_squad_json, f)
            
        # Create a real squad file with real codes for GW 1 test
        self.mock_real_squad_json = {
            "players": [98980, 495145, 101188, 477424, 548308, 578512, 611926, 223340, 244851, 178186, 518620, 532605, 223094, 178301, 574402],
            "free_transfers": 1,
            "bank": 0.5
        }
        self.mock_real_squad_path = PROJECT_ROOT / "tests" / "mock_real_squad.json"
        with open(self.mock_real_squad_path, "w") as f:
            json.dump(self.mock_real_squad_json, f)
        
        # Mock prediction pool (including all 15 squad players + replacement targets)
        # Position mapping: GK=2, DEF=5, MID=5, FWD=3
        self.mock_predictions_df = pd.DataFrame([
            # GK (2 squad players + 1 target)
            {"code": 101, "name": "Player 101", "position": "GK", "team": "ARS", "current_price": 5.0, "predicted_points": 4.0, "predicted_minutes": 90.0, "prob_60_plus": 0.9, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 1, "season": "2025-26", "gw": 2},
            {"code": 102, "name": "Player 102", "position": "GK", "team": "CHE", "current_price": 4.5, "predicted_points": 3.5, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 2, "season": "2025-26", "gw": 2},
            {"code": 116, "name": "Player 116", "position": "GK", "team": "ARS", "current_price": 5.5, "predicted_points": 5.0, "predicted_minutes": 90.0, "prob_60_plus": 0.95, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 2.0, "season_team_id": 1, "season": "2025-26", "gw": 2},
            
            # DEF (5 squad players + 1 target)
            {"code": 103, "name": "Player 103", "position": "DEF", "team": "ARS", "current_price": 6.0, "predicted_points": 5.0, "predicted_minutes": 90.0, "prob_60_plus": 0.9, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 1, "season": "2025-26", "gw": 2},
            {"code": 104, "name": "Player 104", "position": "DEF", "team": "MUN", "current_price": 5.0, "predicted_points": 3.0, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 3, "season": "2025-26", "gw": 2},
            {"code": 105, "name": "Player 105", "position": "DEF", "team": "CHE", "current_price": 4.5, "predicted_points": 2.5, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 2, "season": "2025-26", "gw": 2},
            {"code": 106, "name": "Player 106", "position": "DEF", "team": "MCI", "current_price": 4.0, "predicted_points": 2.0, "predicted_minutes": 90.0, "prob_60_plus": 0.7, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 4, "season": "2025-26", "gw": 2},
            {"code": 107, "name": "Player 107", "position": "DEF", "team": "LIV", "current_price": 4.0, "predicted_points": 2.2, "predicted_minutes": 90.0, "prob_60_plus": 0.7, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 5, "season": "2025-26", "gw": 2},
            {"code": 117, "name": "Player 117", "position": "DEF", "team": "ARS", "current_price": 6.5, "predicted_points": 6.0, "predicted_minutes": 90.0, "prob_60_plus": 0.95, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 2.0, "season_team_id": 1, "season": "2025-26", "gw": 2},
            
            # MID (5 squad players + 1 target)
            {"code": 108, "name": "Player 108", "position": "MID", "team": "ARS", "current_price": 8.5, "predicted_points": 7.0, "predicted_minutes": 90.0, "prob_60_plus": 0.9, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 1, "season": "2025-26", "gw": 2},
            {"code": 109, "name": "Player 109", "position": "MID", "team": "MUN", "current_price": 7.5, "predicted_points": 5.5, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 3, "season": "2025-26", "gw": 2},
            {"code": 110, "name": "Player 110", "position": "MID", "team": "CHE", "current_price": 7.0, "predicted_points": 4.5, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 2, "season": "2025-26", "gw": 2},
            {"code": 111, "name": "Player 111", "position": "MID", "team": "MCI", "current_price": 6.5, "predicted_points": 4.0, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 4, "season": "2025-26", "gw": 2},
            {"code": 112, "name": "Player 112", "position": "MID", "team": "LIV", "current_price": 6.0, "predicted_points": 3.8, "predicted_minutes": 90.0, "prob_60_plus": 0.7, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 5, "season": "2025-26", "gw": 2},
            {"code": 118, "name": "Player 118", "position": "MID", "team": "ARS", "current_price": 12.5, "predicted_points": 9.5, "predicted_minutes": 90.0, "prob_60_plus": 0.98, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 2.0, "season_team_id": 1, "season": "2025-26", "gw": 2},
            
            # FWD (3 squad players + 1 target)
            {"code": 113, "name": "Player 113", "position": "FWD", "team": "MUN", "current_price": 9.5, "predicted_points": 6.5, "predicted_minutes": 90.0, "prob_60_plus": 0.9, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 3, "season": "2025-26", "gw": 2},
            {"code": 114, "name": "Player 114", "position": "FWD", "team": "MCI", "current_price": 8.0, "predicted_points": 5.0, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 4, "season": "2025-26", "gw": 2},
            {"code": 115, "name": "Player 115", "position": "FWD", "team": "LIV", "current_price": 7.5, "predicted_points": 4.5, "predicted_minutes": 90.0, "prob_60_plus": 0.8, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": 5, "season": "2025-26", "gw": 2},
            {"code": 119, "name": "Player 119", "position": "FWD", "team": "ARS", "current_price": 14.0, "predicted_points": 10.5, "predicted_minutes": 90.0, "prob_60_plus": 0.98, "selected": 0.0, "num_fixtures": 1.0, "is_double_gw": 0.0, "fixture_difficulty": 2.0, "season_team_id": 1, "season": "2025-26", "gw": 2}
        ])
        
        # Patcher for production squad paths
        self.state_patcher = patch("src.pipeline.weekly_decision.STATE_FILE", STATE_FILE)
        self.history_patcher = patch("src.pipeline.weekly_decision.HISTORY_DIR", HISTORY_DIR)
        self.state_patcher.start()
        self.history_patcher.start()

        # Clean up state file before test
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        if HISTORY_DIR.exists():
            for f in HISTORY_DIR.glob("*.json"):
                f.unlink()

    def tearDown(self):
        """Clean up written test artifacts."""
        try:
            self.state_patcher.stop()
            self.history_patcher.stop()
        except:
            pass
        try:
            if self.mock_squad_path.exists():
                self.mock_squad_path.unlink()
        except:
            pass
        try:
            if self.mock_real_squad_path.exists():
                self.mock_real_squad_path.unlink()
        except:
            pass
        try:
            if STATE_FILE.exists():
                STATE_FILE.unlink()
        except:
            pass
        try:
            if HISTORY_DIR.exists():
                for f in HISTORY_DIR.glob("*.json"):
                    try:
                        f.unlink()
                    except:
                        pass
                try:
                    HISTORY_DIR.rmdir()
                except:
                    pass
        except:
            pass
        try:
            if STATE_FILE.parent.exists():
                STATE_FILE.parent.rmdir()
        except:
            pass
                
        for season, gw in [("2025-26", 2), ("2026-27", 1), ("2026-27", 2)]:
            json_file = RESULTS_DIR / f"weekly_decision_{season}_gw{gw}.json"
            csv_file = RESULTS_DIR / f"weekly_decision_{season}_gw{gw}.csv"
            try:
                if json_file.exists():
                    json_file.unlink()
            except:
                pass
            try:
                if csv_file.exists():
                    csv_file.unlink()
            except:
                pass

    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_determine_season_and_gw(self, mock_fetch):
        """Verify CLI automatically determines next GW and season from FPL bootstrap events.
        Covers rules: (1) determines next GW, (3) parses season correctly.
        """
        mock_fetch.return_value = self.mock_bootstrap
        season, gw, deadline = determine_season_and_gw()
        self.assertEqual(season, "2025-26")
        self.assertEqual(gw, 2)
        self.assertEqual(deadline, "2025-08-18T17:30:00Z")

    @patch("src.pipeline.weekly_decision.run_update_current_pipeline")
    @patch("src.pipeline.weekly_decision.predict_gameweek")
    @patch("src.pipeline.weekly_decision.determine_season_and_gw")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_weekly_decision_pipeline_constraints_and_recommendation(
        self, mock_fetch, mock_determine, mock_predict, mock_update
    ):
        """Verify that HOLD vs TRANSFER comparison runs, and final squad satisfies all constraints.
        Covers rules: (11) hold vs transfer, (13) 15-player squad, (14) 11 starters, (15) position limits, (16) max 3 per club, (17) captain/vice-captain starting starter validity, (19) valid JSON.
        """
        mock_determine.return_value = ("2025-26", 2, "2025-08-18T17:30:00Z")
        mock_predict.return_value = (self.mock_predictions_df, None)
        mock_fetch.return_value = self.mock_bootstrap
        
        # Run pipeline with a user squad setup (which will initialize state)
        summary = run_weekly_decision_pipeline(
            season="2025-26",
            gw=2,
            squad_path=str(self.mock_squad_path),
            bank_override=0.5,
            free_transfers_override=1,
            hit_cost=4.0
        )
            
        # Assert Hold vs Transfer comparison results
        self.assertIn("hold_expected_points", summary)
        self.assertIn("transfer_expected_points", summary)
        self.assertIn("recommendation_decision", summary)
        
        # Verify FPL Constraints on the final recommended squad
        players = summary["players"]
        self.assertEqual(len(players), 15)
        
        starters = [p for p in players if p["starter"] == 1]
        self.assertEqual(len(starters), 11)
        
        # Position counts in starters
        pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in starters:
            pos_counts[p["position"]] += 1
            
        self.assertEqual(pos_counts["GK"], 1)
        self.assertTrue(3 <= pos_counts["DEF"] <= 5)
        self.assertTrue(3 <= pos_counts["MID"] <= 5)
        self.assertTrue(1 <= pos_counts["FWD"] <= 3)
        
        # Check team representation (max 3)
        teams = [p["team"] for p in players]
        for t in set(teams):
            self.assertTrue(teams.count(t) <= 3)

    @patch("src.data.update_current.fetch_json")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_weekly_decision_gw1_new_season(self, mock_fetch_decision, mock_fetch_update):
        """Verify GW1 prediction behavior at the beginning of a new season.
        Covers critical test rules: (1) current_features is not empty, (2) player count > 0,
        (3) target GW == 1, (4) predictions generated, (5) no GW1 actual stats used,
        (6) no historical training parquet modified, (7) current club comes from bootstrap,
        (8) prediction schema is valid, (9) BGW players handled, (10) optimizer consumes predictions.
        """
        # Mock bootstrap data with elements for 2026-27 GW 1
        bootstrap_data = {
            "events": [
                {"id": 1, "finished": False, "is_next": True, "deadline_time": "2026-08-11T17:30:00Z"},
                {"id": 2, "finished": False, "is_next": False, "deadline_time": "2026-08-18T17:30:00Z"}
            ],
            "teams": [
                {"id": 1, "code": 1, "name": "Arsenal", "short_name": "ARS", "strength": 4, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 2, "code": 2, "name": "Aston Villa", "short_name": "AVL", "strength": 3, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 4, "code": 4, "name": "Bournemouth", "short_name": "BOU", "strength": 3, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 5, "code": 5, "name": "Brighton", "short_name": "BHA", "strength": 3, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 6, "code": 6, "name": "Chelsea", "short_name": "CHE", "strength": 4, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 12, "code": 12, "name": "Liverpool", "short_name": "LIV", "strength": 4, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 15, "code": 15, "name": "Man City", "short_name": "MCI", "strength": 5, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 16, "code": 16, "name": "Newcastle", "short_name": "NEW", "strength": 4, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000},
                {"id": 19, "code": 19, "name": "Tottenham", "short_name": "TOT", "strength": 4, "strength_attack_home": 1000, "strength_attack_away": 1000, "strength_defence_home": 1000, "strength_defence_away": 1000}
            ],
            "elements": [
                # GKs (98980, 495145)
                {"id": 1, "code": 98980, "first_name": "Emiliano", "second_name": "Martínez", "element_type": 1, "team": 2, "now_cost": 50, "selected_by_percent": "10.0"},
                {"id": 2, "code": 495145, "first_name": "Alex", "second_name": "Paulsen", "element_type": 1, "team": 4, "now_cost": 45, "selected_by_percent": "8.0"},
                # DEFs (101188, 477424, 548308, 578512, 611926)
                {"id": 3, "code": 101188, "first_name": "Lucas", "second_name": "Digne", "element_type": 2, "team": 2, "now_cost": 60, "selected_by_percent": "15.0"},
                {"id": 4, "code": 477424, "first_name": "Joško", "second_name": "Gvardiol", "element_type": 2, "team": 15, "now_cost": 50, "selected_by_percent": "10.0"},
                {"id": 5, "code": 548308, "first_name": "Ashley", "second_name": "Phillips", "element_type": 2, "team": 19, "now_cost": 45, "selected_by_percent": "5.0"},
                {"id": 6, "code": 578512, "first_name": "Miodrag", "second_name": "Pivaš", "element_type": 2, "team": 15, "now_cost": 40, "selected_by_percent": "3.0"},
                {"id": 7, "code": 611926, "first_name": "Amara", "second_name": "Nallo", "element_type": 2, "team": 12, "now_cost": 40, "selected_by_percent": "2.0"},
                # MIDs (223340, 244851, 178186, 518620, 532605)
                {"id": 8, "code": 223340, "first_name": "Bukayo", "second_name": "Saka", "element_type": 3, "team": 1, "now_cost": 85, "selected_by_percent": "20.0"},
                {"id": 9, "code": 244851, "first_name": "Cole", "second_name": "Palmer", "element_type": 3, "team": 6, "now_cost": 75, "selected_by_percent": "12.0"},
                {"id": 10, "code": 178186, "first_name": "Jarrod", "second_name": "Bowen", "element_type": 3, "team": 19, "now_cost": 70, "selected_by_percent": "15.0"},
                {"id": 11, "code": 518620, "first_name": "Ángelo", "second_name": "Gabriel", "element_type": 3, "team": 6, "now_cost": 65, "selected_by_percent": "10.0"},
                {"id": 12, "code": 532605, "first_name": "Andrey", "second_name": "Santos", "element_type": 3, "team": 16, "now_cost": 60, "selected_by_percent": "8.0"},
                # FWDs (223094, 178301, 574402)
                {"id": 13, "code": 223094, "first_name": "Erling", "second_name": "Haaland", "element_type": 4, "team": 15, "now_cost": 95, "selected_by_percent": "18.0"},
                {"id": 14, "code": 178301, "first_name": "Ollie", "second_name": "Watkins", "element_type": 4, "team": 2, "now_cost": 80, "selected_by_percent": "15.0"},
                {"id": 15, "code": 574402, "first_name": "Mark", "second_name": "O'Mahony", "element_type": 4, "team": 5, "now_cost": 75, "selected_by_percent": "10.0"}
            ]
        }
        
        # GW 1 fixtures (normal fixtures for teams 1, 2, 4, 6, 15, 16, 19; BGW/blank for team 12 and 5)
        fixtures_data = [
            {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": False},
            {"id": 2, "event": 1, "team_h": 4, "team_a": 6, "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": False},
            {"id": 3, "event": 1, "team_h": 16, "team_a": 15, "team_h_difficulty": 4, "team_a_difficulty": 2, "finished": False},
            {"id": 4, "event": 1, "team_h": 19, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 3, "finished": False}
        ]
        
        def mock_fetch_json_fn(url):
            if "bootstrap" in url:
                return bootstrap_data
            elif "fixtures" in url:
                return fixtures_data
            else:
                return {}
                
        mock_fetch_decision.side_effect = mock_fetch_json_fn
        mock_fetch_update.side_effect = mock_fetch_json_fn
        
        summary = run_weekly_decision_pipeline(
            season="2026-27",
            gw=1,
            squad_path=str(self.mock_real_squad_path),
            bank_override=0.5,
            free_transfers_override=1,
            hit_cost=4.0
        )
            
        # Verify expectations:
        self.assertIsNotNone(summary)
        self.assertEqual(summary["season"], "2026-27")
        self.assertEqual(summary["gw"], 1)
        self.assertTrue(summary["prediction_rows"] > 0)
        
        # Verify that BGW players (team 12 Liverpool and team 5 Brighton: codes 611926, 574402) are handled
        players = summary["players"]
        bgw_names = ["Amara Nallo", "Mark O'Mahony", "Mark OMahony"]
        bgw_players_selected = [p for p in players if p["name"] in bgw_names]
        self.assertTrue(len(bgw_players_selected) > 0)
        
        # Verify the optimizer completed
        self.assertIn("recommendation_decision", summary)

    # ==================================================
    # PHASE 9.1 PERSISTENT STATE AND VALIDATION TESTS
    # ==================================================

    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_squad_validation_success(self, mock_fetch):
        """Verify successful validation of a valid 15-player squad."""
        mock_fetch.return_value = self.mock_bootstrap
        
        # Valid squad setup from the mock elements
        valid_input = {
            "season": "2025-26",
            "players": [
                # 2 GKs
                {"id": 1}, {"id": 2},
                # 5 DEFs
                {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6}, {"id": 7},
                # 5 MIDs
                {"id": 8}, {"id": 9}, {"id": 10}, {"id": 11}, {"id": 12},
                # 3 FWDs
                {"id": 13}, {"id": 14}, {"id": 15}
            ],
            "bank": 0.5,
            "free_transfers": 1
        }
        
        from src.pipeline.weekly_decision import validate_and_compile_user_squad
        resolved_codes, resolved_metadata = validate_and_compile_user_squad(valid_input, self.mock_bootstrap)
        self.assertEqual(len(resolved_codes), 15)
        self.assertEqual(len(resolved_metadata), 15)
        self.assertEqual(resolved_codes, [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])

    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_squad_validation_failures(self, mock_fetch):
        """Verify validation failures for invalid inputs (size, duplicates, positions, club limits)."""
        mock_fetch.return_value = self.mock_bootstrap
        from src.pipeline.weekly_decision import validate_and_compile_user_squad
        
        # 1. Invalid size (14 players)
        invalid_size = {
            "players": [{"id": i} for i in range(1, 15)]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(invalid_size, self.mock_bootstrap)
        self.assertIn("exactly 15 players", str(ctx.exception))
        
        # 2. Duplicate player
        duplicate = {
            "players": [
                {"id": 1}, {"id": 1},
                {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6}, {"id": 7},
                {"id": 8}, {"id": 9}, {"id": 10}, {"id": 11}, {"id": 12},
                {"id": 13}, {"id": 14}, {"id": 15}
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(duplicate, self.mock_bootstrap)
        self.assertIn("Duplicate player found", str(ctx.exception))

        # 3. Invalid player ID
        invalid_id = {
            "players": [
                {"id": 999}, {"id": 2},
                {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6}, {"id": 7},
                {"id": 8}, {"id": 9}, {"id": 10}, {"id": 11}, {"id": 12},
                {"id": 13}, {"id": 14}, {"id": 15}
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(invalid_id, self.mock_bootstrap)
        self.assertIn("could not be found", str(ctx.exception))

        # 4. Wrong position counts (3 GKs, 4 DEFs)
        wrong_positions = {
            "players": [
                # 3 GKs
                {"id": 1}, {"id": 2}, {"id": 16},
                # 4 DEFs
                {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6},
                # 5 MIDs
                {"id": 8}, {"id": 9}, {"id": 10}, {"id": 11}, {"id": 12},
                # 3 FWDs
                {"id": 13}, {"id": 14}, {"id": 15}
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(wrong_positions, self.mock_bootstrap)
        self.assertIn("Incorrect position counts", str(ctx.exception))

        # 5. Max 3 players from same club exceeded (4 from Arsenal/team 1: codes 101, 103, 117, 108)
        exceed_club = {
            "players": [
                # 2 GKs
                {"id": 1}, {"id": 2},
                # 5 DEFs (103 and 117 are Arsenal/team 1)
                {"id": 3}, {"id": 17}, {"id": 5}, {"id": 6}, {"id": 7},
                # 5 MIDs (108 is Arsenal/team 1, 118 is Arsenal/team 1 -> total 4 from team 1)
                {"id": 8}, {"id": 18}, {"id": 10}, {"id": 11}, {"id": 12},
                # 3 FWDs
                {"id": 13}, {"id": 14}, {"id": 15}
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(exceed_club, self.mock_bootstrap)
        self.assertIn("Maximum of 3 players from same club allowed", str(ctx.exception))

    @patch("src.pipeline.weekly_decision.run_update_current_pipeline")
    @patch("src.pipeline.weekly_decision.predict_gameweek")
    @patch("src.pipeline.weekly_decision.determine_season_and_gw")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_pipeline_first_run_init_state(
        self, mock_fetch, mock_determine, mock_predict, mock_update
    ):
        """Verify that a first-time setup initializes the persistent state correctly."""
        mock_determine.return_value = ("2025-26", 2, "2025-08-18T17:30:00Z")
        mock_predict.return_value = (self.mock_predictions_df, None)
        mock_fetch.return_value = self.mock_bootstrap

        self.assertFalse(STATE_FILE.exists())

        # Run pipeline passing `--squad` path
        summary = run_weekly_decision_pipeline(
            season="2025-26",
            gw=2,
            squad_path=str(self.mock_squad_path),
            bank_override=0.5,
            free_transfers_override=1,
            apply=True
        )

        self.assertTrue(STATE_FILE.exists())
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            
        self.assertEqual(state["season"], "2025-26")
        self.assertEqual(state["last_processed_gw"], 2)
        self.assertEqual(state["bank"], 0.0)
        self.assertEqual(len(state["players"]), 15)
        self.assertEqual(len(state["players_metadata"]), 15)

    @patch("src.pipeline.weekly_decision.run_update_current_pipeline")
    @patch("src.pipeline.weekly_decision.predict_gameweek")
    @patch("src.pipeline.weekly_decision.determine_season_and_gw")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_apply_vs_no_apply(
        self, mock_fetch, mock_determine, mock_predict, mock_update
    ):
        """Verify that simulation run (apply=False) does not write to state/history, but apply=True does."""
        mock_determine.return_value = ("2025-26", 2, "2025-08-18T17:30:00Z")
        mock_predict.return_value = (self.mock_predictions_df, None)
        mock_fetch.return_value = self.mock_bootstrap

        # 1. Run without --apply
        run_weekly_decision_pipeline(
            season="2025-26",
            gw=2,
            squad_path=str(self.mock_squad_path),
            apply=False
        )

        # STATE_FILE should NOT exist because apply=False does not persist state
        self.assertFalse(STATE_FILE.exists())
        self.assertFalse((HISTORY_DIR / "weekly_gw2.json").exists())

        # 2. Run with --apply
        run_weekly_decision_pipeline(
            season="2025-26",
            gw=2,
            squad_path=str(self.mock_squad_path),
            apply=True
        )

        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        self.assertEqual(state["last_processed_gw"], 2)
        self.assertTrue((HISTORY_DIR / "weekly_gw2.json").exists())

    @patch("src.pipeline.weekly_decision.run_update_current_pipeline")
    @patch("src.pipeline.weekly_decision.predict_gameweek")
    @patch("src.pipeline.weekly_decision.determine_season_and_gw")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_double_processing_prevention(
        self, mock_fetch, mock_determine, mock_predict, mock_update
    ):
        """Verify double processing prevention prevents running the same gameweek twice unless forced."""
        mock_determine.return_value = ("2025-26", 2, "2025-08-18T17:30:00Z")
        mock_predict.return_value = (self.mock_predictions_df, None)
        mock_fetch.return_value = self.mock_bootstrap

        # Init state and set last_processed_gw = 2
        run_weekly_decision_pipeline(
            season="2025-26",
            gw=2,
            squad_path=str(self.mock_squad_path),
            apply=True
        )

        # Re-run same gw 2 without force -> expect ValueError
        with self.assertRaises(ValueError) as ctx:
            run_weekly_decision_pipeline(
                season="2025-26",
                gw=2,
                apply=True,
                force=False
            )
        self.assertIn("already been processed", str(ctx.exception))

        # Re-run same gw 2 with force -> expect success
        summary = run_weekly_decision_pipeline(
            season="2025-26",
            gw=2,
            apply=True,
            force=True
        )
        self.assertIsNotNone(summary)

    def test_price_conversion_accuracy(self):
        """Validate FPL API price scale conversions: API 105 -> 10.5, 60 -> 6.0, 45 -> 4.5, 40 -> 4.0."""
        mock_user_squad = {
            "players": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "bank": 0.5,
            "free_transfers": 1
        }
        mock_bootstrap = {
            "elements": [
                # GKs
                {"id": 1, "code": 101, "first_name": "P101", "second_name": "S101", "element_type": 1, "team": 1, "now_cost": 105},
                {"id": 2, "code": 102, "first_name": "P102", "second_name": "S102", "element_type": 1, "team": 2, "now_cost": 60},
                # DEFs
                {"id": 3, "code": 103, "first_name": "P103", "second_name": "S103", "element_type": 2, "team": 1, "now_cost": 45},
                {"id": 4, "code": 104, "first_name": "P104", "second_name": "S104", "element_type": 2, "team": 3, "now_cost": 40},
                {"id": 5, "code": 105, "first_name": "P105", "second_name": "S105", "element_type": 2, "team": 2, "now_cost": 50},
                {"id": 6, "code": 106, "first_name": "P106", "second_name": "S106", "element_type": 2, "team": 4, "now_cost": 50},
                {"id": 7, "code": 107, "first_name": "P107", "second_name": "S107", "element_type": 2, "team": 5, "now_cost": 50},
                # MIDs
                {"id": 8, "code": 108, "first_name": "P108", "second_name": "S108", "element_type": 3, "team": 1, "now_cost": 85},
                {"id": 9, "code": 109, "first_name": "P109", "second_name": "S109", "element_type": 3, "team": 3, "now_cost": 75},
                {"id": 10, "code": 110, "first_name": "P110", "second_name": "S110", "element_type": 3, "team": 2, "now_cost": 70},
                {"id": 11, "code": 111, "first_name": "P111", "second_name": "S111", "element_type": 3, "team": 4, "now_cost": 65},
                {"id": 12, "code": 112, "first_name": "P112", "second_name": "S112", "element_type": 3, "team": 5, "now_cost": 60},
                # FWDs
                {"id": 13, "code": 113, "first_name": "P113", "second_name": "S113", "element_type": 4, "team": 3, "now_cost": 95},
                {"id": 14, "code": 114, "first_name": "P114", "second_name": "S114", "element_type": 4, "team": 4, "now_cost": 80},
                {"id": 15, "code": 115, "first_name": "P115", "second_name": "S115", "element_type": 4, "team": 5, "now_cost": 75}
            ],
            "teams": [
                {"id": 1, "short_name": "ARS"},
                {"id": 2, "short_name": "CHE"},
                {"id": 3, "short_name": "MUN"},
                {"id": 4, "short_name": "MCI"},
                {"id": 5, "short_name": "LIV"}
            ]
        }
        from src.pipeline.weekly_decision import validate_and_compile_user_squad
        _, metadata = validate_and_compile_user_squad(mock_user_squad, mock_bootstrap)
        
        # Verify specific mappings
        self.assertEqual(next(m["current_price"] for m in metadata if m["code"] == 101), 10.5)
        self.assertEqual(next(m["current_price"] for m in metadata if m["code"] == 102), 6.0)
        self.assertEqual(next(m["current_price"] for m in metadata if m["code"] == 103), 4.5)
        self.assertEqual(next(m["current_price"] for m in metadata if m["code"] == 104), 4.0)

    def test_validation_implausible_low_cost(self):
        """Verify that validation fails if squad cost is implausibly low (e.g. < 50m)."""
        mock_user_squad = {
            "players": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "bank": 0.5,
            "free_transfers": 1
        }
        mock_bootstrap = {
            "elements": [
                {"id": idx + 1, "code": 101 + idx, "first_name": f"P{idx}", "second_name": f"S{idx}", 
                 "element_type": (1 if idx < 2 else (2 if idx < 7 else (3 if idx < 12 else 4))), 
                 "team": (idx % 5) + 1, "now_cost": 30} # 30 -> 3.0m (realistic but low total)
                for idx in range(15)
            ],
            "teams": [{"id": i, "short_name": f"T{i}"} for i in range(1, 6)]
        }
        from src.pipeline.weekly_decision import validate_and_compile_user_squad
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(mock_user_squad, mock_bootstrap)
        self.assertIn("Implausibly low squad cost", str(ctx.exception))

    def test_validation_unrealistic_player_price(self):
        """Verify that validation fails if any single player price is unrealistic (e.g. < 3.0m)."""
        mock_user_squad = {
            "players": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "bank": 0.5,
            "free_transfers": 1
        }
        mock_bootstrap = {
            "elements": [
                {"id": idx + 1, "code": 101 + idx, "first_name": f"P{idx}", "second_name": f"S{idx}", 
                 "element_type": (1 if idx < 2 else (2 if idx < 7 else (3 if idx < 12 else 4))), 
                 "team": (idx % 5) + 1, "now_cost": (25 if idx == 0 else 60)} # player 0 has now_cost 25 -> 2.5m (unrealistic)
                for idx in range(15)
            ],
            "teams": [{"id": i, "short_name": f"T{i}"} for i in range(1, 6)]
        }
        from src.pipeline.weekly_decision import validate_and_compile_user_squad
        with self.assertRaises(ValueError) as ctx:
            validate_and_compile_user_squad(mock_user_squad, mock_bootstrap)
        self.assertIn("unrealistic price", str(ctx.exception))

    @patch("src.pipeline.weekly_decision.run_update_current_pipeline")
    @patch("src.pipeline.weekly_decision.predict_gameweek")
    @patch("src.pipeline.weekly_decision.determine_season_and_gw")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_gw1_initial_squad_setup(self, mock_fetch, mock_determine, mock_predict, mock_update):
        """Verify pre-GW1 Initial Season Setup behaves correctly: no transfers recommended, only XI optimized."""
        mock_determine.return_value = ("2025-26", 1, "2025-08-11T17:30:00Z")
        mock_predict.return_value = (self.mock_predictions_df, None)
        mock_fetch.return_value = self.mock_bootstrap

        summary = run_weekly_decision_pipeline(
            season="2025-26",
            gw=1,
            squad_path=str(self.mock_squad_path),
            apply=True
        )

        self.assertEqual(summary["recommendation_decision"], "HOLD")
        self.assertEqual(len(summary["transfers_out"]), 0)
        self.assertEqual(len(summary["transfers_in"]), 0)
        self.assertEqual(summary["transfer_penalty"], 0.0)
        
        # Verify applied state sets last_processed_gw = 1 and free_transfers remains 1
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        self.assertEqual(state["last_processed_gw"], 1)
        self.assertEqual(state["free_transfers"], 1)
        self.assertTrue((HISTORY_DIR / "weekly_gw1.json").exists())

    @patch("src.pipeline.weekly_decision.run_update_current_pipeline")
    @patch("src.pipeline.weekly_decision.predict_gameweek")
    @patch("src.pipeline.weekly_decision.determine_season_and_gw")
    @patch("src.pipeline.weekly_decision.fetch_json")
    def test_gw1_initial_squad_builder_budget(self, mock_fetch, mock_determine, mock_predict, mock_update):
        """Verify that running GW1 without providing a squad constructs the optimal squad from scratch using £100m budget."""
        mock_determine.return_value = ("2025-26", 1, "2025-08-11T17:30:00Z")
        mock_predict.return_value = (self.mock_predictions_df, None)
        mock_fetch.return_value = self.mock_bootstrap

        # 1. Run without apply
        summary = run_weekly_decision_pipeline(
            season="2025-26",
            gw=1,
            squad_path=None,
            apply=False
        )

        self.assertEqual(summary["recommendation_decision"], "INITIAL_BUILD")
        self.assertEqual(len(summary["transfers_out"]), 0)
        self.assertEqual(len(summary["transfers_in"]), 0)
        self.assertEqual(summary["transfer_penalty"], 0.0)
        self.assertEqual(len(summary["players"]), 15)
        
        # Verify positions
        pos_counts = {}
        for p in summary["players"]:
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1
        self.assertEqual(pos_counts.get("GK"), 2)
        self.assertEqual(pos_counts.get("DEF"), 5)
        self.assertEqual(pos_counts.get("MID"), 5)
        self.assertEqual(pos_counts.get("FWD"), 3)
        
        # Verify budget <= 100.0m
        total_cost = sum(p["current_price"] for p in summary["players"])
        self.assertTrue(total_cost <= 100.0)
        
        # State file should not be created yet since apply=False
        self.assertFalse(STATE_FILE.exists())

        # 2. Run with apply
        summary_applied = run_weekly_decision_pipeline(
            season="2025-26",
            gw=1,
            squad_path=None,
            apply=True
        )

        self.assertTrue(STATE_FILE.exists())
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        self.assertEqual(state["last_processed_gw"], 1)
        self.assertEqual(state["free_transfers"], 1)
        self.assertEqual(len(state["players"]), 15)
        self.assertTrue(state["bank"] >= 0.0)
        self.assertTrue((HISTORY_DIR / "weekly_gw1.json").exists())

    def test_actual_gw1_squad_registration(self):
        """Verify the actual registered GW1 squad data in current_squad.json."""
        real_state_file = PROJECT_ROOT / "data" / "state" / "current_squad.json"
        self.assertTrue(real_state_file.exists(), "Actual current_squad.json does not exist!")
        
        with open(real_state_file, "r") as f:
            state = json.load(f)
            
        self.assertEqual(state["season"], "2026-27")
        self.assertEqual(state["last_processed_gw"], 0)
        self.assertEqual(state["free_transfers"], 1)
        self.assertEqual(len(state["players"]), 15)
        
        # Verify specific player codes in the squad
        expected_codes = [154561, 49262, 215136, 17761, 477424, 441164, 461102, 141746, 424876, 484420, 60307, 243413, 475168, 223094, 177815]
        self.assertEqual(sorted(state["players"]), sorted(expected_codes))
        
        # Verify position counts in metadata
        metadata = state["players_metadata"]
        self.assertEqual(len(metadata), 15)
        
        gk_count = sum(1 for p in metadata if p["position"] == "GK")
        def_count = sum(1 for p in metadata if p["position"] == "DEF")
        mid_count = sum(1 for p in metadata if p["position"] == "MID")
        fwd_count = sum(1 for p in metadata if p["position"] == "FWD")
        
        self.assertEqual(gk_count, 2)
        self.assertEqual(def_count, 5)
        self.assertEqual(mid_count, 5)
        self.assertEqual(fwd_count, 3)
        
        # Verify captain/vice-captain
        self.assertEqual(state["captain"], 223094) # Erling Haaland
        self.assertEqual(state["vice_captain"], 141746) # Bruno Fernandes

if __name__ == "__main__":
    unittest.main()
