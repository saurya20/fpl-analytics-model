import unittest
import pandas as pd
import numpy as np
from src.optimization.squad_optimizer import optimize_squad, enforce_leakage_guard

class TestSquadOptimizer(unittest.TestCase):
    
    def setUp(self):
        """
        Create a valid synthetic player prediction dataset (30 players).
        This includes sufficient players in each position with sensible prices and teams
        so that a valid optimal squad under £100.0m can always be found.
        """
        # Teams: T1, T2, T3, T4, T5, T6, T7, T8, T9, T10
        # Positions: GK, DEF, MID, FWD
        players_data = []
        
        # 4 Goalkeepers
        for i in range(4):
            players_data.append({
                "code": 100 + i,
                "name": f"GK Player {i}",
                "position": "GK",
                "team": f"T{i+1}",
                "current_price": 4.5 + (i * 0.2), # £4.5m to £5.1m
                "predicted_points": 3.0 + i,      # 3.0 to 6.0 expected pts
                "predicted_minutes": 90.0,
                "prob_60_plus": 0.95
            })
            
        # 10 Defenders
        for i in range(10):
            players_data.append({
                "code": 200 + i,
                "name": f"DEF Player {i}",
                "position": "DEF",
                "team": f"T{(i % 5) + 1}",        # Teams T1 to T5 (2 per team max)
                "current_price": 4.0 + (i * 0.3), # £4.0m to £6.7m
                "predicted_points": 2.0 + (i * 0.5), # 2.0 to 6.5 expected pts
                "predicted_minutes": 90.0,
                "prob_60_plus": 0.90
            })
            
        # 10 Midfielders
        for i in range(10):
            players_data.append({
                "code": 300 + i,
                "name": f"MID Player {i}",
                "position": "MID",
                "team": f"T{(i % 5) + 6}",        # Teams T6 to T10 (2 per team max)
                "current_price": 5.0 + (i * 0.6), # £5.0m to £10.4m
                "predicted_points": 3.0 + (i * 0.6), # 3.0 to 8.4 expected pts
                "predicted_minutes": 90.0,
                "prob_60_plus": 0.92
            })
            
        # 6 Forwards
        for i in range(6):
            players_data.append({
                "code": 400 + i,
                "name": f"FWD Player {i}",
                "position": "FWD",
                "team": f"T{(i % 3) + 1}",        # Teams T1 to T3 (2 per team max)
                "current_price": 5.5 + (i * 1.2), # £5.5m to £11.5m
                "predicted_points": 4.0 + (i * 0.8), # 4.0 to 8.0 expected pts
                "predicted_minutes": 90.0,
                "prob_60_plus": 0.88
            })
            
        self.df = pd.DataFrame(players_data)
        
    def test_optimal_squad_constraints(self):
        """Test constraints 1-19: Validate all FPL rules, starting XI, captains, and bench order."""
        results = optimize_squad(self.df, budget=100.0)
        squad = results["squad"]
        
        # 1. Exactly 15 players selected
        self.assertEqual(len(squad), 15)
        
        # 2. Position constraints
        self.assertEqual(squad[squad["position"] == "GK"].shape[0], 2)
        self.assertEqual(squad[squad["position"] == "DEF"].shape[0], 5)
        self.assertEqual(squad[squad["position"] == "MID"].shape[0], 5)
        self.assertEqual(squad[squad["position"] == "FWD"].shape[0], 3)
        
        # 3. Total cost <= 100m
        self.assertTrue(results["cost"] <= 100.0)
        
        # 4. Maximum 3 players per club
        for club, count in results["club_counts"].items():
            self.assertTrue(count <= 3, f"Club {club} has {count} players selected (max is 3)")
            
        # 5. Starting XI size and structure
        starters = squad[squad["starter"] == 1]
        self.assertEqual(len(starters), 11)
        self.assertEqual(starters[starters["position"] == "GK"].shape[0], 1)
        self.assertTrue(starters[starters["position"] == "DEF"].shape[0] >= 3)
        self.assertTrue(starters[starters["position"] == "MID"].shape[0] >= 1)
        self.assertTrue(starters[starters["position"] == "FWD"].shape[0] >= 1)
        
        # 6. Captain and Vice-captain constraints
        captain = squad[squad["captain"] == 1]
        vice = squad[squad["vice_captain"] == 1]
        self.assertEqual(len(captain), 1)
        self.assertEqual(len(vice), 1)
        self.assertEqual(captain["starter"].values[0], 1, "Captain must be a starter")
        self.assertEqual(vice["starter"].values[0], 1, "Vice-captain must be a starter")
        self.assertNotEqual(captain["code"].values[0], vice["code"].values[0], "Captain and Vice-captain must be different players")
        
        # 7. Bench contains 4 players (1 GK and 3 outfield)
        bench = squad[squad["starter"] == 0]
        self.assertEqual(len(bench), 4)
        self.assertEqual(bench[bench["position"] == "GK"].shape[0], 1)
        self.assertEqual(bench[bench["position"] == "GK"]["bench_order"].values[0], "GK")
        
        # 8. Outfield bench players correctly ordered descending by predicted points
        outfield_sub = bench[bench["position"] != "GK"].sort_values("bench_order")
        self.assertEqual(list(outfield_sub["bench_order"].values), [1, 2, 3])
        points_list = list(outfield_sub["predicted_points"].values)
        self.assertEqual(points_list, sorted(points_list, reverse=True), "Bench outfield players not sorted descending by predicted points")
        
    def test_double_gameweek_handling(self):
        """Test 20: DGW players with high points are naturally selected by expected value."""
        df_dgw = self.setUp_modified_dgw()
        results = optimize_squad(df_dgw, budget=100.0)
        squad = results["squad"]
        
        # Verify DGW player (code 309, pts 16.0) is selected and captained
        self.assertTrue(309 in squad["code"].values, "High-value DGW player was not selected")
        cap_player = squad[squad["captain"] == 1]
        self.assertEqual(cap_player["code"].values[0], 309, "Highest predicted player must be captain")
        
    def setUp_modified_dgw(self):
        df_dgw = self.df.copy()
        # Set player 309 (Midfielder) to represent a massive DGW aggregate points (e.g. 16.0 expected pts)
        df_dgw.loc[df_dgw["code"] == 309, "predicted_points"] = 16.0
        return df_dgw
        
    def test_blank_gameweek_handling(self):
        """Test 21: BGW players remain valid candidates but are avoided unless selected as budget warmers."""
        df_bgw = self.df.copy()
        # Set all premium players expected points to 0.0 (BGW players)
        # Verify that BGW players with high prices are avoided, but cheap BGW players can still be picked if budget forces it
        df_bgw.loc[df_bgw["code"] == 309, "predicted_points"] = 0.0 # BGW
        df_bgw.loc[df_bgw["code"] == 309, "current_price"] = 12.0  # Expensive
        
        results = optimize_squad(df_bgw, budget=100.0)
        squad = results["squad"]
        self.assertFalse(309 in squad["code"].values, "Expensive BGW player with 0 pts should be avoided")
        
        # Set a cheap BGW player to 0 pts and check if they can be selected to fit budget constraints
        df_bgw.loc[df_bgw["code"] == 100, "predicted_points"] = 0.0 # GK BGW
        df_bgw.loc[df_bgw["code"] == 100, "current_price"] = 4.0  # Ultra cheap
        
        results = optimize_squad(df_bgw, budget=84.0) # Set very tight but feasible budget (min possible is ~82.8m)
        squad = results["squad"]
        self.assertTrue(100 in squad["code"].values, "Cheap BGW player should remain a candidate and selected to satisfy tight budget constraint")
        
    def test_leakage_guard(self):
        """Test 22: Verify that leakage guard triggers ValueError on prohibited target columns."""
        df_leak = self.df.copy()
        df_leak["target_points"] = 5.0
        
        with self.assertRaises(ValueError) as ctx:
            optimize_squad(df_leak)
        self.assertIn("DATA LEAKAGE SHIELD TRIGGERED", str(ctx.exception))
        
        df_leak2 = self.df.copy()
        df_leak2["total_points"] = 10.0
        with self.assertRaises(ValueError) as ctx:
            optimize_squad(df_leak2)
        self.assertIn("DATA LEAKAGE SHIELD TRIGGERED", str(ctx.exception))
        
    def test_determinism(self):
        """Test 23: Verify that optimization is deterministic on identical inputs."""
        res1 = optimize_squad(self.df, budget=100.0)
        res2 = optimize_squad(self.df, budget=100.0)
        
        squad1 = res1["squad"].sort_values("code")
        squad2 = res2["squad"].sort_values("code")
        
        pd.testing.assert_frame_equal(squad1, squad2)
        
    def test_no_mutation(self):
        """Test 24: Check that the input predictions DataFrame is not mutated."""
        df_copy = self.df.copy()
        _ = optimize_squad(self.df)
        pd.testing.assert_frame_equal(self.df, df_copy)
        
    def test_infeasible_inputs(self):
        """Test 25: Check that infeasible configurations raise a clean ValueError."""
        df_inf = self.df.copy()
        # Set all player prices to £30.0m (making 15 players cost £450m, impossible under £100m budget)
        df_inf["current_price"] = 30.0
        
        with self.assertRaises(ValueError) as ctx:
            optimize_squad(df_inf, budget=100.0)
        self.assertIn("infeasible", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()
