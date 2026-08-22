import unittest
import pandas as pd
import numpy as np
import json
from src.optimization.transfer_optimizer import optimize_transfers

class TestTransferOptimizer(unittest.TestCase):
    
    def setUp(self):
        """
        Create a valid synthetic player pool (30 players) and current owned squad (15 players).
        """
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
        
        # Select 15 players as current owned squad (GK=2, DEF=5, MID=5, FWD=3)
        # Choosing middle players so we have room to transfer in better ones or transfer out worse ones
        self.current_squad = [
            100, 101,  # GKs
            200, 201, 202, 203, 204,  # DEFs
            300, 301, 302, 303, 304,  # MIDs
            400, 401, 402   # FWDs
        ]
        
    def test_transfer_optimizer_constraints(self):
        """Test constraints 1-8: Verify squad limits, position allocations, budget limits, club counts, valid starting XI, and transfer rules."""
        results = optimize_transfers(
            df=self.df,
            current_squad=self.current_squad,
            bank=1.0,
            free_transfers=1
        )
        squad = results["squad"]
        
        # 1. Exactly 15 players selected
        self.assertEqual(len(squad), 15)
        
        # 2. Position constraints
        self.assertEqual(squad[squad["position"] == "GK"].shape[0], 2)
        self.assertEqual(squad[squad["position"] == "DEF"].shape[0], 5)
        self.assertEqual(squad[squad["position"] == "MID"].shape[0], 5)
        self.assertEqual(squad[squad["position"] == "FWD"].shape[0], 3)
        
        # 3. Budget constraint (cost <= max_budget)
        self.assertTrue(results["cost"] <= results["max_budget"])
        
        # 4. Maximum 3 players per club
        for club, count in results["club_counts"].items():
            self.assertTrue(count <= 3, f"Club {club} has {count} players (max 3)")
            
        # 5. Starting XI size and structure
        starters = squad[squad["starter"] == 1]
        self.assertEqual(len(starters), 11)
        self.assertEqual(starters[starters["position"] == "GK"].shape[0], 1)
        self.assertTrue(starters[starters["position"] == "DEF"].shape[0] >= 3)
        self.assertTrue(starters[starters["position"] == "MID"].shape[0] >= 1)
        self.assertTrue(starters[starters["position"] == "FWD"].shape[0] >= 1)
        
        # 6. Transfer consistency checks
        # Transfers out must be from the current squad
        for player in results["transfers_out"]:
            self.assertTrue(player["code"] in self.current_squad)
            
        # Transfers in must not be in the current squad
        for player in results["transfers_in"]:
            self.assertFalse(player["code"] in self.current_squad)
            
        # Re-check membership
        final_codes = list(squad["code"].values)
        expected_final = set(self.current_squad) - {p["code"] for p in results["transfers_out"]} | {p["code"] for p in results["transfers_in"]}
        self.assertEqual(set(final_codes), expected_final)
        
    def test_free_transfer_and_hits_penalty(self):
        """Test 9-10: Verify available free transfers has no hit, while extra transfers incur configurable hit penalty."""
        # Case A: 1 free transfer, make 1 transfer
        # Force a highly attractive transfer by boosting points of an unowned defender
        df_mod = self.df.copy()
        df_mod.loc[df_mod["code"] == 209, "predicted_points"] = 15.0 # Unowned DEF
        df_mod.loc[df_mod["code"] == 200, "predicted_points"] = 0.5  # Owned DEF
        
        results = optimize_transfers(df_mod, self.current_squad, bank=2.0, free_transfers=1, hit_cost=4.0)
        self.assertEqual(results["num_transfers"], 1)
        self.assertEqual(results["transfer_penalty"], 0.0, "1 transfer with 1 free transfer should not incur hits")
        
        # Case B: 1 free transfer, force 2 transfers (boost another player)
        df_mod.loc[df_mod["code"] == 309, "predicted_points"] = 15.0 # Unowned MID
        df_mod.loc[df_mod["code"] == 300, "predicted_points"] = 0.5  # Owned MID
        
        results = optimize_transfers(df_mod, self.current_squad, bank=5.0, free_transfers=1, hit_cost=4.0)
        self.assertEqual(results["num_transfers"], 2)
        self.assertEqual(results["transfer_penalty"], 4.0, "2 transfers with 1 free transfer should incur a -4 hit")
        
    def test_hold_selected_when_transfer_worse_than_hit(self):
        """Test 11-13: Solver selects HOLD when expected gains do not exceed hit penalty."""
        df_mod = self.df.copy()
        # Unowned player is slightly better (+2.0 pts), but we have 0 free transfers (so transfer costs 4.0 hit)
        df_mod.loc[df_mod["code"] == 209, "predicted_points"] = 5.0 # Unowned (+2.0 pts compared to owned 200)
        df_mod.loc[df_mod["code"] == 200, "predicted_points"] = 3.0 # Owned
        
        results = optimize_transfers(df_mod, self.current_squad, bank=2.0, free_transfers=0, hit_cost=4.0)
        self.assertEqual(results["recommendation"], "HOLD")
        self.assertEqual(results["num_transfers"], 0)
        
    def test_transfer_selected_when_better_than_hit(self):
        """Test 11-13: Solver selects TRANSFER when expected gains exceed hit penalty."""
        df_mod = self.df.copy()
        # Unowned player is significantly better (+6.0 pts), and we have 0 free transfers (costs 4.0 hit)
        df_mod.loc[df_mod["code"] == 209, "predicted_points"] = 9.0 # Unowned (+6.0 pts compared to owned 200)
        df_mod.loc[df_mod["code"] == 200, "predicted_points"] = 3.0 # Owned
        
        results = optimize_transfers(df_mod, self.current_squad, bank=2.0, free_transfers=0, hit_cost=4.0)
        self.assertEqual(results["recommendation"], "TRANSFER")
        self.assertEqual(results["num_transfers"], 1)
        self.assertEqual(results["transfer_penalty"], 4.0)
        
    def test_leakage_guard(self):
        """Test 19-20: Guard triggers ValueError on prohibited target columns."""
        df_leak = self.df.copy()
        df_leak["target_points"] = 5.0
        with self.assertRaises(ValueError) as ctx:
            optimize_transfers(df_leak, self.current_squad)
        self.assertIn("DATA LEAKAGE SHIELD TRIGGERED", str(ctx.exception))
        
    def test_infeasible_owned_squad(self):
        """Test 23: Verify that an invalid/infeasible current squad raises ValueError."""
        # Create an invalid squad of 15 (e.g. 6 defenders instead of 5)
        invalid_squad = [
            100, 101,  # GKs
            200, 201, 202, 203, 204, 205, # 6 DEFs!
            300, 301, 302, 303,  # 4 MIDs
            400, 401, 402   # 3 FWDs
        ]
        with self.assertRaises(ValueError) as ctx:
            optimize_transfers(self.df, invalid_squad)
        self.assertIn("infeasible", str(ctx.exception).lower())

    def test_plausible_squad_cost(self):
        """Verify that optimized squad cost is plausible (>= 50.0m) and does not use scaled-down values."""
        results = optimize_transfers(
            df=self.df,
            current_squad=self.current_squad,
            bank=1.0,
            free_transfers=1
        )
        squad_cost = results["cost"]
        # The 15 squad players have prices ranging from 4.0 to 9.5, total cost must be at least 50.0m
        self.assertTrue(squad_cost >= 50.0, f"Squad cost £{squad_cost}m is implausibly low! Must be in millions (>= 50.0).")

if __name__ == "__main__":
    unittest.main()
