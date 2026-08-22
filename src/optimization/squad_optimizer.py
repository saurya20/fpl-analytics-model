import pandas as pd
import numpy as np
import pulp
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.predict import predict_gameweek, load_config
from src.optimization.constraints import (
    add_squad_size_constraint,
    add_position_constraints,
    add_budget_constraint,
    add_club_constraints,
    add_starting_xi_constraints,
    add_starter_position_constraints,
    add_captaincy_constraints
)
from src.optimization.objective import get_optimization_objective

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

def enforce_leakage_guard(df: pd.DataFrame):
    """
    Prevent targets or raw performance variables from entering the optimizer.
    """
    prohibited = [
        "target_points", "target_minutes", "target_60_plus_minutes",
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bps", "bonus", "starts"
    ]
    detected = [c for c in prohibited if c in df.columns]
    if detected:
        raise ValueError(f"DATA LEAKAGE SHIELD TRIGGERED: Prohibited columns detected in optimizer input: {detected}")


def optimize_squad(df: pd.DataFrame, budget: float = 100.0) -> Dict[str, Any]:
    """
    Integer Linear Programming (ILP) optimizer using PuLP/CBC.
    Returns optimal 15-player squad, starting XI, captain/vice-captain,
    bench ordering, expected points, costs, and formation.
    """
    # 1. Validation Layer
    enforce_leakage_guard(df)
    
    # Resolve column names
    points_col = "predicted_points" if "predicted_points" in df.columns else "expected_points"
    minutes_col = "predicted_minutes" if "predicted_minutes" in df.columns else "expected_minutes"
    prob_col = "prob_60_plus" if "prob_60_plus" in df.columns else "playing_probability"
    
    required = ["code", "name", "position", "team", "current_price"] + [points_col, minutes_col, prob_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in optimizer input: {missing}")
        
    df = df.copy()
    
    # Initialize optimization problem
    prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
    
    # Define binary decision variables keyed by DataFrame index
    x = pulp.LpVariable.dicts("squad", df.index, cat=pulp.LpBinary)
    s = pulp.LpVariable.dicts("starter", df.index, cat=pulp.LpBinary)
    c = pulp.LpVariable.dicts("captain", df.index, cat=pulp.LpBinary)
    v = pulp.LpVariable.dicts("vice", df.index, cat=pulp.LpBinary)
    
    # 2. Add Constraints
    add_squad_size_constraint(prob, x, df)
    add_position_constraints(prob, x, df)
    add_budget_constraint(prob, x, df, budget=budget)
    add_club_constraints(prob, x, df)
    
    add_starting_xi_constraints(prob, x, s, df)
    add_starter_position_constraints(prob, s, df)
    
    add_captaincy_constraints(prob, s, c, v, df)
    
    # 3. Add Objective
    prob += get_optimization_objective(s, c, v, df, target_points_col=points_col)
    
    # 4. Solve Problem
    # Use quiet COIN-OR solver
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]
    
    if status_str != "Optimal":
        raise ValueError(f"FPL Optimization Problem is infeasible or could not be solved! Solver Status: {status_str}")
        
    # 5. Extract selected flags
    df["selected"] = [int(x[i].varValue > 0.5) for i in df.index]
    df["starter"] = [int(s[i].varValue > 0.5) for i in df.index]
    df["captain"] = [int(c[i].varValue > 0.5) for i in df.index]
    df["vice_captain"] = [int(v[i].varValue > 0.5) for i in df.index]
    
    # Filter squad players
    squad_df = df[df["selected"] == 1].copy()
    
    # Calculate costs and expected points
    total_cost = squad_df["current_price"].sum()
    expected_starting_pts = squad_df[squad_df["starter"] == 1][points_col].sum()
    expected_captain_bonus = squad_df[squad_df["captain"] == 1][points_col].sum()
    total_expected_points = expected_starting_pts + expected_captain_bonus
    
    # Starting XI position counts to determine formation
    starters = squad_df[squad_df["starter"] == 1]
    num_def = starters[starters["position"] == "DEF"].shape[0]
    num_mid = starters[starters["position"] == "MID"].shape[0]
    num_fwd = starters[starters["position"] == "FWD"].shape[0]
    formation = f"{num_def}-{num_mid}-{num_fwd}"
    
    # 6. Generate Bench Ordering
    bench_df = squad_df[squad_df["starter"] == 0].copy()
    
    # Backup GK goes to GK bench
    bench_gk = bench_df[bench_df["position"] == "GK"]
    bench_outfield = bench_df[bench_df["position"] != "GK"].sort_values(points_col, ascending=False)
    
    # Assign bench order index (outfield substitutes 1, 2, 3)
    squad_df["bench_order"] = None
    
    if not bench_gk.empty:
        squad_df.loc[bench_gk.index, "bench_order"] = "GK"
        
    for index, (idx, row) in enumerate(bench_outfield.iterrows(), 1):
        squad_df.loc[idx, "bench_order"] = index
        
    # Club distribution
    club_counts = squad_df["team"].value_counts().to_dict()
    
    return {
        "squad": squad_df,
        "cost": float(total_cost),
        "expected_points": float(total_expected_points),
        "expected_starting_pts": float(expected_starting_pts),
        "expected_captain_bonus": float(expected_captain_bonus),
        "formation": formation,
        "club_counts": club_counts,
        "status": status_str
    }


def main():
    parser = argparse.ArgumentParser(description="FPL ILP Squad Optimizer")
    parser.add_argument("--season", type=str, required=True, help="Target season (e.g. 2024-25)")
    parser.add_argument("--gw", type=int, required=True, help="Target gameweek (1-38)")
    parser.add_argument("--budget", type=float, default=100.0, help="Total squad budget (default 100.0)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"FPL SQUAD OPTIMIZATION: {args.season} GW {args.gw}")
    print("=" * 60)
    
    # Find prediction file
    predictions_csv = RESULTS_DIR / f"predictions_{args.season}_gw{args.gw}.csv"
    
    if not predictions_csv.exists():
        print(f"Predictions file not found at {predictions_csv}. Running Phase 3A to generate predictions...")
        try:
            # Re-generate predictions on-the-fly
            predict_gameweek(season=args.season, gw=args.gw, horizon=1)
        except Exception as e:
            print(f"Failed to auto-generate predictions: {e}")
            sys.exit(1)
            
    # Load predictions
    df = pd.read_csv(predictions_csv)
    
    try:
        results = optimize_squad(df, budget=args.budget)
        squad = results["squad"]
        
        print(f"\n==================================================")
        print(f"OPTIMAL FPL SQUAD — {args.season} GW {args.gw}")
        print(f"==================================================")
        print(f"Formation: {results['formation']}")
        print(f"Total Cost: £{results['cost']:.1f}m")
        print(f"Expected Points: {results['expected_points']:.2f}")
        print(f"--------------------------------------------------")
        
        print("STARTING XI\n")
        starters = squad[squad["starter"] == 1].sort_values("position", key=lambda x: x.map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}))
        for idx, row in starters.iterrows():
            pos_label = row["position"]
            cap_label = ""
            if row["captain"] == 1:
                cap_label = " (C)"
            elif row["vice_captain"] == 1:
                cap_label = " (VC)"
            print(f"{pos_label:<4} {row['name']:<30} £{row['current_price']:<4.1f} Exp Pts: {row['predicted_points']:.2f}{cap_label}")
            
        print(f"\nBENCH\n")
        bench = squad[squad["starter"] == 0].sort_values("bench_order", key=lambda x: x.map({"GK": 0, 1: 1, 2: 2, 3: 3}))
        for idx, row in bench.iterrows():
            order = f"SUB {row['bench_order']}" if row["bench_order"] != "GK" else "GK Sub"
            print(f"{order:<8} {row['name']:<30} £{row['current_price']:<4.1f} Exp Pts: {row['predicted_points']:.2f}")
            
        print(f"==================================================")
        print(f"SQUAD SUMMARY")
        print(f"==================================================")
        print(f"Goalkeepers: {squad[squad['position']=='GK'].shape[0]}")
        print(f"Defenders:   {squad[squad['position']=='DEF'].shape[0]}")
        print(f"Midfielders: {squad[squad['position']=='MID'].shape[0]}")
        print(f"Forwards:    {squad[squad['position']=='FWD'].shape[0]}")
        print(f"\nBudget Remaining: £{args.budget - results['cost']:.1f}m")
        print(f"\nClub Distribution:")
        for team, count in results["club_counts"].items():
            print(f"  {team}: {count}")
        print(f"==================================================\n")
        
        # Save output to CSV
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        squad_csv = RESULTS_DIR / f"squad_{args.season}_gw{args.gw}.csv"
        squad_json = RESULTS_DIR / f"squad_{args.season}_gw{args.gw}.json"
        
        squad.to_csv(squad_csv, index=False)
        
        # Write JSON Summary
        summary = {
            "season": args.season,
            "gw": args.gw,
            "formation": results["formation"],
            "total_cost": results["cost"],
            "expected_points": results["expected_points"],
            "expected_starting_pts": results["expected_starting_pts"],
            "expected_captain_bonus": results["expected_captain_bonus"],
            "budget_remaining": float(args.budget - results["cost"]),
            "club_distribution": results["club_counts"],
            "solver_status": results["status"],
            "players": squad[["name", "position", "team", "current_price", "predicted_points", "starter", "captain", "vice_captain", "bench_order"]].to_dict(orient="records")
        }
        with open(squad_json, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Squad details saved to {squad_csv}")
        print(f"Summary JSON saved to {squad_json}")
        
    except Exception as e:
        print(f"\nERROR running squad optimizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
