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
from src.optimization.squad_optimizer import optimize_squad, enforce_leakage_guard
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

def optimize_transfers(
    df: pd.DataFrame,
    current_squad: List[int],
    bank: float = 0.0,
    free_transfers: int = 1,
    hit_cost: float = 4.0
) -> Dict[str, Any]:
    """
    Optimize transfers for the upcoming gameweek using Integer Linear Programming.
    Compares the optimized transfer strategy against a strict Hold baseline.
    """
    # 1. Validation & Leakage Guard
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
    
    # Check current squad size
    if len(current_squad) != 15:
        raise ValueError(f"Current squad must contain exactly 15 players, got {len(current_squad)}")
        
    # Check that all current players exist in the predictions pool
    invalid_owned = [code for code in current_squad if code not in df["code"].values]
    if invalid_owned:
        raise ValueError(f"Current squad players not found in prediction pool: {invalid_owned}")
        
    # Mark owned players in pool
    df["owned"] = df["code"].isin(current_squad).astype(int)
    
    # Budget calculation (current squad selling price + bank)
    # NOTE LIMITATION: We assume selling price = current_price. FPL rules incorporate purchase price
    # and profit tax (0.5 * profit), which requires transaction history tracking.
    current_squad_value = df[df["owned"] == 1]["current_price"].sum()
    max_budget = float(current_squad_value + bank)
    
    # 2. RUN HOLD BASELINE (Strictly 0 transfers)
    hold_df = df[df["owned"] == 1].copy().reset_index(drop=True)
    try:
        hold_results = optimize_squad(hold_df, budget=1000.0) # Cost constraint is moot here
    except Exception as e:
        raise ValueError(f"Hold baseline strategy is infeasible! Check if current squad satisfies FPL formation rules: {e}")
        
    # 3. RUN TRANSFER OPTIMIZATION
    prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)
    
    # Binary variables keyed by DataFrame index
    x = pulp.LpVariable.dicts("squad", df.index, cat=pulp.LpBinary)
    in_vars = pulp.LpVariable.dicts("in", df.index, cat=pulp.LpBinary)
    out_vars = pulp.LpVariable.dicts("out", df.index, cat=pulp.LpBinary)
    s = pulp.LpVariable.dicts("starter", df.index, cat=pulp.LpBinary)
    c = pulp.LpVariable.dicts("captain", df.index, cat=pulp.LpBinary)
    v = pulp.LpVariable.dicts("vice", df.index, cat=pulp.LpBinary)
    
    # Continuous variable representing penalized transfers
    y = pulp.LpVariable("penalized_transfers", lowBound=0, cat=pulp.LpContinuous)
    
    # Add Transfer Logic constraints
    for i in df.index:
        u_i = df.loc[i, "owned"]
        # Final squad membership definition
        prob += x[i] == u_i + in_vars[i] - out_vars[i]
        # Can only transfer in if not already owned
        prob += in_vars[i] <= 1 - u_i
        # Can only transfer out if already owned
        prob += out_vars[i] <= u_i
        # Cannot transfer both in and out in the same week
        prob += in_vars[i] + out_vars[i] <= 1
        
    # Add FPL constraints on the final squad (x)
    add_squad_size_constraint(prob, x, df)
    add_position_constraints(prob, x, df)
    add_budget_constraint(prob, x, df, budget=max_budget)
    add_club_constraints(prob, x, df)
    
    # Add Starting XI constraints on final squad (s)
    add_starting_xi_constraints(prob, x, s, df)
    add_starter_position_constraints(prob, s, df)
    add_captaincy_constraints(prob, s, c, v, df)
    
    # Transfer penalty logic
    prob += y >= pulp.lpSum(in_vars[i] for i in df.index) - free_transfers
    
    # Objective function: Maximize Expected Points - Penalty Hits
    base_objective = get_optimization_objective(s, c, v, df, target_points_col=points_col)
    prob += base_objective - hit_cost * y
    
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]
    
    if status_str != "Optimal":
        raise ValueError(f"FPL Transfer Optimization is infeasible! Status: {status_str}")
        
    # Extract solution flags
    df["selected"] = [int(x[i].varValue > 0.5) for i in df.index]
    df["transferred_in"] = [int(in_vars[i].varValue > 0.5) for i in df.index]
    df["transferred_out"] = [int(out_vars[i].varValue > 0.5) for i in df.index]
    df["starter"] = [int(s[i].varValue > 0.5) for i in df.index]
    df["captain"] = [int(c[i].varValue > 0.5) for i in df.index]
    df["vice_captain"] = [int(v[i].varValue > 0.5) for i in df.index]
    
    squad_df = df[df["selected"] == 1].copy()
    
    # Outfield substitutes ordered descending by points
    bench_gk = squad_df[(squad_df["starter"] == 0) & (squad_df["position"] == "GK")]
    bench_outfield = squad_df[(squad_df["starter"] == 0) & (squad_df["position"] != "GK")].sort_values(points_col, ascending=False)
    
    squad_df["bench_order"] = None
    if not bench_gk.empty:
        squad_df.loc[bench_gk.index, "bench_order"] = "GK"
    for index, (idx, row) in enumerate(bench_outfield.iterrows(), 1):
        squad_df.loc[idx, "bench_order"] = index
        
    # Transfers in / out
    transfers_in = df[df["transferred_in"] == 1][["code", "name", "position", "team", "current_price", points_col]].to_dict(orient="records")
    transfers_out = df[df["transferred_out"] == 1][["code", "name", "position", "team", "current_price", points_col]].to_dict(orient="records")
    
    num_transfers = len(transfers_in)
    transfer_penalty = float(max(0, num_transfers - free_transfers) * hit_cost)
    
    transfer_cost = squad_df["current_price"].sum()
    expected_starting_pts = squad_df[squad_df["starter"] == 1][points_col].sum()
    expected_captain_bonus = squad_df[squad_df["captain"] == 1][points_col].sum()
    transfer_expected_points = expected_starting_pts + expected_captain_bonus
    net_expected_points = transfer_expected_points - transfer_penalty
    
    # Delta comparison
    delta = net_expected_points - hold_results["expected_points"]
    recommendation = "TRANSFER" if delta > 1e-5 else "HOLD"
    
    # Formation
    starters = squad_df[squad_df["starter"] == 1]
    num_def = starters[starters["position"] == "DEF"].shape[0]
    num_mid = starters[starters["position"] == "MID"].shape[0]
    num_fwd = starters[starters["position"] == "FWD"].shape[0]
    formation = f"{num_def}-{num_mid}-{num_fwd}"
    
    # Club counts
    club_counts = squad_df["team"].value_counts().to_dict()
    
    return {
        "hold_results": hold_results,
        "transfers_in": transfers_in,
        "transfers_out": transfers_out,
        "num_transfers": num_transfers,
        "transfer_penalty": transfer_penalty,
        "expected_points": transfer_expected_points,
        "net_expected_points": net_expected_points,
        "expected_starting_pts": expected_starting_pts,
        "expected_captain_bonus": expected_captain_bonus,
        "delta": delta,
        "recommendation": recommendation,
        "squad": squad_df,
        "cost": float(transfer_cost),
        "max_budget": max_budget,
        "formation": formation,
        "club_counts": club_counts,
        "status": status_str
    }


def main():
    parser = argparse.ArgumentParser(description="FPL Weekly Transfer Optimizer")
    parser.add_argument("--season", type=str, required=True, help="Target season (e.g. 2024-25)")
    parser.add_argument("--gw", type=int, required=True, help="Target gameweek (1-38)")
    parser.add_argument("--squad", type=str, required=True, help="Path to current_squad.json")
    parser.add_argument("--free-transfers", type=int, default=1, help="Available free transfers (default 1)")
    parser.add_argument("--hit-cost", type=float, default=4.0, help="Points hit penalty per extra transfer (default 4.0)")
    parser.add_argument("--allow-hits", action="store_true", help="Allow making transfers that incur hit penalties")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"FPL WEEKLY TRANSFER OPTIMIZER: {args.season} GW {args.gw}")
    print("=" * 60)
    
    # Load current squad JSON
    squad_path = Path(args.squad)
    if not squad_path.exists():
        print(f"Current squad file not found at {squad_path}!")
        sys.exit(1)
        
    with open(squad_path, "r") as f:
        squad_data = json.load(f)
        
    current_player_codes = squad_data.get("players", [])
    bank = squad_data.get("bank", 0.0)
    
    # Load predictions CSV
    predictions_csv = RESULTS_DIR / f"predictions_{args.season}_gw{args.gw}.csv"
    if not predictions_csv.exists():
        print(f"Predictions file not found at {predictions_csv}. Running Phase 3A to generate predictions...")
        try:
            predict_gameweek(season=args.season, gw=args.gw, horizon=1)
        except Exception as e:
            print(f"Failed to auto-generate predictions: {e}")
            sys.exit(1)
            
    df = pd.read_csv(predictions_csv)
    
    # Resolve configurable hit cost limit
    configured_hit_cost = args.hit_cost
    if not args.allow_hits:
        # If hits are not allowed, we force a massive hit cost (e.g. 1000 points)
        # to guarantee the solver will never exceed available free transfers!
        configured_hit_cost = 1000.0
        
    try:
        results = optimize_transfers(
            df=df,
            current_squad=current_player_codes,
            bank=bank,
            free_transfers=args.free_transfers,
            hit_cost=configured_hit_cost
        )
        
        print(f"\n==================================================")
        print(f"TRANSFER OPTIMIZATION DECISION: {results['recommendation']}")
        print(f"==================================================")
        print(f"Current Squad Expected Points (Hold): {results['hold_results']['expected_points']:.2f}")
        print(f"Transfer Strategy Expected Points:      {results['expected_points']:.2f}")
        print(f"Transfer Hit Penalty:                  -{results['transfer_penalty']:.1f}")
        print(f"Net Transfer Expected Points:           {results['net_expected_points']:.2f}")
        print(f"Net Improvement (Delta):               {results['delta']:+.2f}")
        print(f"--------------------------------------------------")
        
        if results["num_transfers"] > 0:
            print("RECOMMENDED TRANSFERS:\n")
            for player in results["transfers_out"]:
                print(f"  OUT: {player['name']:<25} ({player['position']}) £{player['current_price']:.1f}m | Exp Pts: {player['predicted_points']:.2f}")
            for player in results["transfers_in"]:
                print(f"  IN:  {player['name']:<25} ({player['position']}) £{player['current_price']:.1f}m | Exp Pts: {player['predicted_points']:.2f}")
            print(f"\nTransfers made: {results['num_transfers']} (Free: {args.free_transfers}, Hits: {max(0, results['num_transfers'] - args.free_transfers)})")
        else:
            print("RECOMMENDATION: HOLD (No transfers recommended this week.)")
            
        print(f"\n==================================================")
        print(f"OPTIMIZED SQUAD (POST-TRANSFERS)")
        print(f"==================================================")
        print(f"Formation: {results['formation']}")
        print(f"Total Cost: £{results['cost']:.1f}m (Max Available: £{results['max_budget']:.1f}m)")
        print(f"--------------------------------------------------")
        
        squad = results["squad"]
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
        print(f"SQUAD BUDGET SUMMARY")
        print(f"==================================================")
        print(f"Final Squad Value:  £{results['cost']:.1f}m")
        print(f"Remaining Bank Cash: £{results['max_budget'] - results['cost']:.1f}m")
        print(f"==================================================\n")
        
        # Save output to CSV / JSON
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        transfer_csv = RESULTS_DIR / f"transfers_{args.season}_gw{args.gw}.csv"
        transfer_json = RESULTS_DIR / f"transfers_{args.season}_gw{args.gw}.json"
        
        squad.to_csv(transfer_csv, index=False)
        
        # Write JSON Summary
        summary = {
            "season": args.season,
            "gw": args.gw,
            "hold_expected_points": results["hold_results"]["expected_points"],
            "transfer_expected_points": results["expected_points"],
            "transfer_penalty": results["transfer_penalty"],
            "net_expected_points": results["net_expected_points"],
            "net_improvement_delta": results["delta"],
            "recommendation_decision": results["recommendation"],
            "transfers_out": results["transfers_out"],
            "transfers_in": results["transfers_in"],
            "formation": results["formation"],
            "total_cost": results["cost"],
            "max_budget": results["max_budget"],
            "budget_remaining": float(results["max_budget"] - results["cost"]),
            "club_distribution": results["club_counts"],
            "solver_status": results["status"],
            "players": squad[["name", "position", "team", "current_price", "predicted_points", "starter", "captain", "vice_captain", "bench_order"]].to_dict(orient="records")
        }
        with open(transfer_json, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Transfer details saved to {transfer_csv}")
        print(f"Summary JSON saved to {transfer_json}")
        
    except Exception as e:
        print(f"\nERROR running transfer optimizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
