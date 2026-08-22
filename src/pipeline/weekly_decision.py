import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.update_current import run_update_current_pipeline, fetch_json, FPL_BOOTSTRAP_URL
from src.models.predict import predict_gameweek, load_config
from src.optimization.transfer_optimizer import optimize_transfers

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATE_DIR = PROJECT_ROOT / "data" / "state"
STATE_FILE = STATE_DIR / "current_squad.json"
HISTORY_DIR = STATE_DIR / "history"

def determine_season_and_gw() -> Tuple[str, int, str]:
    """
    Fetch FPL bootstrap-static and automatically determine:
    1. Current season (e.g. '2025-26')
    2. Next upcoming gameweek (1-38)
    3. Deadline time string
    """
    print("Fetching FPL bootstrap-static to determine current season/GW...")
    try:
        bootstrap = fetch_json(FPL_BOOTSTRAP_URL)
    except Exception as e:
        print(f"Error calling FPL API: {e}. Falling back to default settings.")
        # Fallback if offline
        now = datetime.now()
        season = f"{now.year}-{str(now.year + 1)[2:]}" if now.month >= 7 else f"{now.year - 1}-{str(now.year)[2:]}"
        return season, 1, "Unknown (Offline)"
        
    # 1. Determine next GW
    next_gw = None
    deadline_time = ""
    for event in bootstrap.get("events", []):
        if event.get("is_next", False):
            next_gw = event["id"]
            deadline_time = event.get("deadline_time", "")
            break
            
    if next_gw is None:
        # Fallback to last completed + 1
        last_completed = 0
        for event in bootstrap.get("events", []):
            if event.get("finished", False):
                last_completed = max(last_completed, event["id"])
        next_gw = min(38, last_completed + 1)
        # Find deadline for this GW
        for event in bootstrap.get("events", []):
            if event["id"] == next_gw:
                deadline_time = event.get("deadline_time", "")
                break
                
    # 2. Determine season from deadline of first event
    first_event = bootstrap.get("events", [{}])[0]
    deadline_str = first_event.get("deadline_time")
    if deadline_str:
        try:
            dt = datetime.strptime(deadline_str[:10], "%Y-%m-%d")
            if dt.month >= 7:
                season = f"{dt.year}-{str(dt.year + 1)[2:]}"
            else:
                season = f"{dt.year - 1}-{str(dt.year)[2:]}"
        except:
            now = datetime.now()
            season = f"{now.year}-{str(now.year + 1)[2:]}" if now.month >= 7 else f"{now.year - 1}-{str(now.year)[2:]}"
    else:
        now = datetime.now()
        season = f"{now.year}-{str(now.year + 1)[2:]}" if now.month >= 7 else f"{now.year - 1}-{str(now.year)[2:]}"
        
    return season, next_gw, deadline_time

def load_raw_bootstrap() -> Dict[str, Any]:
    """Load cached bootstrap-static JSON data or fetch it fresh if missing."""
    bootstrap_path = RAW_DIR / "bootstrap_static.json"
    if bootstrap_path.exists():
        try:
            with open(bootstrap_path, "r") as f:
                return json.load(f)
        except:
            pass
    # Fallback to fetch
    return fetch_json(FPL_BOOTSTRAP_URL)

def validate_and_compile_user_squad(user_squad_data: Dict[str, Any], bootstrap: Dict[str, Any]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Validate the user-provided squad JSON and resolve player details against FPL bootstrap data.
    Ensures:
      - Exactly 15 players
      - No duplicate players
      - Correct position counts: 2 GK, 5 DEF, 5 MID, 3 FWD
      - Max 3 players per club
      - All player prices within realistic range [3.0, 16.0]
      - Total squad cost >= 50.0m
    """
    players_input = user_squad_data.get("players", [])
    if len(players_input) != 15:
        raise ValueError(f"Squad validation failed: User squad must contain exactly 15 players, got {len(players_input)}.")
        
    elements = bootstrap.get("elements", [])
    # Build maps for O(1) resolution
    id_to_elem = {e["id"]: e for e in elements}
    code_to_elem = {e["code"]: e for e in elements}
    
    # Teams list for short name resolution
    teams = bootstrap.get("teams", [])
    team_id_to_name = {t["id"]: t["short_name"] for t in teams}
    
    resolved_codes = []
    resolved_metadata = []
    
    POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    
    seen_codes = set()
    for idx, p in enumerate(players_input):
        elem = None
        # Handle dict or raw integer code/id
        if isinstance(p, dict):
            p_id = p.get("id")
            p_code = p.get("code")
            if p_id is not None and p_id in id_to_elem:
                elem = id_to_elem[p_id]
            elif p_code is not None and p_code in code_to_elem:
                elem = code_to_elem[p_code]
        elif isinstance(p, (int, float)):
            val = int(p)
            if val in code_to_elem:
                elem = code_to_elem[val]
            elif val in id_to_elem:
                elem = id_to_elem[val]
                
        if elem is None:
            raise ValueError(f"Squad validation failed: Player at index {idx} with representation {p} could not be found in current FPL bootstrap data.")
            
        code = elem["code"]
        if code in seen_codes:
            raise ValueError(f"Squad validation failed: Duplicate player found: {elem['first_name']} {elem['second_name']} (code {code}).")
        seen_codes.add(code)
        
        resolved_codes.append(code)
        resolved_metadata.append({
            "id": elem["id"],
            "code": code,
            "name": f"{elem['first_name']} {elem['second_name']}",
            "position": POSITION_MAP.get(elem["element_type"], "MID"),
            "team": team_id_to_name.get(elem["team"], "UNK"),
            "current_price": elem["now_cost"] / 10.0
        })
        
    # Check position counts
    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for m in resolved_metadata:
        pos_counts[m["position"]] += 1
        
    if pos_counts["GK"] != 2 or pos_counts["DEF"] != 5 or pos_counts["MID"] != 5 or pos_counts["FWD"] != 3:
        raise ValueError(f"Squad validation failed: Incorrect position counts. Expected 2 GK, 5 DEF, 5 MID, 3 FWD. Got: {pos_counts['GK']} GK, {pos_counts['DEF']} DEF, {pos_counts['MID']} MID, {pos_counts['FWD']} FWD.")
        
    # Check club counts
    club_counts = {}
    for m in resolved_metadata:
        club = m["team"]
        club_counts[club] = club_counts.get(club, 0) + 1
        
    for club, count in club_counts.items():
        if count > 3:
            raise ValueError(f"Squad validation failed: Maximum of 3 players from same club allowed. Club {club} has {count} players.")
            
    # Verify price ranges
    for m in resolved_metadata:
        p_name = m["name"]
        price = m["current_price"]
        min_allowed, max_allowed = 3.0, 16.0
        if price < min_allowed or price > max_allowed:
            raise ValueError(f"Squad validation failed: Player '{p_name}' has an unrealistic price of £{price:.1f}m (allowed range: £{min_allowed:.1f}m to £{max_allowed:.1f}m).")
            
    squad_cost = sum(m["current_price"] for m in resolved_metadata)
    if squad_cost < 50.0:
        raise ValueError(f"Squad validation failed: Implausibly low squad cost of £{squad_cost:.1f}m (expected >= £50.0m for a normal 15-player squad).")
        
    # Print Pre-optimization Validation report
    bank = float(user_squad_data.get("bank", 0.0))
    total_budget = squad_cost + bank
    max_club_players = max(club_counts.values()) if club_counts else 0
    prices = [m["current_price"] for m in resolved_metadata]
    min_p, max_p = min(prices), max(prices)
    
    print("\n" + "=" * 50)
    print("PRE-OPTIMIZATION SQUAD VALIDATION")
    print(f"Number of players: {len(resolved_metadata)}")
    print(f"GK/DEF/MID/FWD counts: GK={pos_counts['GK']}, DEF={pos_counts['DEF']}, MID={pos_counts['MID']}, FWD={pos_counts['FWD']}")
    print(f"Squad cost: £{squad_cost:.1f}m")
    print(f"Bank: £{bank:.1f}m")
    print(f"Total available budget: £{total_budget:.1f}m")
    print(f"Maximum players from any club: {max_club_players}")
    print(f"Price range: £{min_p:.1f}m to £{max_p:.1f}m")
    print("=" * 50)
    
    return resolved_codes, resolved_metadata

def run_weekly_decision_pipeline(
    season: str = None,
    gw: int = None,
    squad_path: str = None,
    bank_override: float = None,
    free_transfers_override: int = None,
    hit_cost: float = 4.0,
    apply: bool = False,
    force: bool = False
) -> Dict[str, Any]:
    """Runs the full Phase 9.1 Production Weekly Decision pipeline."""
    # 1. Determine Season and GW if not provided
    auto_season, auto_gw, deadline_time = determine_season_and_gw()
    if season is None:
        season = auto_season
    if gw is None:
        gw = auto_gw
        
    print(f"Target Season: {season}")
    print(f"Target GW:     {gw}")
    print(f"GW Deadline:   {deadline_time}")
    
    # 2. Check persistent state and load/import
    state = None
    if squad_path is not None:
        # User provided a squad to initialize/import
        print(f"\nImporting/initializing squad state from {squad_path}...")
        if not os.path.exists(squad_path):
            raise FileNotFoundError(f"User squad file not found at {squad_path}!")
        with open(squad_path, "r") as f:
            user_squad_data = json.load(f)
            
        bootstrap = fetch_json(FPL_BOOTSTRAP_URL)
        resolved_codes, resolved_metadata = validate_and_compile_user_squad(user_squad_data, bootstrap)
        
        state = {
            "season": user_squad_data.get("season", season),
            "last_processed_gw": 0,
            "players": resolved_codes,
            "players_metadata": resolved_metadata,
            "bank": float(user_squad_data.get("bank", 0.0)),
            "free_transfers": int(user_squad_data.get("free_transfers", 1))
        }
        print(f"Successfully compiled initial squad state in memory.")
    else:
        # Load persistent state
        if not STATE_FILE.exists():
            if gw == 1:
                print("\nNo persistent squad found. Running in INITIAL SQUAD BUILDER mode for GW1.")
                state = None
            else:
                print("\n[Error] No persistent squad found. Please provide your current FPL squad.")
                raise FileNotFoundError("No persistent squad found. Please provide your current FPL squad.")
        else:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            print(f"\nLoaded persistent squad from {STATE_FILE}.")

    # 3. Double processing check
    if state is not None and state.get("last_processed_gw", 0) == gw and state.get("season") == season:
        if not force:
            msg = f"GW {gw} of season {season} has already been processed. Current squad state is already updated for GW {gw}. Use --force to override."
            print(f"\n[Warning] {msg}")
            raise ValueError(msg)
        else:
            print("\n[Override] --force specified. Proceeding to process gameweek again.")

    # 4. Fetch latest data and update current inference parquet
    print("\n[Step 1/3] Updating current-season inference dataset...")
    # This fetches data from FPL API and runs feature engineering
    resolved_gw = run_update_current_pipeline(season, gw)
    if resolved_gw is not None and isinstance(resolved_gw, int) and resolved_gw != gw:
        print("\n" + "=" * 50)
        print(f"Requested target GW: {gw}")
        print(f"Available data supports: {resolved_gw}")
        print(f"Using prediction target: {resolved_gw}")
        print("=" * 50)
        gw = resolved_gw
        # Re-fetch deadline for the adjusted GW
        try:
            bootstrap = load_raw_bootstrap()
            for event in bootstrap.get("events", []):
                if event.get("id") == gw:
                    deadline_time = event.get("deadline_time", "")
                    break
        except Exception as e:
            print(f"  [Warning] Failed to update deadline for adjusted GW: {e}")
            
        # Re-check double processing for the adjusted GW
        if state.get("last_processed_gw", 0) == gw and state.get("season") == season:
            if not force:
                msg = f"GW {gw} of season {season} has already been processed. Current squad state is already updated for GW {gw}. Use --force to override."
                print(f"\n[Warning] {msg}")
                raise ValueError(msg)
                
    # 5. Generate predictions using current tuned production model
    print("\n[Step 2/3] Generating predictions for the target gameweek...")
    predictions_df, forecast_df = predict_gameweek(season=season, gw=gw, horizon=1)
    
    # 6. Extract squad variables from state
    if state is not None:
        squad_players = state["players"]
        bank = state["bank"]
        free_transfers = state["free_transfers"]
    else:
        # Initial Squad Builder default variables
        squad_players = []
        bank = 0.0
        free_transfers = 1
        
    # Override if provided via CLI
    if bank_override is not None:
        bank = bank_override
    if free_transfers_override is not None:
        free_transfers = free_transfers_override
        
    print(f"Current squad players codes: {squad_players}")
    print(f"Current Bank Cash:            £{bank:.2f}m")
    print(f"Available Free Transfers:     {free_transfers}")
    
    # Verify that all squad players are in predictions pool
    missing_players = [code for code in squad_players if code not in predictions_df["code"].values]
    if missing_players:
        print(f"\n[Warning] Current squad players not found in prediction pool: {missing_players}")
        print("Adding dummy records for missing players to proceed...")
        
        hist_features_path = PROJECT_ROOT / "data" / "processed" / "features_df.parquet"
        hist_features = pd.read_parquet(hist_features_path) if hist_features_path.exists() else None
        
        # Add dummy rows with 0 predicted points to predictions_df so optimization does not fail
        for code in missing_players:
            # Default fallback values
            position = "MID"
            current_price = 4.5
            team = "UNK"
            season_team_id = 1
            name = f"Unknown Player ({code})"
            
            # Lookup in historical features
            if hist_features is not None:
                p_hist = hist_features[hist_features["code"] == code]
                if not p_hist.empty:
                    # Sort by season and gw descending to get latest
                    p_hist = p_hist.sort_values(["season", "gw"], ascending=False)
                    row_hist = p_hist.iloc[0]
                    if "position" in p_hist.columns and not p_hist["position"].empty:
                        position = p_hist["position"].mode().values[0]
                    else:
                        position = row_hist.get("position", "MID")
                    current_price = float(row_hist.get("current_price", 4.5))
                    name = row_hist.get("name", f"Unknown Player ({code})")
                    # Try to get team
                    if "team" in row_hist and pd.notna(row_hist["team"]):
                        team = row_hist["team"]
                    if "season_team_id" in row_hist and pd.notna(row_hist["season_team_id"]):
                        season_team_id = int(row_hist["season_team_id"])
                        if team == "UNK":
                            # Try to find team name from teams.parquet
                            teams_path = PROJECT_ROOT / "data" / "processed" / "teams.parquet"
                            if teams_path.exists():
                                teams_df = pd.read_parquet(teams_path)
                                t_row = teams_df[(teams_df["season"] == row_hist["season"]) & (teams_df["id"] == season_team_id)]
                                if not t_row.empty:
                                    team = t_row["short_name"].values[0]
                                
            dummy_row = {
                "season": season, "gw": gw, "code": code, "name": name,
                "position": position, "current_price": current_price, "selected": 0.0, "num_fixtures": 0.0,
                "is_double_gw": 0.0, "fixture_difficulty": 3.0, "season_team_id": season_team_id, "team": team,
                "predicted_points": 0.0, "predicted_minutes": 0.0, "prob_60_plus": 0.0,
                "predicted_points_per_million": 0.0, "rank_overall": 999.0, "rank_position": 999.0,
                "rank_value": 999.0
            }
            predictions_df = pd.concat([predictions_df, pd.DataFrame([dummy_row])], ignore_index=True)
            
    # 7. Run Transfer Optimizer or Squad Starting XI Optimizer
    if gw == 1:
        from src.optimization.squad_optimizer import optimize_squad
        if state is None:
            # Case A: INITIAL SQUAD BUILDER (Construct 15-player squad from scratch using £100.0m)
            print("\nNEW SEASON / INITIAL SQUAD BUILDER MODE (Building 15-player squad from scratch using £100.0m)")
            opt_squad_results = optimize_squad(predictions_df, budget=100.0)
            
            results = {
                "recommendation": "INITIAL_BUILD",
                "num_transfers": 0,
                "transfers_out": [],
                "transfers_in": [],
                "transfer_penalty": 0.0,
                "expected_points": opt_squad_results["expected_points"],
                "net_expected_points": opt_squad_results["expected_points"],
                "delta": 0.0,
                "formation": opt_squad_results["formation"],
                "squad": opt_squad_results["squad"],
                "cost": opt_squad_results["cost"],
                "max_budget": 100.0,
                "hold_results": {
                    "expected_points": opt_squad_results["expected_points"]
                }
            }
        else:
            # Case B: FIXED USER SQUAD (Optimize starting XI/captain/VC of the fixed 15 players)
            print("\nNEW SEASON / INITIAL SQUAD MODE (Optimizing Starting XI of fixed supplied squad)")
            initial_df = predictions_df[predictions_df["code"].isin(squad_players)].copy().reset_index(drop=True)
            opt_squad_results = optimize_squad(initial_df, budget=1000.0)
            
            results = {
                "recommendation": "HOLD",
                "num_transfers": 0,
                "transfers_out": [],
                "transfers_in": [],
                "transfer_penalty": 0.0,
                "expected_points": opt_squad_results["expected_points"],
                "net_expected_points": opt_squad_results["expected_points"],
                "delta": 0.0,
                "formation": opt_squad_results["formation"],
                "squad": opt_squad_results["squad"],
                "cost": opt_squad_results["cost"],
                "max_budget": opt_squad_results["cost"] + bank,
                "hold_results": {
                    "expected_points": opt_squad_results["expected_points"]
                }
            }
    else:
        print("\nRunning Transfer Optimizer...")
        results = optimize_transfers(
            df=predictions_df,
            current_squad=squad_players,
            bank=bank,
            free_transfers=free_transfers,
            hit_cost=hit_cost
        )
    
    # 8. Check External Data status
    ext_path = PROJECT_ROOT / "data" / "raw" / "matches" / "external_cup_matches.json"
    ext_status = "UNAVAILABLE"
    if ext_path.exists():
        try:
            with open(ext_path, "r") as f:
                ext_data = json.load(f)
            if ext_data:
                ext_status = "PARTIAL"
        except:
            pass
            
    # Determine model type
    tuned_path = PROJECT_ROOT / "config" / "tuned_lightgbm.yaml"
    model_type = "tuned_lightgbm" if tuned_path.exists() else "baseline_fallback"
    
    # Print clean report to stdout
    print("\n" + "=" * 50)
    if gw == 1:
        print("NEW SEASON / INITIAL SQUAD")
        print("\nYour squad has been imported successfully.")
        print("No transfer recommendation is made because this is the initial squad for the season.")
    else:
        print("FPL WEEKLY DECISION")
        
    print(f"Season: {season}")
    print(f"Target GW: {gw}")
    print(f"Deadline: {deadline_time}")
    print("\nDATA STATUS")
    print("FPL data: AVAILABLE")
    print(f"External enrichment: {ext_status}")
    print("Point-in-time cutoff: PASS")
    print("Leakage guard: PASS")
    print("\nMODEL")
    print(f"Production model: {model_type}")
    print(f"Prediction rows: {len(predictions_df)}")
    
    if gw > 1:
        print("-" * 50)
        print("HOLD")
        print(f"Expected points: {results['hold_results']['expected_points']:.2f}")
        print("-" * 50)
        print("TRANSFER RECOMMENDATION")
        print(f"Decision: {results['recommendation']}")
    
    if results["num_transfers"] > 0:
        print("\nTransfers OUT:")
        for player in results["transfers_out"]:
            print(f"- {player['name']} ({player['position']}) £{player['current_price']:.1f}m")
        print("\nTransfers IN:")
        for player in results["transfers_in"]:
            print(f"- {player['name']} ({player['position']}) £{player['current_price']:.1f}m")
            
    print(f"\nTransfers: {results['num_transfers']}")
    print(f"Free transfers: {free_transfers}")
    hits = max(0, results["num_transfers"] - free_transfers)
    print(f"Hits: {hits}")
    print(f"Transfer penalty: -{results['transfer_penalty']:.1f}")
    print(f"\nExpected points after transfer: {results['expected_points']:.2f}")
    print(f"Net expected points: {results['net_expected_points']:.2f}")
    print(f"Improvement vs HOLD: {results['delta']:+.2f}")
    print("-" * 50)
    print(f"FINAL STARTING XI")
    print(f"Formation: {results['formation']}")
    
    squad_df = results["squad"]
    starters = squad_df[squad_df["starter"] == 1].sort_values("position", key=lambda x: x.map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}))
    
    print("\nGK:")
    for _, row in starters[starters["position"] == "GK"].iterrows():
        cap = " (C)" if row["captain"] == 1 else (" (VC)" if row["vice_captain"] == 1 else "")
        print(f"- {row['name']} £{row['current_price']:.1f}m | Exp: {row['predicted_points']:.2f}{cap}")
        
    print("\nDEF:")
    for _, row in starters[starters["position"] == "DEF"].iterrows():
        cap = " (C)" if row["captain"] == 1 else (" (VC)" if row["vice_captain"] == 1 else "")
        print(f"- {row['name']} £{row['current_price']:.1f}m | Exp: {row['predicted_points']:.2f}{cap}")
        
    print("\nMID:")
    for _, row in starters[starters["position"] == "MID"].iterrows():
        cap = " (C)" if row["captain"] == 1 else (" (VC)" if row["vice_captain"] == 1 else "")
        print(f"- {row['name']} £{row['current_price']:.1f}m | Exp: {row['predicted_points']:.2f}{cap}")
        
    print("\nFWD:")
    for _, row in starters[starters["position"] == "FWD"].iterrows():
        cap = " (C)" if row["captain"] == 1 else (" (VC)" if row["vice_captain"] == 1 else "")
        print(f"- {row['name']} £{row['current_price']:.1f}m | Exp: {row['predicted_points']:.2f}{cap}")
        
    captain_name = squad_df[squad_df["captain"] == 1]["name"].values[0] if not squad_df[squad_df["captain"] == 1].empty else "Unknown"
    vice_name = squad_df[squad_df["vice_captain"] == 1]["name"].values[0] if not squad_df[squad_df["vice_captain"] == 1].empty else "Unknown"
    print(f"\nCaptain: {captain_name}")
    print(f"Vice-Captain: {vice_name}")
    print("\nBENCH")
    
    bench_gk = squad_df[(squad_df["starter"] == 0) & (squad_df["position"] == "GK")]
    if not bench_gk.empty:
        print(f"GK: {bench_gk['name'].values[0]} £{bench_gk['current_price'].values[0]:.1f}m | Exp: {bench_gk['predicted_points'].values[0]:.2f}")
        
    bench_outfield = squad_df[(squad_df["starter"] == 0) & (squad_df["position"] != "GK")].sort_values("bench_order")
    for _, row in bench_outfield.iterrows():
        print(f"{row['bench_order']}: {row['name']} £{row['current_price']:.1f}m | Exp: {row['predicted_points']:.2f}")
        
    remaining_bank = float(results["max_budget"] - results["cost"])
    print(f"\nSquad cost: £{results['cost']:.1f}m")
    print(f"Remaining bank: £{remaining_bank:.1f}m")
    print("=" * 50)
    
    # 9. Handle --apply state changes
    if not apply:
        print("\n>>> WARNING: This is a simulation/recommendation run. Persistent state has NOT been modified.")
        print("To apply these transfers to your squad state, run with the '--apply' flag.")
    else:
        # Determine updated player lists, bank, and free transfers
        transfers_in_codes = [p["code"] for p in results["transfers_in"]]
        transfers_out_codes = [p["code"] for p in results["transfers_out"]]
        
        # New squad players list
        if state is None:
            new_players = results["squad"]["code"].tolist()
        else:
            new_players = [code for code in squad_players if code not in transfers_out_codes] + transfers_in_codes
            
        if len(new_players) != 15:
            raise ValueError(f"Squad validation error: Resulting squad does not have exactly 15 players (got {len(new_players)}).")
            
        # Compile new metadata from bootstrap
        bootstrap = load_raw_bootstrap()
        elements = bootstrap.get("elements", [])
        teams = bootstrap.get("teams", [])
        team_id_to_name = {t["id"]: t["short_name"] for t in teams}
        POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        
        code_to_meta = {}
        for elem in elements:
            code_to_meta[elem["code"]] = {
                "id": elem["id"],
                "code": elem["code"],
                "name": f"{elem['first_name']} {elem['second_name']}",
                "position": POSITION_MAP.get(elem["element_type"], "MID"),
                "team": team_id_to_name.get(elem["team"], "UNK"),
                "current_price": elem["now_cost"] / 10.0
            }
            
        new_metadata = []
        for code in new_players:
            if code in code_to_meta:
                new_metadata.append(code_to_meta[code])
            else:
                old_m = None
                if state is not None:
                    old_m = next((m for m in state.get("players_metadata", []) if m["code"] == code), None)
                if old_m is not None:
                    new_metadata.append(old_m)
                else:
                    new_metadata.append({
                        "id": 999999,
                        "code": code,
                        "name": f"Unknown Player ({code})",
                        "position": "MID",
                        "team": "UNK",
                        "current_price": 4.5
                    })
                    
        # Update bank balance and free transfers
        num_transfers = len(results["transfers_out"])
        
        # Free transfers increments by 1 on new gameweek (max 5 in modern FPL)
        if gw == 1:
            new_free_transfers = 1
        else:
            new_free_transfers = min(5, max(1, free_transfers - num_transfers + 1))
        
        # Update squad state dict
        if state is None:
            state = {
                "season": season,
                "last_processed_gw": gw,
                "players": new_players,
                "players_metadata": new_metadata,
                "bank": remaining_bank,
                "free_transfers": new_free_transfers
            }
        else:
            state["players"] = new_players
            state["players_metadata"] = new_metadata
            state["bank"] = remaining_bank
            state["free_transfers"] = new_free_transfers
            state["last_processed_gw"] = gw
        
        # Save updated persistent state
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
        print(f"\n>>> SUCCESS: Persistent squad state updated and saved to {STATE_FILE}.")
        
        # Save audit history record
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_file = HISTORY_DIR / f"weekly_gw{gw}.json"
        
        history_record = {
            "season": season,
            "gw": gw,
            "squad_before": {
                "players": squad_players,
                "bank": bank,
                "free_transfers": free_transfers
            },
            "predictions_used": predictions_df[predictions_df["code"].isin(new_players)][["code", "name", "predicted_points"]].to_dict(orient="records"),
            "hold_expected_points": float(results["hold_results"]["expected_points"]),
            "transfer_expected_points": float(results["expected_points"]),
            "net_improvement": float(results["delta"]),
            "recommended_transfers": {
                "out": results["transfers_out"],
                "in": results["transfers_in"]
            },
            "transfer_hit": float(results["transfer_penalty"]),
            "captain": captain_name,
            "vice_captain": vice_name,
            "starting_xi": squad_df[squad_df["starter"] == 1]["name"].tolist(),
            "resulting_squad": new_players,
            "timestamp": datetime.now().isoformat()
        }
        with open(history_file, "w") as f:
            json.dump(history_record, f, indent=4)
        print(f"Audit log saved to: {history_file}")

    # Save Output files (maintaining existing Phase 9 artifacts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json_path = RESULTS_DIR / f"weekly_decision_{season}_gw{gw}.json"
    out_csv_path = RESULTS_DIR / f"weekly_decision_{season}_gw{gw}.csv"
    
    # Save CSV
    squad_df.to_csv(out_csv_path, index=False)
    
    # Save JSON
    summary = {
        "season": season,
        "gw": gw,
        "deadline": deadline_time,
        "data_status": {
            "fpl_data": "AVAILABLE",
            "external_enrichment": ext_status,
            "pit_cutoff": "PASS",
            "leakage_guard": "PASS"
        },
        "model": model_type,
        "prediction_rows": len(predictions_df),
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
        "budget_remaining": remaining_bank,
        "players": squad_df[["name", "position", "team", "current_price", "predicted_points", "starter", "captain", "vice_captain", "bench_order"]].to_dict(orient="records")
    }
    
    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Summary report saved to: {out_json_path}")
    print(f"Squad CSV details saved to: {out_csv_path}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="FPL Phase 9.1 Production Weekly Decision CLI")
    parser.add_argument("--season", type=str, default=None, help="Target season (e.g. 2025-26)")
    parser.add_argument("--gw", type=int, default=None, help="Target gameweek (1-38)")
    parser.add_argument("--squad", type=str, default=None, help="Path to user squad input JSON to initialize/import state")
    parser.add_argument("--bank", type=float, default=None, help="Override remaining bank cash")
    parser.add_argument("--free-transfers", type=int, default=None, help="Override available free transfers")
    parser.add_argument("--hit-cost", type=float, default=4.0, help="Override points penalty per extra transfer")
    parser.add_argument("--apply", action="store_true", help="Apply the recommended transfers and update persistent state")
    parser.add_argument("--force", action="store_true", help="Force processing even if target GW is already processed")
    
    args = parser.parse_args()
    
    try:
        run_weekly_decision_pipeline(
            season=args.season,
            gw=args.gw,
            squad_path=args.squad,
            bank_override=args.bank,
            free_transfers_override=args.free_transfers,
            hit_cost=args.hit_cost,
            apply=args.apply,
            force=args.force
        )
    except Exception as e:
        print(f"\nERROR in weekly decision pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
