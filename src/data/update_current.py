import pandas as pd
import numpy as np
import json
import argparse
import sys
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.player_features import build_player_grid, add_player_rolling_features
from src.features.team_features import build_team_rolling_features
from src.features.fixture_features import build_fixture_features

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Official FPL API Endpoints
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_LIVE_GW_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

MAP_STATS = {
    "minutes": "minutes",
    "total_points": "total_points",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "bonus": "bonus",
    "bps": "bps",
    "starts": "starts",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded"
}

def load_config() -> Dict[str, Any]:
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def fetch_json(url: str) -> Any:
    """Fetch JSON data from a URL using standard urllib.request."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise ConnectionError(f"Failed to fetch data from FPL API ({url}): {e}")

def get_last_completed_gw(bootstrap_data: Dict[str, Any]) -> int:
    """Identify the latest completed gameweek automatically from FPL bootstrap."""
    last_completed = 0
    for event in bootstrap_data.get("events", []):
        if event.get("finished", False):
            last_completed = max(last_completed, event["id"])
    return last_completed

def compile_current_season_tables(
    season: str,
    target_gw: int,
    bootstrap_data: Dict[str, Any],
    fixtures_data: List[Dict[str, Any]],
    live_gws_data: Dict[int, Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compile current season data into player_gw, players, fixtures, and teams DataFrames,
    matching the historical schema exactly.
    """
    # 1. Compile Teams
    teams_rows = []
    for t in bootstrap_data["teams"]:
        teams_rows.append({
            "id": t["id"],
            "code": t["code"],
            "name": t["name"],
            "short_name": t["short_name"],
            "strength": t["strength"],
            "strength_attack_home": t["strength_attack_home"],
            "strength_attack_away": t["strength_attack_away"],
            "strength_defence_home": t["strength_defence_home"],
            "strength_defence_away": t["strength_defence_away"],
            "season": season
        })
    teams_df = pd.DataFrame(teams_rows)

    # 2. Compile Players metadata
    players_rows = []
    player_team_map = {}
    for p in bootstrap_data["elements"]:
        player_id = p["id"]
        team_id = p["team"]
        player_team_map[player_id] = team_id
        
        players_rows.append({
            "player_id": player_id,
            "code": p["code"],
            "name": f"{p['first_name']} {p['second_name']}",
            "position": POSITION_MAP.get(p["element_type"], "MID"),
            "element_type": p["element_type"],
            "team": team_id,
            "current_price": p["now_cost"] / 10.0,
            "selected_ratio": float(p["selected_by_percent"]) / 100.0,
            "season": season
        })
    players_df = pd.DataFrame(players_rows)

    # 3. Compile Fixtures
    fixtures_rows = []
    for f in fixtures_data:
        # FPL API Event field contains the gameweek number, or None if unassigned
        event_gw = f.get("event")
        fixtures_rows.append({
            "id": f["id"],
            "event": event_gw if event_gw is not None else np.nan,
            "team_h": f["team_h"],
            "team_a": f["team_a"],
            "team_h_score": f.get("team_h_score", np.nan),
            "team_a_score": f.get("team_a_score", np.nan),
            "finished": f.get("finished", False),
            "minutes": f.get("minutes", 0),
            "team_h_difficulty": f.get("team_h_difficulty", 3),
            "team_a_difficulty": f.get("team_a_difficulty", 3),
            "season": season
        })
    fixtures_df = pd.DataFrame(fixtures_rows)

    # 4. Compile player_gw details
    pgw_rows = []
    player_meta_map = {r["player_id"]: r for r in players_rows}
    
    for gw, live_data in live_gws_data.items():
        for elem in live_data.get("elements", []):
            player_id = elem["id"]
            player_team_id = player_team_map.get(player_id)
            
            # Skip if player team is unknown
            if player_team_id is None:
                continue
                
            explain_list = elem.get("explain", [])
            # In player_gw.parquet, players with blank gameweeks have no fixture rows.
            if not explain_list:
                continue
                
            p_meta = player_meta_map.get(player_id, {})
            p_name = p_meta.get("name", f"Player {player_id}")
            p_pos = p_meta.get("position", "MID")
            p_team = p_meta.get("team", player_team_id)
            
            for fixt_explain in explain_list:
                fixture_id = fixt_explain["fixture"]
                
                # Retrieve stats and maps
                stat_values = {v: 0.0 for v in MAP_STATS.values()}
                
                for stat_item in fixt_explain.get("stats", []):
                    identifier = stat_item["identifier"]
                    if identifier in MAP_STATS:
                        col_name = MAP_STATS[identifier]
                        val = stat_item["value"]
                        if "expected" in col_name or "ict_index" in col_name or "influence" in col_name or "creativity" in col_name or "threat" in col_name:
                            try:
                                val = float(val)
                            except:
                                val = 0.0
                        stat_values[col_name] = val
                
                # Determine was_home and opponent
                f_row = fixtures_df[fixtures_df["id"] == fixture_id]
                was_home = 1
                opponent_team = 0
                if not f_row.empty:
                    h_team = f_row["team_h"].values[0]
                    a_team = f_row["team_a"].values[0]
                    if player_team_id == h_team:
                        was_home = 1
                        opponent_team = a_team
                    else:
                        was_home = 0
                        opponent_team = h_team
                        
                row_dict = {
                    "player_id": player_id,
                    "name": p_name,
                    "position": p_pos,
                    "team": p_team,
                    "season": season,
                    "gw": gw,
                    "fixture": fixture_id,
                    "was_home": was_home,
                    "opponent_team": int(opponent_team),
                    "value": float(df_price := p_meta.get("current_price", 5.0)) * 10.0,
                    "selected": float(df_sel := p_meta.get("selected_ratio", 0.0)),
                    "transfers_in": int(elem["stats"].get("transfers_in", 0)),
                    "transfers_out": int(elem["stats"].get("transfers_out", 0)),
                    "transfers_balance": int(elem["stats"].get("transfers_in", 0) - elem["stats"].get("transfers_out", 0))
                }
                
                # Update with main stats
                row_dict.update(stat_values)
                pgw_rows.append(row_dict)
                
    pgw_df = pd.DataFrame(pgw_rows)
    if pgw_df.empty:
        # If no live gameweeks completed yet and target_gw is 1, build player_gw rows from fixtures
        if target_gw == 1:
            gw_fixtures = fixtures_df[fixtures_df["event"] == target_gw]
            for _, player in players_df.iterrows():
                p_id = player["player_id"]
                p_team = player["team"]
                
                # Find fixtures for this team in GW 1
                team_fixtures = gw_fixtures[(gw_fixtures["team_h"] == p_team) | (gw_fixtures["team_a"] == p_team)]
                
                p_meta = player_meta_map.get(p_id, {})
                p_name = p_meta.get("name", player["name"])
                p_pos = p_meta.get("position", player["position"])
                p_team = p_meta.get("team", player["team"])
                
                for _, f in team_fixtures.iterrows():
                    was_home = 1 if f["team_h"] == p_team else 0
                    opponent_team = f["team_a"] if was_home == 1 else f["team_h"]
                    fixture_id = f["id"]
                    
                    stat_values = {v: 0.0 for v in MAP_STATS.values()}
                    row_dict = {
                        "player_id": p_id,
                        "name": p_name,
                        "position": p_pos,
                        "team": p_team,
                        "season": season,
                        "gw": target_gw,
                        "fixture": fixture_id,
                        "was_home": was_home,
                        "opponent_team": int(opponent_team),
                        "value": float(player["current_price"]) * 10.0,
                        "selected": float(player["selected_ratio"]),
                        "transfers_in": 0,
                        "transfers_out": 0,
                        "transfers_balance": 0
                    }
                    row_dict.update(stat_values)
                    pgw_rows.append(row_dict)
            pgw_df = pd.DataFrame(pgw_rows)
            
    if pgw_df.empty:
        hist_pgw_cols = ["player_id", "name", "position", "team", "season", "gw", "fixture", "was_home", "opponent_team", "value", "selected", "transfers_in", "transfers_out", "transfers_balance"] + list(MAP_STATS.values())
        pgw_df = pd.DataFrame(columns=hist_pgw_cols)
        # Set proper dtypes
        for col in pgw_df.columns:
            if col in ["player_id", "fixture", "opponent_team", "transfers_in", "transfers_out", "transfers_balance", "gw"]:
                pgw_df[col] = pgw_df[col].astype(int)
            elif col in ["value", "selected"] or col in MAP_STATS.values():
                pgw_df[col] = pgw_df[col].astype(float)
            else:
                pgw_df[col] = pgw_df[col].astype(str)
    return pgw_df, players_df, fixtures_df, teams_df

def run_update_current_pipeline(season: str, gw: int):
    """
    Weekly production ingestion pipeline:
    1. Downloads current FPL data up to GW N-1.
    2. Compiles current season parquets.
    3. Concatenates with untouched historical parquets.
    4. Computes schema-compatible player features for GW N.
    5. Saves strictly to current_features.parquet.
    """
    config = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching bootstrap data from FPL API...")
    bootstrap = fetch_json(FPL_BOOTSTRAP_URL)
    with open(RAW_DIR / "bootstrap_static.json", "w") as f:
        json.dump(bootstrap, f, indent=4)
        
    print(f"Fetching fixtures data from FPL API...")
    fixtures = fetch_json(FPL_FIXTURES_URL)
    with open(RAW_DIR / "fixtures.json", "w") as f:
        json.dump(fixtures, f, indent=4)
        
    last_completed = get_last_completed_gw(bootstrap)
    print(f"FPL API reports latest completed gameweek: GW {last_completed}")
    
    # Validation boundary
    if gw - 1 > last_completed:
        print(f"[Warning] Target GW {gw} expects completed GW {gw-1}, but FPL reports only up to GW {last_completed} finished. Falling back to GW {last_completed + 1}...")
        gw = last_completed + 1
        
    # Fetch live GW stats
    live_gws_data = {}
    for g in range(1, gw):
        print(f"Fetching live stats for GW {g}...")
        try:
            gw_data = fetch_json(FPL_LIVE_GW_URL.format(gw=g))
            with open(RAW_DIR / f"live_gw{g}.json", "w") as f:
                json.dump(gw_data, f, indent=4)
        except Exception as e:
            print(f"  [Warning] Failed to fetch live stats for GW {g}: {e}. Trying to load local cache...")
            cache_path = RAW_DIR / f"live_gw{g}.json"
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    gw_data = json.load(f)
            else:
                print(f"  [Warning] Local cache not found for GW {g}. Falling back to empty element stats.")
                gw_data = {"elements": []}
        live_gws_data[g] = gw_data
        
    print("\nCompiling current season tables...")
    curr_pgw, curr_players, curr_fixtures, curr_teams = compile_current_season_tables(
        season, gw, bootstrap, fixtures, live_gws_data
    )
    
    print(f"  Current player_gw rows: {len(curr_pgw):,}")
    print(f"  Current players rows:   {len(curr_players):,}")
    
    # Load historical datasets (strictly to read, not modify)
    hist_pgw = pd.read_parquet(PROCESSED_DIR / "player_gw.parquet")
    hist_players = pd.read_parquet(PROCESSED_DIR / "players.parquet")
    hist_fixtures = pd.read_parquet(PROCESSED_DIR / "fixtures.parquet")
    hist_teams = pd.read_parquet(PROCESSED_DIR / "teams.parquet")
    
    # Exclude target season to prevent duplicate entries
    combined_pgw = pd.concat([hist_pgw[hist_pgw["season"] != season], curr_pgw], ignore_index=True)
    combined_players = pd.concat([hist_players[hist_players["season"] != season], curr_players], ignore_index=True)
    combined_fixtures = pd.concat([hist_fixtures[hist_fixtures["season"] != season], curr_fixtures], ignore_index=True)
    combined_teams = pd.concat([hist_teams[hist_teams["season"] != season], curr_teams], ignore_index=True)
    
    print("\nExecuting feature engineering over combined datasets...")
    # Generate player grid
    player_grid = build_player_grid(combined_pgw, combined_players)
    player_features = add_player_rolling_features(player_grid)
    
    # Generate team features
    team_features = build_team_rolling_features(combined_pgw, combined_players)
    
    # Generate fixture features
    fixture_features = build_fixture_features(
        combined_pgw, combined_players, combined_fixtures, combined_teams, team_features
    )
    
    # Merge
    features_df = pd.merge(player_features, fixture_features, on=["season", "gw", "code"], how="left")
    
    # Fill BGW and opposition NaNs
    features_df["num_fixtures"] = features_df["num_fixtures"].fillna(0.0)
    features_df["is_double_gw"] = features_df["is_double_gw"].fillna(0.0)
    features_df["was_home_mean"] = features_df["was_home_mean"].fillna(0.0)
    features_df["fixture_difficulty_mean"] = features_df["fixture_difficulty_mean"].fillna(3.0)
    features_df["fixture_1_difficulty"] = features_df["fixture_1_difficulty"].fillna(3.0)
    
    opp_fill_zero = [
        "opponent_team_scored_last_3_sum", "opponent_team_scored_last_5_sum",
        "opponent_team_conceded_last_3_sum", "opponent_team_conceded_last_5_sum",
        "opponent_team_clean_sheets_last_5_sum", "opponent_static_strength_mean",
        "opponent_static_defence_mean", "opponent_static_attack_mean"
    ]
    for col in opp_fill_zero:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0.0)
            
    # Mock target columns for schema compatibility (they will be validation-guarded at inference)
    features_df["target_points"] = 0.0
    features_df["target_minutes"] = 0.0
    features_df["target_60_plus_minutes"] = 0.0
    
    # Drop raw matchday stats of the target GW to prevent leakage and match the trained schema
    cols_to_drop = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps", "starts",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "creativity", "threat", "influence", "ict_index",
        "defensive_contribution", "tackles", "clearances_blocks_interceptions", "recoveries",
        "derived_starts"
    ]
    features_df = features_df.drop(columns=cols_to_drop, errors="ignore")
    
    # Extract only the target season and upcoming GW row to current_features
    current_features_df = features_df[(features_df["season"] == season) & (features_df["gw"] == gw)].copy()
    
    # Pre-GW1 player mapping and historical feature alignment
    if last_completed == 0 and gw == 1:
        print("\nAligning pre-GW1 player rolling features with latest historical records...")
        hist_features_path = PROCESSED_DIR / "features_df.parquet"
        if hist_features_path.exists():
            # Identify columns to copy (rolling averages, form, derived stats, etc.)
            rolling_feature_cols = []
            for cat in ["player_base_rolling", "player_expected", "player_derived", "player_workload", "fixture_congestion", "team_rolling"]:
                if cat in config["features"]:
                    rolling_feature_cols.extend(config["features"][cat])
            
            # Determine which columns actually exist in features_df schema by reading head(1)
            schema_df = pd.read_parquet(hist_features_path, engine="pyarrow").head(1)
            cols_to_copy = [c for c in rolling_feature_cols if c in schema_df.columns and c in current_features_df.columns]
            
            # Read only the subset of columns to save memory
            cols_to_read = ["season", "gw", "code"] + cols_to_copy
            hist_features = pd.read_parquet(hist_features_path, columns=cols_to_read)
            
            # Sort descending by season and gw so we get the most recent record
            hist_sorted = hist_features.sort_values(["season", "gw"], ascending=False)
            latest_hist_map = hist_sorted.groupby("code").first().reset_index()
            
            # Drop the current empty/zeroed columns in current_features_df
            current_features_df = current_features_df.drop(columns=cols_to_copy, errors="ignore")
            
            # Merge latest historical values
            latest_hist_sub = latest_hist_map[["code"] + cols_to_copy].copy()
            current_features_df = pd.merge(current_features_df, latest_hist_sub, on="code", how="left")
            
            # Fill missing values for new players with 0.0 safe defaults
            for col in cols_to_copy:
                current_features_df[col] = current_features_df[col].fillna(0.0)
                
            # Re-verify and set player experience columns to 0.0 (starts, minutes, gws this season)
            exp_cols = ["player_gws_this_season", "player_minutes_this_season", "player_starts_this_season"]
            for col in exp_cols:
                if col in current_features_df.columns:
                    current_features_df[col] = 0.0
                    
            # Re-verify and set value form/season/change
            if "value_form" in current_features_df.columns and "total_points_last_5" in current_features_df.columns:
                current_features_df["value_form"] = (current_features_df["total_points_last_5"] / current_features_df["current_price"]).fillna(0.0)
            if "value_season" in current_features_df.columns:
                current_features_df["value_season"] = 0.0
            if "value_for_money" in current_features_df.columns:
                current_features_df["value_for_money"] = 0.0
            if "price_change_recent" in current_features_df.columns:
                current_features_df["price_change_recent"] = 0.0
            if "ownership_change_recent" in current_features_df.columns:
                current_features_df["ownership_change_recent"] = 0.0
            if "transfers_in_ratio" in current_features_df.columns:
                current_features_df["transfers_in_ratio"] = 0.0
            if "transfers_out_ratio" in current_features_df.columns:
                current_features_df["transfers_out_ratio"] = 0.0
            if "transfers_balance_ratio" in current_features_df.columns:
                current_features_df["transfers_balance_ratio"] = 0.0
                
            print(f"  Successfully aligned historical features for {len(current_features_df)} players!")
    
    # Compute current workload and fixture congestion features
    from src.features.current_workload import compute_workload_features
    print("\nComputing player workload and fixture congestion features...")
    workload_df = compute_workload_features(
        season=season,
        target_gw=gw,
        player_gw_df=curr_pgw,
        players_df=curr_players,
        fixtures_df=curr_fixtures,
        teams_df=curr_teams,
        bootstrap_raw=bootstrap
    )
    
    # Merge workload features into current_features_df
    # (these new columns are ignored by the existing LightGBM model but available for evaluation)
    current_features_df = pd.merge(
        current_features_df,
        workload_df,
        on=["season", "gw", "code", "name"],
        how="left"
    )
    
    # Verify duplicates on (season, gw, code)
    dups = current_features_df.duplicated(subset=["season", "gw", "code"]).sum()
    assert dups == 0, f"ERROR: Duplicates found in current features: {dups}"
    
    # Save current features parquet
    output_path = PROCESSED_DIR / "current_features.parquet"
    current_features_df.to_parquet(output_path, index=False)
    
    print(f"\nSuccessfully generated inference features for {season} GW {gw}!")
    print(f"  Rows count: {len(current_features_df):,}")
    print(f"  Columns count: {len(current_features_df.columns)}")
    print(f"  Saved exclusively to: {output_path}")
    print(f"  Verified: Historical features_df.parquet remains completely untouched.")
    return gw

def main():
    parser = argparse.ArgumentParser(description="FPL Current Weekly Data Ingestion")
    parser.add_argument("--season", type=str, required=True, help="Target season (e.g. 2024-25)")
    parser.add_argument("--gw", type=int, required=True, help="Target upcoming gameweek (1-38)")
    args = parser.parse_args()
    
    try:
        run_update_current_pipeline(args.season, args.gw)
    except Exception as e:
        print(f"\nERROR running in-season ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
