import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def load_json(path: Path) -> Any:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def compute_workload_features(
    season: str,
    target_gw: int,
    player_gw_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    bootstrap_raw: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute player workload and fixture congestion features.
    Only uses matches that occurred strictly before the target GW deadline.
    """
    # 1. Parse target gameweek deadline cutoff
    deadline_str = None
    if bootstrap_raw and "events" in bootstrap_raw:
        for event in bootstrap_raw["events"]:
            if event["id"] == target_gw:
                deadline_str = event.get("deadline_time")
                break
                
    if not deadline_str:
        # Fallback approximation (e.g. today's date)
        cutoff_date = pd.Timestamp.now(tz="UTC")
    else:
        cutoff_date = pd.to_datetime(deadline_str, utc=True)
        
    print(f"Target GW {target_gw} deadline cutoff: {cutoff_date}")
    
    # 2. Match FPL player codes to external matches using mapping table
    mapping_path = PROCESSED_DIR / "player_id_mapping.parquet"
    mapping_dict = {} # fpl_code -> fbref_id
    if mapping_path.exists():
        mapping_df = pd.read_parquet(mapping_path)
        for _, row in mapping_df.iterrows():
            mapping_dict[int(row["fpl_code"])] = str(row["fbref_id"])
            
    # Load external matches
    ext_matches_path = RAW_DIR / "matches" / "external_cup_matches.json"
    external_matches = load_json(ext_matches_path) or []
    
    # Convert FPL fixtures kickoff times to datetime
    fixtures_df = fixtures_df.copy()
    if "kickoff_time" in fixtures_df.columns:
        fixtures_df["kickoff_time_dt"] = pd.to_datetime(fixtures_df["kickoff_time"], utc=True)
    elif "id" in fixtures_df.columns:
        # Fallback mock times if kickoff_time is not in live compiled fixtures
        # (e.g. during test environments, schedule GWs chronologically 7 days apart)
        fixtures_df["kickoff_time_dt"] = pd.to_datetime("2024-08-10T12:00:00Z", utc=True) + pd.to_timedelta(fixtures_df["event"].fillna(1) * 7, unit="D")
    else:
        fixtures_df["kickoff_time_dt"] = pd.to_datetime("2024-08-10T12:00:00Z", utc=True)
        
    # Map kickoff_time and kickoff_time_dt to player_gw
    player_gw_df = player_gw_df.copy()
    player_gw_df = pd.merge(
        player_gw_df,
        fixtures_df[["id", "kickoff_time_dt"]].rename(columns={"id": "fixture"}),
        on="fixture",
        how="left"
    )
    # Ensure kickoff_time_dt is timezone-aware
    player_gw_df["kickoff_time_dt"] = pd.to_datetime(player_gw_df["kickoff_time_dt"], utc=True)
    
    # Filter PL match history strictly before deadline cutoff
    pl_history = player_gw_df[player_gw_df["kickoff_time_dt"] < cutoff_date].copy()
    
    # 3. Compile all match logs for every player
    player_workload_rows = []
    
    # Map player code to id and team
    for _, player in players_df.iterrows():
        p_id = player["player_id"]
        p_code = int(player["code"])
        p_team = player["team"]
        
        # A. Filter player's PL matches
        p_pl = pl_history[pl_history["player_id"] == p_id].copy()
        
        # B. Filter player's external cup matches
        p_ext_id = mapping_dict.get(p_code)
        p_ext = []
        if p_ext_id:
            for match in external_matches:
                if str(match.get("fbref_id")) == p_ext_id:
                    m_date = pd.to_datetime(match["date"], utc=True)
                    if m_date < cutoff_date:
                        p_ext.append({
                            "date": m_date,
                            "minutes": int(match.get("minutes", 0)),
                            "started": int(match.get("started", 0)),
                            "goals": int(match.get("goals", 0)),
                            "assists": int(match.get("assists", 0))
                        })
        p_ext_df = pd.DataFrame(p_ext)
        
        # C. Compute PL rolling features
        pl_min_14 = 0.0
        if not p_pl.empty:
            p_pl["days_ago"] = (cutoff_date - p_pl["kickoff_time_dt"]).dt.days
            pl_min_14 = float(p_pl[p_pl["days_ago"] < 14]["minutes"].sum())
            
        # D. Compute External rolling features
        ext_min_7 = 0.0
        ext_min_14 = 0.0
        ext_min_21 = 0.0
        ext_app_7 = 0
        ext_app_14 = 0
        ext_app_21 = 0
        ext_starts_14 = 0
        ext_goals_14 = 0
        ext_assists_14 = 0
        
        if not p_ext_df.empty:
            p_ext_df["days_ago"] = (cutoff_date - p_ext_df["date"]).dt.days
            
            ext_7 = p_ext_df[p_ext_df["days_ago"] < 7]
            ext_14 = p_ext_df[p_ext_df["days_ago"] < 14]
            ext_21 = p_ext_df[p_ext_df["days_ago"] < 21]
            
            ext_min_7 = float(ext_7["minutes"].sum())
            ext_min_14 = float(ext_14["minutes"].sum())
            ext_min_21 = float(ext_21["minutes"].sum())
            
            ext_app_7 = int((ext_7["minutes"] > 0).sum())
            ext_app_14 = int((ext_14["minutes"] > 0).sum())
            ext_app_21 = int((ext_21["minutes"] > 0).sum())
            
            ext_starts_14 = int(ext_14["started"].sum())
            ext_goals_14 = int(ext_14["goals"].sum())
            ext_assists_14 = int(ext_14["assists"].sum())
            
        # E. Total stats
        total_min_14 = pl_min_14 + ext_min_14
        total_comp_min_7 = (float(p_pl[(cutoff_date - p_pl["kickoff_time_dt"]).dt.days < 7]["minutes"].sum()) if not p_pl.empty else 0.0) + ext_min_7
        total_comp_min_14 = total_min_14
        total_comp_min_21 = (float(p_pl[(cutoff_date - p_pl["kickoff_time_dt"]).dt.days < 21]["minutes"].sum()) if not p_pl.empty else 0.0) + ext_min_21
        
        # F. Rest days
        # Find player last match date
        last_match_date = None
        
        pl_dates = p_pl["kickoff_time_dt"].tolist() if not p_pl.empty else []
        ext_dates = p_ext_df["date"].tolist() if not p_ext_df.empty else []
        all_match_dates = sorted(pl_dates + ext_dates)
        
        days_since_player_last_match = 99.0
        if all_match_dates:
            last_match_date = all_match_dates[-1]
            days_since_player_last_match = float((cutoff_date - last_match_date).total_seconds() / 86400.0)
            
        # G. Team Congestion & Rest
        # Filter all fixtures played/scheduled for this team
        team_fixtures = fixtures_df[
            ((fixtures_df["team_h"] == p_team) | (fixtures_df["team_a"] == p_team))
        ].copy()
        
        # Past fixtures
        team_past = team_fixtures[team_fixtures["kickoff_time_dt"] < cutoff_date].copy()
        team_past["days_ago"] = (cutoff_date - team_past["kickoff_time_dt"]).dt.days
        
        team_matches_7 = int(team_past[team_past["days_ago"] < 7].shape[0])
        team_matches_14 = int(team_past[team_past["days_ago"] < 14].shape[0])
        
        # Rest days since team last match
        days_since_team_last_match = 99.0
        if not team_past.empty:
            team_last_match = team_past.sort_values("kickoff_time_dt")["kickoff_time_dt"].iloc[-1]
            days_since_team_last_match = float((cutoff_date - team_last_match).total_seconds() / 86400.0)
            
        # Future fixtures (GW N kickoff and next GWs kickoff)
        team_future = team_fixtures[team_fixtures["kickoff_time_dt"] >= cutoff_date].copy()
        team_future["days_until"] = (team_future["kickoff_time_dt"] - cutoff_date).dt.days
        
        team_matches_next_7 = int(team_future[team_future["days_until"] <= 7].shape[0])
        
        days_until_next_match = 99.0
        if not team_future.empty:
            team_next_match = team_future.sort_values("kickoff_time_dt")["kickoff_time_dt"].iloc[0]
            days_until_next_match = float((team_next_match - cutoff_date).total_seconds() / 86400.0)
            
        # Congestion score: count of matches in last 14 + next 7
        fixture_congestion_score = float(team_matches_14 + team_matches_next_7)
        
        player_workload_rows.append({
            "season": season,
            "gw": target_gw,
            "code": p_code,
            "name": player["name"],
            # Player features
            "pl_minutes_last_14d": pl_min_14,
            "external_minutes_last_7d": ext_min_7,
            "external_minutes_last_14d": ext_min_14,
            "external_minutes_last_21d": ext_min_21,
            "total_minutes_last_14d": total_min_14,
            "external_appearances_last_7d": ext_app_7,
            "external_appearances_last_14d": ext_app_14,
            "external_appearances_last_21d": ext_app_21,
            "external_starts_last_14d": ext_starts_14,
            "external_goals_last_14d": ext_goals_14,
            "external_assists_last_14d": ext_assists_14,
            "total_competitive_minutes_last_7d": total_comp_min_7,
            "total_competitive_minutes_last_14d": total_comp_min_14,
            "total_competitive_minutes_last_21d": total_comp_min_21,
            "days_since_player_last_match": days_since_player_last_match,
            # Team features
            "team_matches_last_7d": team_matches_7,
            "team_matches_last_14d": team_matches_14,
            "team_matches_next_7d": team_matches_next_7,
            "days_since_team_last_match": days_since_team_last_match,
            "days_until_next_match": days_until_next_match,
            "fixture_congestion_score": fixture_congestion_score
        })
        
    workload_df = pd.DataFrame(player_workload_rows)
    
    # Save standalone workload features parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    workload_df.to_parquet(PROCESSED_DIR / "current_player_workload.parquet", index=False)
    
    return workload_df
