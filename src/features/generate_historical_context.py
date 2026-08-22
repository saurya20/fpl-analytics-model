import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    print("=" * 60)
    print("HISTORICAL FOOTBALL CONTEXT FEATURE GENERATION")
    print("=" * 60)
    
    # 1. Load Parquets
    print("Loading historical datasets...")
    features_path = PROCESSED_DIR / "features_df.parquet"
    player_gw_path = PROCESSED_DIR / "player_gw.parquet"
    fixtures_path = PROCESSED_DIR / "fixtures.parquet"
    players_path = PROCESSED_DIR / "players.parquet"
    
    for path in [features_path, player_gw_path, fixtures_path, players_path]:
        if not path.exists():
            print(f"Error: Required file missing at {path}")
            sys.exit(1)
            
    features_df = pd.read_parquet(features_path)
    player_gw_df = pd.read_parquet(player_gw_path)
    fixtures_df = pd.read_parquet(fixtures_path)
    players_df = pd.read_parquet(players_path)
    
    print(f"Loaded features_df: {features_df.shape}")
    print(f"Loaded player_gw_df: {player_gw_df.shape}")
    print(f"Loaded fixtures_df: {fixtures_df.shape}")
    
    # 2. Parse Datetimes
    print("\nParsing dates and deadlines...")
    # Convert kickoff_time in player_gw
    player_gw_df["kickoff_time_dt"] = pd.to_datetime(player_gw_df["kickoff_time"], utc=True)
    
    # Group fixtures to get deadline_time for each (season, gw)
    fixtures_df["deadline_time_dt"] = pd.to_datetime(fixtures_df["deadline_time"], utc=True)
    fixtures_df["kickoff_time_dt"] = pd.to_datetime(fixtures_df["kickoff_time"], utc=True)
    
    # Fallback to kickoff_time - 1 hour if deadline_time is missing
    fixtures_df["deadline_time_dt"] = fixtures_df["deadline_time_dt"].fillna(
        fixtures_df["kickoff_time_dt"] - pd.to_timedelta(1, unit="h")
    )
    
    # Get GW deadlines
    deadlines = fixtures_df.groupby(["season", "event"])["deadline_time_dt"].first().reset_index()
    deadlines = deadlines.rename(columns={"event": "gw", "deadline_time_dt": "deadline_time"})
    
    print(f"Compiled deadlines table: {deadlines.shape[0]} season-GW events.")
    
    # Create player mapping (player_id -> code) from players metadata
    player_code_map = players_df[["player_id", "code", "season"]].drop_duplicates()
    
    # 2.5 Load Player Mapping and External Matches
    print("Loading player mapping and external matches...")
    mapping_path = PROCESSED_DIR / "player_id_mapping.parquet"
    mapping_dict = {}
    if mapping_path.exists():
        mapping_df = pd.read_parquet(mapping_path)
        for _, row in mapping_df.iterrows():
            mapping_dict[int(row["fpl_code"])] = str(row["fbref_id"])
            
    import json
    ext_matches_path = PROJECT_ROOT / "data" / "raw" / "matches" / "external_cup_matches.json"
    external_matches = []
    if ext_matches_path.exists():
        with open(ext_matches_path, "r") as f:
            external_matches = json.load(f) or []
            
    # Map external matches to FPL codes
    ext_match_rows = []
    for m in external_matches:
        fb_id = str(m.get("fbref_id"))
        for fpl_c, ext_id in mapping_dict.items():
            if ext_id == fb_id:
                ext_match_rows.append({
                    "code": fpl_c,
                    "kickoff_time_dt": pd.to_datetime(m["date"], utc=True),
                    "minutes": float(m.get("minutes", 0)),
                    "starts": float(m.get("started", 0)),
                    "goals_scored": float(m.get("goals", 0)),
                    "assists": float(m.get("assists", 0)),
                    "is_external": True
                })
    if ext_match_rows:
        ext_matches_df = pd.DataFrame(ext_match_rows)
    else:
        ext_matches_df = pd.DataFrame(columns=["code", "kickoff_time_dt", "minutes", "starts", "goals_scored", "assists", "is_external"])

    # Merge deadline_time into features_df
    features_df = pd.merge(features_df, deadlines, on=["season", "gw"], how="left")
    
    # Fill missing deadlines with today's date if any (fallback)
    features_df["deadline_time"] = pd.to_datetime(features_df["deadline_time"], utc=True).fillna(pd.Timestamp.now(tz="UTC"))
    
    # 3. Compute Features Season-by-Season (Vectorized Join)
    print("\nComputing rolling workload and fixture congestion features season-by-season...")
    
    seasons = sorted(features_df["season"].unique())
    all_context_rows = []
    
    for season in seasons:
        t0 = pd.Timestamp.now()
        features_sub = features_df[features_df["season"] == season].copy()
        player_gw_sub = player_gw_df[player_gw_df["season"] == season].copy()
        fixtures_sub = fixtures_df[fixtures_df["season"] == season].copy()
        
        if features_sub.empty or player_gw_sub.empty or fixtures_sub.empty:
            print(f"  Skipping season {season} (empty subset)")
            continue
            
        # Merge player code to player_gw_sub if not already present
        if "code" not in player_gw_sub.columns:
            player_gw_sub = pd.merge(
                player_gw_sub,
                player_code_map[player_code_map["season"] == season][["player_id", "code"]],
                on="player_id",
                how="left"
            )
        
        # Prepare PL matches log
        pl_logs = player_gw_sub[["code", "kickoff_time_dt", "minutes", "goals_scored", "assists", "starts"]].copy()
        pl_logs["is_external"] = False
        
        # Prepare External matches log (filtered to the player codes in this season for speed)
        season_codes = set(features_sub["code"].unique())
        ext_logs = ext_matches_df[ext_matches_df["code"].isin(season_codes)].copy()
        
        # Combine
        combined_logs = pd.concat([pl_logs, ext_logs], ignore_index=True)
        combined_logs["kickoff_time_dt"] = pd.to_datetime(combined_logs["kickoff_time_dt"], utc=True)
        
        # A. PLAYER WORKLOAD FEATURES
        # Join matches to features player-GW rows on code
        p_matches_joined = pd.merge(
            features_sub[["gw", "code", "deadline_time"]].drop_duplicates(),
            combined_logs[["code", "kickoff_time_dt", "minutes", "goals_scored", "assists", "starts", "is_external"]],
            on="code",
            how="left"
        )
        
        # Filter past matches strictly before cutoff
        p_matches_joined = p_matches_joined[p_matches_joined["kickoff_time_dt"] < p_matches_joined["deadline_time"]].copy()
        
        # Compute days difference
        p_matches_joined["days_diff"] = (p_matches_joined["deadline_time"] - p_matches_joined["kickoff_time_dt"]).dt.total_seconds() / 86400.0
        
        # Split past matches into PL and External
        p_pl = p_matches_joined[p_matches_joined["is_external"] == False]
        p_ext = p_matches_joined[p_matches_joined["is_external"] == True]
        
        # Group by (gw, code) to aggregate rolling PL workload
        p_pl_7 = p_pl[p_pl["days_diff"] < 7]
        p_pl_14 = p_pl[p_pl["days_diff"] < 14]
        p_pl_21 = p_pl[p_pl["days_diff"] < 21]
        
        agg_pl_7 = p_pl_7.groupby(["gw", "code"])["minutes"].sum().reset_index().rename(columns={"minutes": "pl_min_7"})
        agg_pl_14 = p_pl_14.groupby(["gw", "code"])["minutes"].sum().reset_index().rename(columns={"minutes": "pl_min_14"})
        agg_pl_21 = p_pl_21.groupby(["gw", "code"])["minutes"].sum().reset_index().rename(columns={"minutes": "pl_min_21"})
        
        # Group by (gw, code) to aggregate rolling External workload
        p_ext_7 = p_ext[p_ext["days_diff"] < 7]
        p_ext_14 = p_ext[p_ext["days_diff"] < 14]
        p_ext_21 = p_ext[p_ext["days_diff"] < 21]
        
        agg_ext_7 = p_ext_7.groupby(["gw", "code"]).agg(
            ext_min_7=("minutes", "sum"),
            ext_app_7=("minutes", lambda x: (x > 0).sum())
        ).reset_index()
        
        agg_ext_14 = p_ext_14.groupby(["gw", "code"]).agg(
            ext_min_14=("minutes", "sum"),
            ext_app_14=("minutes", lambda x: (x > 0).sum()),
            ext_starts_14=("starts", "sum"),
            ext_goals_14=("goals_scored", "sum"),
            ext_assists_14=("assists", "sum")
        ).reset_index()
        
        agg_ext_21 = p_ext_21.groupby(["gw", "code"]).agg(
            ext_min_21=("minutes", "sum"),
            ext_app_21=("minutes", lambda x: (x > 0).sum())
        ).reset_index()
        
        # Days since player last match (PL or External)
        agg_last_match = p_matches_joined.groupby(["gw", "code"])["days_diff"].min().reset_index().rename(columns={"days_diff": "days_since_player_last_match"})
        
        # Merge player workloads
        workload_sub = features_sub[["gw", "code"]].drop_duplicates().copy()
        workload_sub = pd.merge(workload_sub, agg_pl_7, on=["gw", "code"], how="left")
        workload_sub = pd.merge(workload_sub, agg_pl_14, on=["gw", "code"], how="left")
        workload_sub = pd.merge(workload_sub, agg_pl_21, on=["gw", "code"], how="left")
        
        workload_sub = pd.merge(workload_sub, agg_ext_7, on=["gw", "code"], how="left")
        workload_sub = pd.merge(workload_sub, agg_ext_14, on=["gw", "code"], how="left")
        workload_sub = pd.merge(workload_sub, agg_ext_21, on=["gw", "code"], how="left")
        
        workload_sub = pd.merge(workload_sub, agg_last_match, on=["gw", "code"], how="left")
        
        # Fill workload NaNs
        workload_sub["pl_min_7"] = workload_sub["pl_min_7"].fillna(0.0)
        workload_sub["pl_min_14"] = workload_sub["pl_min_14"].fillna(0.0)
        workload_sub["pl_min_21"] = workload_sub["pl_min_21"].fillna(0.0)
        
        workload_sub["ext_min_7"] = workload_sub["ext_min_7"].fillna(0.0)
        workload_sub["ext_min_14"] = workload_sub["ext_min_14"].fillna(0.0)
        workload_sub["ext_min_21"] = workload_sub["ext_min_21"].fillna(0.0)
        
        workload_sub["ext_app_7"] = workload_sub["ext_app_7"].fillna(0).astype(int)
        workload_sub["ext_app_14"] = workload_sub["ext_app_14"].fillna(0).astype(int)
        workload_sub["ext_app_21"] = workload_sub["ext_app_21"].fillna(0).astype(int)
        
        workload_sub["ext_starts_14"] = workload_sub["ext_starts_14"].fillna(0).astype(int)
        workload_sub["ext_goals_14"] = workload_sub["ext_goals_14"].fillna(0).astype(int)
        workload_sub["ext_assists_14"] = workload_sub["ext_assists_14"].fillna(0).astype(int)
        
        workload_sub["days_since_player_last_match"] = workload_sub["days_since_player_last_match"].fillna(99.0)
        
        # B. TEAM CONGESTION FEATURES
        # Construct team matches database
        home_m = fixtures_sub[["team_h", "kickoff_time_dt"]].rename(columns={"team_h": "team"})
        away_m = fixtures_sub[["team_a", "kickoff_time_dt"]].rename(columns={"team_a": "team"})
        team_matches = pd.concat([home_m, away_m], ignore_index=True)
        
        # Join team matches to features player-GW rows on season_team_id
        t_matches_joined = pd.merge(
            features_sub[["gw", "season_team_id", "deadline_time"]].drop_duplicates(),
            team_matches,
            left_on="season_team_id",
            right_on="team",
            how="left"
        )
        
        # Compute days difference
        t_matches_joined["days_diff"] = (t_matches_joined["deadline_time"] - t_matches_joined["kickoff_time_dt"]).dt.total_seconds() / 86400.0
        
        # Split past and future fixtures
        t_past = t_matches_joined[t_matches_joined["days_diff"] > 0].copy()
        t_future = t_matches_joined[t_matches_joined["days_diff"] <= 0].copy()
        
        # Agg past matches
        agg_past = t_past.groupby(["gw", "season_team_id"]).agg(
            team_matches_7=("days_diff", lambda x: int((x < 7).sum())),
            team_matches_14=("days_diff", lambda x: int((x < 14).sum())),
            days_since_team_last_match=("days_diff", "min")
        ).reset_index()
        
        # Agg future matches
        agg_future = t_future.groupby(["gw", "season_team_id"]).agg(
            team_matches_next_7=("days_diff", lambda x: int((x >= -7).sum())),
            days_until_next_match=("days_diff", lambda x: float(-x.max()) if not x.empty else 99.0)
        ).reset_index()
        
        # Align features workload and team congestion
        season_df = features_sub[["season", "gw", "code", "name", "season_team_id"]].copy()
        season_df = pd.merge(season_df, workload_sub, on=["gw", "code"], how="left")
        season_df = pd.merge(season_df, agg_past, on=["gw", "season_team_id"], how="left")
        season_df = pd.merge(season_df, agg_future, on=["gw", "season_team_id"], how="left")
        
        # Fill congestion NaNs
        season_df["team_matches_7"] = season_df["team_matches_7"].fillna(0).astype(int)
        season_df["team_matches_14"] = season_df["team_matches_14"].fillna(0).astype(int)
        season_df["team_matches_next_7"] = season_df["team_matches_next_7"].fillna(0).astype(int)
        season_df["days_since_team_last_match"] = season_df["days_since_team_last_match"].fillna(99.0)
        season_df["days_until_next_match"] = season_df["days_until_next_match"].fillna(99.0)
        
        # Final formatting and alignment to features_sub columns
        season_df["pl_minutes_last_14d"] = season_df["pl_min_14"]
        season_df["external_minutes_last_7d"] = season_df["ext_min_7"]
        season_df["external_minutes_last_14d"] = season_df["ext_min_14"]
        season_df["external_minutes_last_21d"] = season_df["ext_min_21"]
        season_df["external_appearances_last_7d"] = season_df["ext_app_7"]
        season_df["external_appearances_last_14d"] = season_df["ext_app_14"]
        season_df["external_appearances_last_21d"] = season_df["ext_app_21"]
        season_df["external_starts_last_14d"] = season_df["ext_starts_14"]
        season_df["external_goals_last_14d"] = season_df["ext_goals_14"]
        season_df["external_assists_last_14d"] = season_df["ext_assists_14"]
        
        # Total competitive (equals PL + External)
        season_df["total_minutes_last_14d"] = season_df["pl_min_14"] + season_df["ext_min_14"]
        season_df["total_competitive_minutes_last_7d"] = season_df["pl_min_7"] + season_df["ext_min_7"]
        season_df["total_competitive_minutes_last_14d"] = season_df["pl_min_14"] + season_df["ext_min_14"]
        season_df["total_competitive_minutes_last_21d"] = season_df["pl_min_21"] + season_df["ext_min_21"]
        
        # Team congestion features
        season_df["team_matches_last_7d"] = season_df["team_matches_7"]
        season_df["team_matches_last_14d"] = season_df["team_matches_14"]
        season_df["team_matches_next_7d"] = season_df["team_matches_next_7"]
        
        # Score = last 14 matches + next 7 matches
        season_df["fixture_congestion_score"] = season_df["team_matches_14"] + season_df["team_matches_next_7"]
        
        # Keep only required columns
        out_cols = [
            "season", "gw", "code", "name",
            "pl_minutes_last_14d",
            "external_minutes_last_7d", "external_minutes_last_14d", "external_minutes_last_21d",
            "total_minutes_last_14d",
            "external_appearances_last_7d", "external_appearances_last_14d", "external_appearances_last_21d",
            "external_starts_last_14d", "external_goals_last_14d", "external_assists_last_14d",
            "total_competitive_minutes_last_7d", "total_competitive_minutes_last_14d", "total_competitive_minutes_last_21d",
            "days_since_player_last_match",
            "team_matches_last_7d", "team_matches_last_14d", "team_matches_next_7d",
            "days_since_team_last_match", "days_until_next_match", "fixture_congestion_score"
        ]
        season_df = season_df[out_cols]
        
        all_context_rows.append(season_df)
        print(f"  Processed {season} in {pd.Timestamp.now() - t0} | Rows: {season_df.shape[0]:,}")
        
    # 4. Save Features
    context_df = pd.concat(all_context_rows, ignore_index=True)
    out_path = PROCESSED_DIR / "historical_context_features.parquet"
    context_df.to_parquet(out_path, index=False)
    
    print("\nFeatures compiled successfully!")
    print(f"  Shape: {context_df.shape}")
    print(f"  Output Parquet saved to: {out_path}")
    print(f"  Real Premier League coverage: {len(seasons)} seasons ({seasons[0]} to {seasons[-1]})")
    print(f"  External cup match coverage: {len(external_matches)} mapped external records.")

if __name__ == "__main__":
    main()

