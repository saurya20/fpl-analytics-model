import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Any

def build_team_rolling_features(player_gw_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build point-in-time rolling team features.
    1. Resolve player's team and opponent team for all gameweek records.
    2. Extract unique matches from the players' perspective.
    3. Aggregate goals scored/conceded and clean sheets for each team and gameweek.
    4. Construct a complete grid of (season, gw, team_id) to avoid gaps.
    5. Calculate rolling team features and shift them by 1.
    """
    # Merge player_gw with players to get season_team_id
    players_sub = players_df[["player_id", "season", "team"]].rename(columns={"team": "season_team_id"})
    df = pd.merge(
        player_gw_df[["player_id", "season", "gw", "was_home", "opponent_team", "team_h_score", "team_a_score", "fixture"]],
        players_sub,
        on=["player_id", "season"],
        how="left"
    )
    
    # Fill in missing season_team_id (for older seasons where player_id changed or if missing)
    # We fallback: if was_home, player's team can be derived or we just drop na. Let's make sure we ffill
    df["season_team_id"] = df["season_team_id"].ffill().bfill()
    df["opponent_team_id"] = df["opponent_team"]
    
    # Handle NaN values in team scores (COVID season 2019-20 had some nulls)
    df["team_h_score"] = df["team_h_score"].fillna(0.0)
    df["team_a_score"] = df["team_a_score"].fillna(0.0)
    
    # Extract unique matches by grouping by (season, gw, fixture)
    # Since all players in a match see the same fixture, we can group by fixture
    matches = df.groupby(["season", "gw", "fixture"]).first().reset_index()
    
    # Create two perspectives for each match (home and away)
    home_p = matches.copy()
    home_p["team_id"] = home_p["season_team_id"]
    home_p["opponent_id"] = home_p["opponent_team_id"]
    home_p["goals_scored"] = home_p["team_h_score"]
    home_p["goals_conceded"] = home_p["team_a_score"]
    home_p["is_home"] = 1.0
    
    away_p = matches.copy()
    away_p["team_id"] = away_p["opponent_team_id"]
    away_p["opponent_id"] = away_p["season_team_id"]
    away_p["goals_scored"] = away_p["team_a_score"]
    away_p["goals_conceded"] = home_p["team_h_score"] # Note: use home_p's team_h_score to match row index
    away_p["is_home"] = 0.0
    
    # Concatenate and drop duplicates to ensure we have exactly one row per team-match
    team_matches = pd.concat([home_p, away_p], ignore_index=True)
    team_matches = team_matches[["season", "gw", "team_id", "opponent_id", "goals_scored", "goals_conceded", "is_home"]].drop_duplicates()
    
    # Calculate clean sheet
    team_matches["clean_sheet"] = (team_matches["goals_conceded"] == 0).astype(float)
    
    # Aggregate matches by team and gameweek
    # In a double gameweek, a team plays twice, so we sum goals/clean sheets
    team_gw_stats = team_matches.groupby(["season", "gw", "team_id"]).agg(
        goals_scored=("goals_scored", "sum"),
        goals_conceded=("goals_conceded", "sum"),
        clean_sheets=("clean_sheet", "sum"),
        num_matches=("goals_scored", "count")
    ).reset_index()
    
    # Construct complete grid of all seasons, gameweeks (1 to 38), and teams
    seasons = sorted(team_gw_stats["season"].unique())
    gws = list(range(1, 39))
    
    grid_rows = []
    for season in seasons:
        # Get unique teams that played in this season
        season_teams = sorted(team_gw_stats[team_gw_stats["season"] == season]["team_id"].unique())
        if not season_teams:
            season_teams = list(range(1, 21)) # Fallback
            
        for gw, team in itertools.product(gws, season_teams):
            grid_rows.append({"season": season, "gw": gw, "team_id": team})
            
    grid_df = pd.DataFrame(grid_rows)
    
    # Merge aggregated stats onto grid
    team_gw_full = pd.merge(grid_df, team_gw_stats, on=["season", "gw", "team_id"], how="left")
    team_gw_full = team_gw_full.fillna({
        "goals_scored": 0.0,
        "goals_conceded": 0.0,
        "clean_sheets": 0.0,
        "num_matches": 0.0
    })
    
    # Sort chronologically
    team_gw_full = team_gw_full.sort_values(["season", "team_id", "gw"]).reset_index(drop=True)
    
    # Helper to calculate rolling sum and shift by 1 within season-team groups
    def get_rolling_sum(col: str, window: int) -> pd.Series:
        return team_gw_full.groupby(["season", "team_id"])[col].rolling(window, min_periods=1).sum().reset_index(level=[0, 1], drop=True).shift(1)

    # Calculate team-level rolling features
    team_gw_full["team_goals_scored_last_3"] = get_rolling_sum("goals_scored", 3).fillna(0.0)
    team_gw_full["team_goals_scored_last_5"] = get_rolling_sum("goals_scored", 5).fillna(0.0)
    team_gw_full["team_goals_conceded_last_3"] = get_rolling_sum("goals_conceded", 3).fillna(0.0)
    team_gw_full["team_goals_conceded_last_5"] = get_rolling_sum("goals_conceded", 5).fillna(0.0)
    team_gw_full["team_clean_sheets_last_5"] = get_rolling_sum("clean_sheets", 5).fillna(0.0)
    
    # Calculate point-in-time team overall stats
    team_gw_full["team_goals_scored_season"] = team_gw_full.groupby(["season", "team_id"])["goals_scored"].cumsum().shift(1).fillna(0.0)
    team_gw_full["team_goals_conceded_season"] = team_gw_full.groupby(["season", "team_id"])["goals_conceded"].cumsum().shift(1).fillna(0.0)
    team_gw_full["team_matches_played_season"] = team_gw_full.groupby(["season", "team_id"])["num_matches"].cumsum().shift(1).fillna(0.0)
    
    # Averages
    team_gw_full["team_goals_scored_per_match_season"] = np.where(
        team_gw_full["team_matches_played_season"] > 0,
        team_gw_full["team_goals_scored_season"] / team_gw_full["team_matches_played_season"],
        0.0
    )
    team_gw_full["team_goals_conceded_per_match_season"] = np.where(
        team_gw_full["team_matches_played_season"] > 0,
        team_gw_full["team_goals_conceded_season"] / team_gw_full["team_matches_played_season"],
        0.0
    )
    
    # Select only features and keys
    features_cols = [
        "season", "gw", "team_id",
        "team_goals_scored_last_3", "team_goals_scored_last_5",
        "team_goals_conceded_last_3", "team_goals_conceded_last_5",
        "team_clean_sheets_last_5", "team_goals_scored_per_match_season",
        "team_goals_conceded_per_match_season"
    ]
    
    return team_gw_full[features_cols]
