import pandas as pd
import numpy as np
from typing import Dict, List, Any

def build_fixture_features(
    player_gw_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    team_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build upcoming fixture features for the prediction unit (player x GW).
    1. Resolve team and opponent for every player-fixture row.
    2. Join with the point-in-time team and opponent rolling stats.
    3. Look up official fixture difficulty where available, otherwise use default.
    4. Group by (season, gw, code) to aggregate double gameweek fixture attributes.
    """
    # Merge player_gw with players to get code and season_team_id
    players_sub = players_df[["player_id", "season", "code", "team"]].rename(columns={"team": "season_team_id"})
    df = pd.merge(
        player_gw_df[["player_id", "season", "gw", "was_home", "opponent_team", "fixture"]],
        players_sub,
        on=["player_id", "season"],
        how="left"
    )
    
    # Fill in missing season_team_id (ffill/bfill)
    df["season_team_id"] = df["season_team_id"].ffill().bfill()
    df["opponent_team_id"] = df["opponent_team"]
    df["was_home_bool"] = df["was_home"].astype(bool)
    
    # 1. Join with official fixtures if available to get difficulty
    if not fixtures_df.empty:
        fixtures_sub = fixtures_df[["id", "season", "team_h_difficulty", "team_a_difficulty"]].rename(
            columns={"id": "fixture"}
        )
        df = pd.merge(df, fixtures_sub, on=["fixture", "season"], how="left")
        
        # Determine difficulty based on was_home
        df["fixture_difficulty"] = np.where(
            df["was_home_bool"],
            df["team_h_difficulty"],
            df["team_a_difficulty"]
        )
    else:
        df["fixture_difficulty"] = np.nan
        
    # Default difficulty to 3.0 if missing (pre-2018-19 or missing values)
    df["fixture_difficulty"] = df["fixture_difficulty"].fillna(3.0)
    
    # 2. Join with team rolling features for player's team
    # Rename team features to player_team_*
    team_feats_player = team_features_df.copy().rename(columns={
        "team_id": "season_team_id",
        "team_goals_scored_last_3": "player_team_scored_last_3",
        "team_goals_scored_last_5": "player_team_scored_last_5",
        "team_goals_conceded_last_3": "player_team_conceded_last_3",
        "team_goals_conceded_last_5": "player_team_conceded_last_5",
        "team_clean_sheets_last_5": "player_team_clean_sheets_last_5",
        "team_goals_scored_per_match_season": "player_team_scored_season",
        "team_goals_conceded_per_match_season": "player_team_conceded_season",
    })
    df = pd.merge(df, team_feats_player, on=["season", "gw", "season_team_id"], how="left")
    
    # 3. Join with team rolling features for opponent's team
    # Rename team features to opponent_team_*
    team_feats_opp = team_features_df.copy().rename(columns={
        "team_id": "opponent_team_id",
        "team_goals_scored_last_3": "opponent_team_scored_last_3",
        "team_goals_scored_last_5": "opponent_team_scored_last_5",
        "team_goals_conceded_last_3": "opponent_team_conceded_last_3",
        "team_goals_conceded_last_5": "opponent_team_conceded_last_5",
        "team_clean_sheets_last_5": "opponent_team_clean_sheets_last_5",
        "team_goals_scored_per_match_season": "opponent_team_scored_season",
        "team_goals_conceded_per_match_season": "opponent_team_conceded_season",
    })
    df = pd.merge(df, team_feats_opp, on=["season", "gw", "opponent_team_id"], how="left")
    
    # 4. Optional: static team overall strength from teams table (where available, 2019-20 onwards)
    if not teams_df.empty:
        teams_sub = teams_df[["id", "season", "strength", "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]].rename(
            columns={"id": "opponent_team_id"}
        )
        df = pd.merge(df, teams_sub, on=["opponent_team_id", "season"], how="left")
        
        # Opponent strengths depending on player's was_home
        # If player is home, opponent is away (strength_defence_away, strength_attack_away)
        # If player is away, opponent is home (strength_defence_home, strength_attack_home)
        df["opponent_static_strength"] = df["strength"]
        df["opponent_static_defence"] = np.where(df["was_home_bool"], df["strength_defence_away"], df["strength_defence_home"])
        df["opponent_static_attack"] = np.where(df["was_home_bool"], df["strength_attack_away"], df["strength_attack_home"])
        df = df.drop(columns=["strength", "strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"])
    else:
        df["opponent_static_strength"] = np.nan
        df["opponent_static_defence"] = np.nan
        df["opponent_static_attack"] = np.nan
        
    # Fill static strengths with defaults (3 = medium strength, etc.)
    # Replace 0 (unpopulated API values) or NaN with defaults
    df["opponent_static_strength"] = df["opponent_static_strength"].replace(0, np.nan).fillna(3.0)
    df["opponent_static_defence"] = df["opponent_static_defence"].replace(0, np.nan).fillna(1000.0)
    df["opponent_static_attack"] = df["opponent_static_attack"].replace(0, np.nan).fillna(1000.0)
    
    # 5. Group by (season, gw, code) to aggregate double gameweeks
    # Convert was_home to float for mean was_home
    df["was_home_val"] = df["was_home_bool"].astype(float)
    
    # Aggregate DGW fixture attributes
    fixture_features = df.groupby(["season", "gw", "code"]).agg(
        num_fixtures=("fixture", "count"),
        was_home_mean=("was_home_val", "mean"),
        fixture_difficulty_mean=("fixture_difficulty", "mean"),
        fixture_difficulty_min=("fixture_difficulty", "min"),
        fixture_difficulty_max=("fixture_difficulty", "max"),
        
        # Sum opponent rolling stats (toughness sum)
        opponent_team_scored_last_3_sum=("opponent_team_scored_last_3", "sum"),
        opponent_team_scored_last_5_sum=("opponent_team_scored_last_5", "sum"),
        opponent_team_conceded_last_3_sum=("opponent_team_conceded_last_3", "sum"),
        opponent_team_conceded_last_5_sum=("opponent_team_conceded_last_5", "sum"),
        opponent_team_clean_sheets_last_5_sum=("opponent_team_clean_sheets_last_5", "sum"),
        
        # Player's team rolling stats (constant across DGW unless one home one away, so take mean)
        player_team_scored_last_3_mean=("player_team_scored_last_3", "mean"),
        player_team_scored_last_5_mean=("player_team_scored_last_5", "mean"),
        player_team_conceded_last_3_mean=("player_team_conceded_last_3", "mean"),
        player_team_conceded_last_5_mean=("player_team_conceded_last_5", "mean"),
        player_team_clean_sheets_last_5_mean=("player_team_clean_sheets_last_5", "mean"),
        
        # Opponent static strength
        opponent_static_strength_mean=("opponent_static_strength", "mean"),
        opponent_static_defence_mean=("opponent_static_defence", "mean"),
        opponent_static_attack_mean=("opponent_static_attack", "mean")
    ).reset_index()
    
    # Derive extra DGW attributes
    fixture_features["is_double_gw"] = (fixture_features["num_fixtures"] >= 2).astype(float)
    
    # Handle the second match difficulties specifically
    # For a normal GW, fixture_difficulty_2 will be filled with a neutral default or NaN.
    # To be useful for tree models, we can extract fixture_1_difficulty and fixture_2_difficulty
    # Let's do that by finding the first and second fixture difficulties in each group
    sorted_df = df.sort_values(["season", "gw", "code", "fixture"])
    
    sorted_df["fixture_rank"] = sorted_df.groupby(["season", "gw", "code"]).cumcount() + 1
    
    # Pivot difficulties
    difficulty_pivot = sorted_df.pivot(
        index=["season", "gw", "code"],
        columns="fixture_rank",
        values="fixture_difficulty"
    ).reset_index()
    
    # Rename columns
    difficulty_pivot = difficulty_pivot.rename(columns={1: "fixture_1_difficulty", 2: "fixture_2_difficulty"})
    # Keep only 1 and 2
    difficulty_cols = ["season", "gw", "code", "fixture_1_difficulty"]
    if "fixture_2_difficulty" in difficulty_pivot.columns:
        difficulty_cols.append("fixture_2_difficulty")
    else:
        difficulty_pivot["fixture_2_difficulty"] = np.nan
        difficulty_cols.append("fixture_2_difficulty")
        
    difficulty_pivot = difficulty_pivot[difficulty_cols]
    
    # Merge pivots
    fixture_features = pd.merge(fixture_features, difficulty_pivot, on=["season", "gw", "code"], how="left")
    
    # Fill second fixture difficulty with NaN so models understand there is no second fixture
    # (Do not impute with 0 because that would imply a super easy match)
    
    return fixture_features
