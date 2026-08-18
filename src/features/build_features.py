import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.player_features import build_player_grid, add_player_rolling_features
from src.features.team_features import build_team_rolling_features
from src.features.fixture_features import build_fixture_features

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

def run_feature_engineering_pipeline():
    """
    Run the complete feature engineering pipeline:
    1. Load processed parquet files.
    2. Generate player-level grid and rolling features.
    3. Generate team-level point-in-time rolling features.
    4. Generate fixture-level features (DGW aggregates and opponent form).
    5. Merge and define prediction targets.
    6. Run safety checks (leakage, duplicates, shape).
    7. Save final dataset to parquet.
    """
    print("=" * 60)
    print("FPL FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    print("\n[1/5] Loading processed datasets...")
    player_gw = pd.read_parquet(PROCESSED_DATA / "player_gw.parquet")
    players = pd.read_parquet(PROCESSED_DATA / "players.parquet")
    fixtures = pd.read_parquet(PROCESSED_DATA / "fixtures.parquet")
    teams = pd.read_parquet(PROCESSED_DATA / "teams.parquet")
    
    print(f"  player_gw rows: {len(player_gw):,}")
    print(f"  players rows: {len(players):,}")
    print(f"  fixtures rows: {len(fixtures):,}")
    print(f"  teams rows: {len(teams):,}")
    
    print("\n[2/5] Building player-GW grid and rolling player features...")
    player_grid = build_player_grid(player_gw, players)
    player_features = add_player_rolling_features(player_grid)
    print(f"  Player features shape: {player_features.shape}")
    
    print("\n[3/5] Building team rolling form features...")
    team_features = build_team_rolling_features(player_gw, players)
    print(f"  Team features shape: {team_features.shape}")
    
    print("\n[4/5] Building fixture features & opponent rolling form...")
    fixture_features = build_fixture_features(player_gw, players, fixtures, teams, team_features)
    print(f"  Fixture features shape: {fixture_features.shape}")
    
    print("\n[5/5] Merging player, team, and fixture features...")
    # Merge player features with fixture features on season, gw, code
    features_df = pd.merge(
        player_features,
        fixture_features,
        on=["season", "gw", "code"],
        how="left"
    )
    
    # Handle Blank Gameweeks (BGW)
    # Players who had a BGW will have no rows in fixture_features, resulting in NaNs
    features_df["num_fixtures"] = features_df["num_fixtures"].fillna(0.0)
    features_df["is_double_gw"] = features_df["is_double_gw"].fillna(0.0)
    features_df["was_home_mean"] = features_df["was_home_mean"].fillna(0.0)
    features_df["fixture_difficulty_mean"] = features_df["fixture_difficulty_mean"].fillna(3.0)
    features_df["fixture_1_difficulty"] = features_df["fixture_1_difficulty"].fillna(3.0)
    
    # Fill rolling opponent stats with 0.0 for blank gameweeks
    opp_fill_zero = [
        "opponent_team_scored_last_3_sum", "opponent_team_scored_last_5_sum",
        "opponent_team_conceded_last_3_sum", "opponent_team_conceded_last_5_sum",
        "opponent_team_clean_sheets_last_5_sum", "opponent_static_strength_mean",
        "opponent_static_defence_mean", "opponent_static_attack_mean"
    ]
    for col in opp_fill_zero:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0.0)
            
    # Define targets
    features_df["target_points"] = features_df["total_points"]
    features_df["target_minutes"] = features_df["minutes"]
    features_df["target_60_plus_minutes"] = (features_df["minutes"] >= 60.0).astype(float)
    
    # Identify and drop the raw performance columns of the prediction gameweek (GW N)
    # to guarantee there is zero data leakage into the features.
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
    
    print("\nData Quality & Validation Checks:")
    print("-" * 40)
    print(f"Final Features Dataframe Shape: {features_df.shape}")
    
    # Check for duplicates on (season, gw, code)
    dups = features_df.duplicated(subset=["season", "gw", "code"]).sum()
    print(f"Duplicate player-GW rows: {dups}")
    assert dups == 0, "ERROR: Duplicates found on (season, gw, code)!"
    
    # Check that targets have no NaNs
    print(f"Missing target_points: {features_df['target_points'].isna().sum()}")
    print(f"Missing target_minutes: {features_df['target_minutes'].isna().sum()}")
    print(f"Missing target_60_plus_minutes: {features_df['target_60_plus_minutes'].isna().sum()}")
    
    # Check points_per_90 missing values (expected NaNs in pre-2022-23 or low minutes)
    for col in ["points_per_90_last_5", "xG_last_5", "xA_last_5"]:
        if col in features_df.columns:
            missing = features_df[col].isna().sum()
            print(f"Missing {col}: {missing} ({missing/len(features_df):.2%})")
            
    # Save the dataset
    features_df.to_parquet(PROCESSED_DATA / "features_df.parquet", index=False)
    print(f"\nSaved final features dataset to: {PROCESSED_DATA / 'features_df.parquet'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_feature_engineering_pipeline()
