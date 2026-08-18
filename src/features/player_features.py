import pandas as pd
import numpy as np
from typing import List, Dict, Any

def build_player_grid(player_gw_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Merge player_gw with players on player_id and season to get player 'code' and position.
    2. Aggregate fixture-level records to player-GW level (summing points, minutes, etc.).
    3. Construct a complete chronological grid for each active player starting from their first 
       appearance in FPL (first_gw) to gameweek 38, ensuring there are no gaps for rolling windows.
    """
    # Merge to get player code and position
    players_sub = players_df[["player_id", "season", "code", "element_type", "team"]].rename(
        columns={"element_type": "element_type_raw", "team": "season_team_id"}
    )
    
    df = pd.merge(
        player_gw_df,
        players_sub,
        on=["player_id", "season"],
        how="left"
    )
    
    # Import standardize_position
    from src.utils.helpers import standardize_position
    df["position"] = df["element_type_raw"].apply(lambda x: standardize_position(x) if pd.notna(x) else "MID")
    
    # Fill in derived team and opponent if missing
    # We resolve team ID: use fixture if available, otherwise fallback to players table
    # Standardize was_home
    df["was_home_bool"] = df["was_home"].astype(bool)
    
    # In player_gw, opponent_team is opponent_team_id
    df["opponent_team_id"] = df["opponent_team"]
    
    # Clean up position
    df["position"] = df["position"].fillna("MID")
    
    # Let's aggregate double gameweeks to player-GW level
    # We group by (season, gw, code)
    # Define aggregation logic
    # To preserve NaNs for expected stats in older seasons, we use a custom lambda that returns NaN if all values are NaN.
    # Note: lambda x: x.sum(min_count=1) handles this perfectly.
    
    expected_cols = ["expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]
    
    agg_dict = {
        # Sum performance statistics
        "total_points": "sum",
        "minutes": "sum",
        "goals_scored": "sum",
        "assists": "sum",
        "clean_sheets": "sum",
        "goals_conceded": "sum",
        "own_goals": "sum",
        "penalties_saved": "sum",
        "penalties_missed": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "saves": "sum",
        "bonus": "sum",
        "bps": "sum",
        "starts": "sum",
        "creativity": "sum",
        "threat": "sum",
        "influence": "sum",
        "ict_index": "sum",
        "defensive_contribution": "sum",
        "tackles": "sum",
        "clearances_blocks_interceptions": "sum",
        "recoveries": "sum",
        
        # Player-GW level constants (take first)
        "value": "first",
        "selected": "first",
        "transfers_balance": "first",
        "transfers_in": "first",
        "transfers_out": "first",
        "player_id": "first",
        "name": "first",
        "position": "first",
        "season_team_id": "first",
    }
    
    # Add expected stats columns with lambda that uses min_count=1
    for col in expected_cols:
        agg_dict[col] = lambda x: x.sum(min_count=1)
        
    # Only aggregate columns that exist
    actual_agg_dict = {}
    for k, v in agg_dict.items():
        if k in df.columns:
            actual_agg_dict[k] = v
            
    player_gw_agg = df.groupby(["season", "gw", "code"]).agg(actual_agg_dict).reset_index()
    
    # Determine the first gameweek each player was active in each season
    first_gws = player_gw_agg.groupby(["season", "code"])["gw"].min().reset_index().rename(columns={"gw": "first_gw"})
    
    # Construct complete grid
    grid_rows = []
    for _, row in first_gws.iterrows():
        season = row["season"]
        code = row["code"]
        first_gw = row["first_gw"]
        for gw in range(first_gw, 39):
            grid_rows.append({"season": season, "code": code, "gw": gw})
            
    grid_df = pd.DataFrame(grid_rows)
    
    # Merge aggregated data back onto the grid
    final_df = pd.merge(grid_df, player_gw_agg, on=["season", "code", "gw"], how="left")
    
    # Fill missing values for blank gameweeks
    # If the player had a BGW, they played 0 minutes, scored 0 points, etc.
    expected_cols = ["expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]
    fill_zero_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps", "starts",
        "creativity", "threat", "influence", "ict_index",
        "defensive_contribution", "tackles", "clearances_blocks_interceptions", "recoveries"
    ]
    fill_zero_dict = {c: 0.0 for c in fill_zero_cols if c in final_df.columns}
    final_df = final_df.fillna(fill_zero_dict)
    
    # Fill expected stats with 0.0 only for seasons >= 2022-23 (where expected stats are tracked)
    # For older seasons, leave them as NaN so we know they are unavailable
    for col in expected_cols:
        if col in final_df.columns:
            mask = final_df["season"] >= "2022-23"
            final_df.loc[mask, col] = final_df.loc[mask, col].fillna(0.0)
    
    # Forward-fill player constants like value, name, position, season_team_id, selected
    # Group by (season, code) and forward fill, then backward fill (in case they are missing at the start)
    fill_ffill_cols = ["value", "name", "position", "season_team_id", "selected", "player_id"]
    for col in fill_ffill_cols:
        if col in final_df.columns:
            final_df[col] = final_df.groupby(["season", "code"])[col].ffill().bfill()
            
    # For transfers metrics, if they are missing (e.g. in BGWs), they are 0.0
    fill_zero_transfers = ["transfers_balance", "transfers_in", "transfers_out"]
    fill_transfers_dict = {c: 0.0 for c in fill_zero_transfers if c in final_df.columns}
    final_df = final_df.fillna(fill_transfers_dict)
    
    return final_df


def add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate player-level rolling features strictly on previous gameweeks (chronologically).
    The dataframe must be sorted by (season, code, gw).
    """
    df = df.sort_values(["season", "code", "gw"]).reset_index(drop=True)
    
    # Helper to calculate rolling sum/mean and shift by 1 within season-player groups
    def get_rolling_metric(col: str, window: int, agg_type: str = "sum") -> pd.Series:
        grouped = df.groupby(["season", "code"])[col]
        if agg_type == "sum":
            return grouped.rolling(window, min_periods=1).sum().reset_index(level=[0, 1], drop=True).shift(1)
        else:
            return grouped.rolling(window, min_periods=1).mean().reset_index(level=[0, 1], drop=True).shift(1)

    # 1. PLAYER FORM
    df["total_points_last_1"] = get_rolling_metric("total_points", 1, "sum")
    df["total_points_last_3"] = get_rolling_metric("total_points", 3, "sum")
    df["total_points_last_5"] = get_rolling_metric("total_points", 5, "sum")
    df["total_points_last_10"] = get_rolling_metric("total_points", 10, "sum")
    
    # 2. MINUTES / STARTING
    df["minutes_last_1"] = get_rolling_metric("minutes", 1, "sum")
    df["minutes_last_3"] = get_rolling_metric("minutes", 3, "sum")
    df["minutes_last_5"] = get_rolling_metric("minutes", 5, "sum")
    df["minutes_last_10"] = get_rolling_metric("minutes", 10, "sum")
    
    # Derived starts column to handle missing historical starts
    # Pre-2022-23, starts is estimated as minutes >= 60
    if "starts" in df.columns:
        df["derived_starts"] = np.where(
            df["season"] < "2022-23",
            (df["minutes"] >= 60).astype(float),
            df["starts"].fillna((df["minutes"] >= 60).astype(float))
        )
    else:
        df["derived_starts"] = (df["minutes"] >= 60).astype(float)
        
    df["starts_last_3"] = get_rolling_metric("derived_starts", 3, "sum")
    df["starts_last_5"] = get_rolling_metric("derived_starts", 5, "sum")
    df["starts_last_10"] = get_rolling_metric("derived_starts", 10, "sum")
    
    # Cumulative starts/appearances for starts_rate and average minutes
    df["is_appearance"] = (df["minutes"] > 0).astype(float)
    
    # Chronological cumulative starts, appearances, and minutes before current GW N
    df["cum_minutes"] = df.groupby(["season", "code"])["minutes"].cumsum().shift(1)
    df["cum_appearances"] = df.groupby(["season", "code"])["is_appearance"].cumsum().shift(1)
    df["cum_starts"] = df.groupby(["season", "code"])["derived_starts"].cumsum().shift(1)
    df["cum_gws"] = df.groupby(["season", "code"])["gw"].cumcount() # Represents GWs played before this
    
    df["average_minutes_per_appearance"] = np.where(
        df["cum_appearances"] > 0,
        df["cum_minutes"] / df["cum_appearances"],
        0.0
    )
    df["starts_rate"] = np.where(
        df["cum_gws"] > 0,
        df["cum_starts"] / df["cum_gws"],
        0.0
    )
    
    # Points per 90 (recent and season) with shrinkage/thresholds
    # Direct points per 90 only calculated if player played >= 90 mins, else filled with position avg
    # We will compute position averages during merge/imputation phase or use position global averages
    df["points_per_90_last_3"] = np.where(
        df["minutes_last_3"] >= 90.0,
        df["total_points_last_3"] / (df["minutes_last_3"] / 90.0),
        np.nan # Will impute with position-season averages
    )
    df["points_per_90_last_5"] = np.where(
        df["minutes_last_5"] >= 90.0,
        df["total_points_last_5"] / (df["minutes_last_5"] / 90.0),
        np.nan
    )
    df["points_per_90_last_10"] = np.where(
        df["minutes_last_10"] >= 90.0,
        df["total_points_last_10"] / (df["minutes_last_10"] / 90.0),
        np.nan
    )
    
    df["points_per_90_season"] = np.where(
        df["cum_minutes"] >= 90.0,
        df.groupby(["season", "code"])["total_points"].cumsum().shift(1) / (df["cum_minutes"] / 90.0),
        np.nan
    )
    
    # 3. ATTACKING
    df["goals_last_3"] = get_rolling_metric("goals_scored", 3, "sum")
    df["goals_last_5"] = get_rolling_metric("goals_scored", 5, "sum")
    df["goals_last_10"] = get_rolling_metric("goals_scored", 10, "sum")
    df["assists_last_3"] = get_rolling_metric("assists", 3, "sum")
    df["assists_last_5"] = get_rolling_metric("assists", 5, "sum")
    
    # Expected stats rolling averages (only available from 2022-23 onwards)
    # We calculate them, and they will naturally be NaN for pre-2022-23 seasons
    df["xG_last_3"] = get_rolling_metric("expected_goals", 3, "sum")
    df["xG_last_5"] = get_rolling_metric("expected_goals", 5, "sum")
    df["xG_last_10"] = get_rolling_metric("expected_goals", 10, "sum")
    df["xA_last_3"] = get_rolling_metric("expected_assists", 3, "sum")
    df["xA_last_5"] = get_rolling_metric("expected_assists", 5, "sum")
    df["xA_last_10"] = get_rolling_metric("expected_assists", 10, "sum")
    df["xGI_last_3"] = get_rolling_metric("expected_goal_involvements", 3, "sum")
    df["xGI_last_5"] = get_rolling_metric("expected_goal_involvements", 5, "sum")
    
    # 4. UNDERLYING PERFORMANCE (creativity, threat, influence, ict_index, bps, bonus, defensive_contribution)
    for col in ["creativity", "threat", "influence", "ict_index", "bps", "bonus", "defensive_contribution"]:
        if col in df.columns:
            df[f"{col}_last_3"] = get_rolling_metric(col, 3, "mean")
            df[f"{col}_last_5"] = get_rolling_metric(col, 5, "mean")
            
    # 5. DEFENSIVE
    df["clean_sheets_last_5"] = get_rolling_metric("clean_sheets", 5, "sum")
    df["goals_conceded_last_3"] = get_rolling_metric("goals_conceded", 3, "sum")
    df["goals_conceded_last_5"] = get_rolling_metric("goals_conceded", 5, "sum")
    
    df["tackles_last_5"] = get_rolling_metric("tackles", 5, "sum")
    df["clearances_blocks_interceptions_last_5"] = get_rolling_metric("clearances_blocks_interceptions", 5, "sum")
    df["recoveries_last_5"] = get_rolling_metric("recoveries", 5, "sum")
    
    # 6. GOALKEEPER
    df["saves_last_3"] = get_rolling_metric("saves", 3, "sum")
    df["saves_last_5"] = get_rolling_metric("saves", 5, "sum")
    df["saves_last_10"] = get_rolling_metric("saves", 10, "sum")
    df["saves_per_90_last_5"] = np.where(
        df["minutes_last_5"] >= 90.0,
        df["saves_last_5"] / (df["minutes_last_5"] / 90.0),
        np.nan
    )
    
    # 7. SEASON EXPERIENCE
    df["player_gws_this_season"] = df["cum_gws"]
    df["player_minutes_this_season"] = df["cum_minutes"].fillna(0.0)
    df["player_starts_this_season"] = df["cum_starts"].fillna(0.0)
    
    # 8. PLAYER VALUE / PRICE (value column in player_gw represents price in tenths of million, e.g. 55 = 5.5m)
    # Current price is value at beginning of GW N (which is value from GW N)
    df["current_price"] = df["value"] / 10.0  # Convert to millions (e.g. 5.5)
    
    # Price change
    df["prev_price"] = df.groupby(["season", "code"])["current_price"].shift(1)
    df["price_change_recent"] = (df["current_price"] - df["prev_price"]).fillna(0.0)
    
    # Cumulative points
    df["cum_points"] = df.groupby(["season", "code"])["total_points"].cumsum().shift(1).fillna(0.0)
    
    df["value_for_money"] = df["cum_points"] / df["current_price"]
    df["value_form"] = df["total_points_last_5"].fillna(0.0) / df["current_price"]
    df["value_season"] = df["cum_points"] / df["current_price"]
    
    # 9. OWNERSHIP / TRANSFERS
    # We calculate relative ownership in FPL
    # Max selected in gameweek to normalize selected
    gw_max_selected = df.groupby(["season", "gw"])["selected"].transform("max")
    df["selected_ratio"] = (df["selected"] / gw_max_selected).fillna(0.0)
    df["prev_selected_ratio"] = df.groupby(["season", "code"])["selected_ratio"].shift(1)
    df["ownership_change_recent"] = (df["selected_ratio"] - df["prev_selected_ratio"]).fillna(0.0)
    
    # Normalize transfers by max transfers in that gameweek
    gw_max_transfers_in = df.groupby(["season", "gw"])["transfers_in"].transform("max")
    gw_max_transfers_out = df.groupby(["season", "gw"])["transfers_out"].transform("max")
    
    df["transfers_in_ratio"] = (df["transfers_in"] / gw_max_transfers_in).fillna(0.0)
    df["transfers_out_ratio"] = (df["transfers_out"] / gw_max_transfers_out).fillna(0.0)
    df["transfers_balance_ratio"] = df["transfers_in_ratio"] - df["transfers_out_ratio"]
    
    # Clean up temporary columns
    temp_cols = ["cum_minutes", "cum_appearances", "cum_starts", "cum_gws", "is_appearance", "prev_price", "cum_points", "prev_selected_ratio"]
    df = df.drop(columns=temp_cols, errors="ignore")
    
    return df
