import pandas as pd
from pathlib import Path

def standardize_position(element_type: int) -> str:
    """Map FPL element_type (position) to canonical categories: GK, DEF, MID, FWD."""
    pos_map = {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD",
        5: "MID"  # Fallback/Manager mapping in older seasons
    }
    return pos_map.get(element_type, "MID")


def generate_identity_consistency_report(processed_dir: Path) -> pd.DataFrame:
    """
    Run an identity consistency check.
    Verify if a player_code has stable mappings across different seasons.
    """
    print("\n" + "=" * 50)
    print("PLAYER IDENTITY CONSISTENCY REPORT")
    print("=" * 50)
    
    players_path = processed_dir / "players.parquet"
    player_gw_path = processed_dir / "player_gw.parquet"
    
    if not players_path.exists() or not player_gw_path.exists():
        print("Processed data files not found. Cannot run identity check.")
        return pd.DataFrame()
        
    players = pd.read_parquet(players_path)
    player_gw = pd.read_parquet(player_gw_path)
    
    # Merge player_gw with players on player_id and season
    merged = pd.merge(
        player_gw[["player_id", "season", "name"]],
        players[["player_id", "season", "code", "element_type", "web_name"]],
        on=["player_id", "season"],
        how="left"
    )
    
    # Check for missing codes
    missing_codes = merged["code"].isna().sum()
    print(f"Total GW records checked: {len(merged):,}")
    print(f"Records missing stable player 'code': {missing_codes} ({missing_codes/len(merged):.2%})")
    
    # Group by player code and count unique names and positions
    code_groups = merged.groupby("code").agg(
        names_count=("name", "nunique"),
        unique_names=("name", lambda x: list(x.unique())),
        unique_positions=("element_type", lambda x: [standardize_position(p) for p in x.unique() if pd.notna(p)]),
        seasons_count=("season", "nunique"),
        records_count=("name", "count")
    ).reset_index()
    
    # Check for position instability
    unstable_pos = code_groups[code_groups["unique_positions"].apply(len) > 1]
    print(f"Unique player codes: {len(code_groups):,}")
    print(f"Player codes with multiple positions: {len(unstable_pos)}")
    if len(unstable_pos) > 0:
        print("\nWARNING: Players with unstable positions across seasons:")
        for _, row in unstable_pos.head(10).iterrows():
            print(f"  Code {row['code']}: Names: {row['unique_names']} | Positions: {row['unique_positions']}")
            
    # Check for multiple names associated with the same code (spelling variations)
    spelling_vars = code_groups[code_groups["names_count"] > 1]
    print(f"Player codes with name variations (e.g. spelling changes): {len(spelling_vars)}")
    if len(spelling_vars) > 0:
        print("\nSpelling/name variation samples (top 5):")
        for _, row in spelling_vars.head(5).iterrows():
            print(f"  Code {row['code']}: {row['unique_names']}")
            
    print("\nIdentity check completed!")
    print("=" * 50 + "\n")
    return code_groups
