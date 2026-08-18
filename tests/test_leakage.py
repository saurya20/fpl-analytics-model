import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

def test_pipeline_correctness():
    print("=" * 60)
    print("RUNNING FEATURE ENGINE CORRECTNESS & LEAKAGE CHECK")
    print("=" * 60)
    
    processed_dir = PROJECT_ROOT / "data" / "processed"
    features_path = processed_dir / "features_df.parquet"
    player_gw_path = processed_dir / "player_gw.parquet"
    players_path = processed_dir / "players.parquet"
    
    if not features_path.exists():
        print(f"ERROR: {features_path} does not exist. Run build_features.py first.")
        sys.exit(1)
        
    features = pd.read_parquet(features_path)
    player_gw = pd.read_parquet(player_gw_path)
    players = pd.read_parquet(players_path)
    
    # 1. Check duplicate rows
    print("\n[1/5] Checking duplicates...")
    dups = features.duplicated(subset=["season", "gw", "code"]).sum()
    print(f"  Duplicate rows on (season, gw, code): {dups}")
    assert dups == 0, "Duplicate rows found in features dataset!"
    
    # 2. Check targets
    print("\n[2/5] Checking targets...")
    for target in ["target_points", "target_minutes", "target_60_plus_minutes"]:
        nans = features[target].isna().sum()
        print(f"  {target} NaNs: {nans}")
        assert nans == 0, f"NaNs found in {target}!"
        
    # 3. Check expected stats NaNs in older seasons
    print("\n[3/5] Checking expected stats distribution...")
    pre_22_23 = features[features["season"] < "2022-23"]
    post_22_23 = features[features["season"] >= "2022-23"]
    
    xG_nans_pre = pre_22_23["xG_last_5"].isna().sum()
    xG_nans_post = post_22_23["xG_last_5"].isna().sum()
    
    print(f"  Pre-2022-23 rows: {len(pre_22_23):,} | xG_last_5 NaNs: {xG_nans_pre} ({xG_nans_pre/len(pre_22_23):.2%})")
    print(f"  Post-2022-23 rows: {len(post_22_23):,} | xG_last_5 NaNs: {xG_nans_post} ({xG_nans_post/len(post_22_23):.2%})")
    
    assert xG_nans_pre == len(pre_22_23), "xG_last_5 should be 100% NaN for pre-2022-23 seasons!"
    
    # 4. Leakage Verification Check
    # We will pick a few random players and manually recalculate their rolling points for GW N
    # using the raw player_gw table to verify that features only contain past gameweeks (N-1, N-2, etc.).
    print("\n[4/5] Running strict leakage verification...")
    
    # Map player_gw to player code
    players_sub = players[["player_id", "season", "code"]].drop_duplicates()
    raw_merged = pd.merge(player_gw, players_sub, on=["player_id", "season"], how="left")
    
    # Let's aggregate raw points at (season, gw, code) level
    raw_pts = raw_merged.groupby(["season", "gw", "code"])["total_points"].sum().reset_index()
    
    # Pick a few sample codes and check
    sample_codes = features["code"].unique()[:5]
    
    for code in sample_codes:
        player_feats = features[features["code"] == code]
        player_raw = raw_pts[raw_pts["code"] == code]
        
        # Test across seasons and GWs
        seasons = player_feats["season"].unique()
        for season in seasons:
            s_feats = player_feats[player_feats["season"] == season].sort_values("gw")
            s_raw = player_raw[player_raw["season"] == season].sort_values("gw")
            
            for test_gw in [5, 10, 15, 20, 25]:
                feat_row = s_feats[s_feats["gw"] == test_gw]
                if feat_row.empty:
                    continue
                    
                # Get total points from features
                points_last_3_feat = feat_row["total_points_last_3"].values[0]
                
                # Manually calculate from raw (should be sum of test_gw-1, test_gw-2, test_gw-3 in this season)
                past_gws = [test_gw - 1, test_gw - 2, test_gw - 3]
                raw_past_pts = s_raw[s_raw["gw"].isin(past_gws)]["total_points"].sum()
                
                # Check match
                print(f"  Code {code} | Season {season} | GW {test_gw:02d}: Feature={points_last_3_feat} | Raw={raw_past_pts}")
                assert np.isclose(points_last_3_feat, raw_past_pts), f"LEAKAGE OR INTEGRITY ERROR: GW {test_gw} rolling sum mismatch!"
            
    print("\n[5/5] Checking Blank Gameweek (BGW) representation...")
    # BGWs should have num_fixtures == 0
    bgw_rows = features[features["num_fixtures"] == 0]
    print(f"  Number of player-BGW rows detected: {len(bgw_rows):,}")
    if len(bgw_rows) > 0:
        print("  Sample BGW row targets (should be 0):")
        print(bgw_rows[["season", "gw", "code", "target_points", "target_minutes"]].head(3))
        # For genuine BGW, points and minutes must be 0
        assert (bgw_rows["target_points"] == 0).all(), "BGW target points must be 0!"
        assert (bgw_rows["target_minutes"] == 0).all(), "BGW target minutes must be 0!"
        
    print("\nALL CORRECTNESS & LEAKAGE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_pipeline_correctness()
