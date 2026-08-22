import pandas as pd
import json
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def normalize_name(name: str) -> str:
    """
    Standardize a player name by converting to lowercase, stripping accents,
    replacing hyphens, and trimming whitespace.
    """
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize('NFD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    name = name.lower().strip()
    name = name.replace("-", " ").replace("'", "").replace(".", "")
    return name

def load_mapping_overrides() -> Dict[str, str]:
    """
    Load manual mapping overrides from config/player_mapping_overrides.json.
    """
    overrides_file = CONFIG_PATH / "player_mapping_overrides.json"
    if not overrides_file.exists():
        CONFIG_PATH.mkdir(parents=True, exist_ok=True)
        with open(overrides_file, "w") as f:
            json.dump({}, f, indent=4)
        return {}
    try:
        with open(overrides_file, "r") as f:
            data = json.load(f)
            # Ensure keys are strings of FPL codes
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"[Warning] Failed to load mapping overrides: {e}")
        return {}

def build_player_id_mapping(
    fpl_players: pd.DataFrame,
    external_players: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate player mapping table.
    - Matches FPL player 'code' to external player ID ('fbref_id').
    - Utilizes name normalization, FPL club constraints, and manual overrides.
    - If external_players is not provided, initialized mapping will map FPL codes to placeholder IDs.
    """
    overrides = load_mapping_overrides()
    
    mapping_rows = []
    
    # If no external pool is supplied, mock a perfect match for bootstrap validation
    external_pool = []
    if external_players is not None and not external_players.empty:
        for _, row in external_players.iterrows():
            ext_id = str(row["fbref_id"])
            ext_name = row["fbref_name"]
            norm_name = normalize_name(ext_name)
            ext_team = row.get("team", "")
            external_pool.append({
                "id": ext_id,
                "name": ext_name,
                "norm_name": norm_name,
                "team": ext_team
            })
            
    # Iterate through FPL players
    for _, row in fpl_players.iterrows():
        fpl_code = str(int(row["code"]))
        fpl_name = row["name"]
        fpl_team = row.get("team", "")
        norm_fpl = normalize_name(fpl_name)
        
        fbref_id = None
        fbref_name = None
        confidence = "unmatched"
        
        # 1. Check manual override first
        if fpl_code in overrides:
            fbref_id = overrides[fpl_code]
            ext_match = next((item for item in external_pool if item["id"] == fbref_id), None)
            if ext_match:
                fbref_name = ext_match["name"]
                confidence = "override"
            else:
                fbref_name = f"Override ID: {fbref_id}"
                confidence = "override_unverified"
                
        # 2. Try exact name & team match if external pool exists
        elif external_pool:
            potential_matches = []
            
            for ext_info in external_pool:
                # If name matches exactly
                if ext_info["norm_name"] == norm_fpl:
                    # Double check team matches (if available)
                    if not fpl_team or not ext_info["team"] or str(fpl_team).lower() == str(ext_info["team"]).lower():
                        potential_matches.append((ext_info["id"], ext_info["name"], "exact"))
                # Try parts matching (first or last name matches)
                elif norm_fpl in ext_info["norm_name"] or ext_info["norm_name"] in norm_fpl:
                    if not fpl_team or not ext_info["team"] or str(fpl_team).lower() == str(ext_info["team"]).lower():
                        potential_matches.append((ext_info["id"], ext_info["name"], "fuzzy"))
                        
            if len(potential_matches) == 1:
                fbref_id, fbref_name, confidence = potential_matches[0]
            elif len(potential_matches) > 1:
                # Ambiguous matches: report rather than silently mismatching
                print(f"[Warning] Ambiguous identity matches for FPL '{fpl_name}' (Code: {fpl_code}): "
                      f"{[m[1] for m in potential_matches]}. Mapping to unmatched fallback.")
                fbref_id = f"ext_{fpl_code}"
                fbref_name = fpl_name
                confidence = "ambiguous_fallback"
            else:
                # No match found, report unmatched
                print(f"[Warning] Unmatched player: '{fpl_name}' (FPL Code: {fpl_code}). Mapping to unmatched fallback.")
                fbref_id = f"ext_{fpl_code}"
                fbref_name = fpl_name
                confidence = "unmatched_fallback"
        else:
            # Fallback placeholder mapping
            fbref_id = f"ext_{fpl_code}"
            fbref_name = fpl_name
            confidence = "placeholder"
            
        mapping_rows.append({
            "fpl_code": int(fpl_code),
            "fpl_name": fpl_name,
            "fbref_id": fbref_id,
            "fbref_name": fbref_name,
            "confidence": confidence
        })
        
    mapping_df = pd.DataFrame(mapping_rows)
    
    # Check for player identity collisions (excluding fallbacks)
    valid_mappings = mapping_df[~mapping_df["confidence"].isin(["unmatched_fallback", "placeholder", "ambiguous_fallback"])]
    duplicated_ext_ids = valid_mappings[valid_mappings.duplicated(subset=["fbref_id"], keep=False)]
    if not duplicated_ext_ids.empty:
        for ext_id in duplicated_ext_ids["fbref_id"].unique():
            collision_rows = duplicated_ext_ids[duplicated_ext_ids["fbref_id"] == ext_id]
            fpl_names = collision_rows["fpl_name"].tolist()
            fpl_codes = collision_rows["fpl_code"].tolist()
            print(f"[Warning] Player identity collision detected! External ID '{ext_id}' matches multiple FPL players: "
                  f"{list(zip(fpl_names, fpl_codes))}. Review player overrides json.")
                  
    # Save the mapping
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mapping_df.to_parquet(PROCESSED_DIR / "player_id_mapping.parquet", index=False)
    
    return mapping_df

if __name__ == "__main__":
    # Test loading and building placeholder map
    players_path = PROCESSED_DIR / "players.parquet"
    if players_path.exists():
        players_df = pd.read_parquet(players_path)
        # Select distinct player codes
        fpl_sub = players_df[["code", "name"]].drop_duplicates(subset=["code"])
        mapping = build_player_id_mapping(fpl_sub)
        print(f"Generated identity map containing {len(mapping)} players.")
    else:
        print("[Warning] data/processed/players.parquet not found. Run dataset builder first.")
