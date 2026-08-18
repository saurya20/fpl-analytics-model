from pathlib import Path
import re

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAASTAV_ROOT = PROJECT_ROOT.parent / "Fantasy-Premier-League"
VAASTAV_DATA = VAASTAV_ROOT / "data"

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"


# ============================================================
# HELPERS
# ============================================================

def get_seasons():
    """Return all season directories available in Vaastav."""
    
    seasons = []

    for path in VAASTAV_DATA.iterdir():
        if path.is_dir() and re.match(r"^\d{4}-\d{2}$", path.name):
            seasons.append(path.name)

    return sorted(seasons)


def read_csv(path):
    """Read a CSV while handling historical encoding differences."""

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path}: {last_error}"
    )

# ============================================================
# GAMEWEEK DATA
# ============================================================

def load_gameweek_data(seasons):
    """Load all GW CSV files across all seasons."""

    frames = []

    for season in seasons:

        gw_directory = VAASTAV_DATA / season / "gws"

        if not gw_directory.exists():
            print(f"Skipping {season}: no GW directory")
            continue

        # Only match files like gw1.csv, gw2.csv, etc.
        gw_files = sorted(
            [
                f for f in gw_directory.glob("gw*.csv")
                if re.fullmatch(r"gw\d+\.csv", f.name)
            ],
            key=lambda x: int(
                re.search(r"gw(\d+)\.csv", x.name).group(1)
            )
        )

        print(f"{season}: {len(gw_files)} gameweeks")

        for gw_file in gw_files:

            match = re.match(r"gw(\d+)\.csv", gw_file.name)

            if not match:
                continue

            gameweek = int(match.group(1))

            df = read_csv(gw_file)

            df["season"] = season
            df["gw"] = gameweek

            # Rename FPL's player identifier
            if "element" in df.columns:
                df = df.rename(columns={"element": "player_id"})

            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )


# ============================================================
# PLAYER DATA
# ============================================================

def load_players(seasons):
    """Load player metadata for every season."""

    frames = []

    for season in seasons:

        path = VAASTAV_DATA / season / "players_raw.csv"

        if not path.exists():
            print(f"Skipping players for {season}")
            continue

        df = read_csv(path)

        df["season"] = season

        # 'id' is the stable FPL player identifier
        if "id" in df.columns:
            df = df.rename(columns={"id": "player_id"})

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )


# ============================================================
# FIXTURES
# ============================================================

def load_fixtures(seasons):
    """Load fixture data for every season."""

    frames = []

    for season in seasons:

        path = VAASTAV_DATA / season / "fixtures.csv"

        if not path.exists():
            print(f"Skipping fixtures for {season}")
            continue

        df = read_csv(path)

        df["season"] = season

        # The stats column contains a large JSON-like string.
        # We keep the raw data in Vaastav but exclude it from
        # our initial processed fixture table.
        if "stats" in df.columns:
            df = df.drop(columns=["stats"])

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )


# ============================================================
# TEAMS
# ============================================================

def load_teams(seasons):
    """Load team information for every season."""

    frames = []

    for season in seasons:

        path = VAASTAV_DATA / season / "teams.csv"

        if not path.exists():
            print(f"Skipping teams for {season}")
            continue

        df = read_csv(path)

        df["season"] = season

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_gameweek_data(df):

    print("\nGameweek validation")
    print("-" * 40)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "player_id" in df.columns:
        print(
            f"Unique players: "
            f"{df['player_id'].nunique():,}"
        )

    if "season" in df.columns:
        print(
            f"Seasons: "
            f"{df['season'].nunique()}"
        )

    if "gw" in df.columns:
        print(
            f"Gameweeks: "
            f"{df['gw'].nunique()}"
        )

    if {"season", "gw", "player_id"}.issubset(df.columns):

        duplicates = df.duplicated(
            subset=["season", "gw", "player_id", "fixture"]
        ).sum()

        print(
            f"Duplicate player-GW-fixture rows: {duplicates}"
        )


def validate_players(df):

    print("\nPlayer validation")
    print("-" * 40)

    print(f"Rows: {len(df):,}")

    if "player_id" in df.columns:
        print(
            f"Unique player IDs: "
            f"{df['player_id'].nunique():,}"
        )


def validate_fixtures(df):

    print("\nFixture validation")
    print("-" * 40)

    print(f"Rows: {len(df):,}")

    if "season" in df.columns:
        print(
            f"Seasons: "
            f"{df['season'].nunique()}"
        )


# ============================================================
# SAVE
# ============================================================

def save_data(gameweeks, players, fixtures, teams):

    PROCESSED_DATA.mkdir(
        parents=True,
        exist_ok=True
    )

    gameweeks.to_parquet(
        PROCESSED_DATA / "player_gw.parquet",
        index=False
    )

    players.to_parquet(
        PROCESSED_DATA / "players.parquet",
        index=False
    )

    fixtures.to_parquet(
        PROCESSED_DATA / "fixtures.parquet",
        index=False
    )

    teams.to_parquet(
        PROCESSED_DATA / "teams.parquet",
        index=False
    )

    print("\nSaved:")
    print("  data/processed/player_gw.parquet")
    print("  data/processed/players.parquet")
    print("  data/processed/fixtures.parquet")
    print("  data/processed/teams.parquet")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FPL DATASET BUILDER")
    print("=" * 60)

    print(f"\nVaastav data: {VAASTAV_DATA}")

    if not VAASTAV_DATA.exists():
        raise FileNotFoundError(
            f"Vaastav data not found: {VAASTAV_DATA}"
        )

    seasons = get_seasons()

    print(f"\nFound {len(seasons)} seasons:")
    for season in seasons:
        print(f"  - {season}")

    print("\nLoading gameweek data...")
    gameweeks = load_gameweek_data(seasons)

    if not gameweeks.empty and {"season", "gw", "player_id", "fixture"}.issubset(gameweeks.columns):
        initial_len = len(gameweeks)
        gameweeks = gameweeks.drop_duplicates(subset=["season", "gw", "player_id", "fixture"], keep="first")
        print(f"Dropped {initial_len - len(gameweeks)} duplicate player-GW-fixture rows.")

    print("\nLoading player data...")
    players = load_players(seasons)

    print("\nLoading fixture data...")
    fixtures = load_fixtures(seasons)

    print("\nLoading team data...")
    teams = load_teams(seasons)

    # Validation
    validate_gameweek_data(gameweeks)
    validate_players(players)
    validate_fixtures(fixtures)

    # Save
    save_data(
        gameweeks,
        players,
        fixtures,
        teams
    )

    print("\nDataset build complete!")


if __name__ == "__main__":
    main()