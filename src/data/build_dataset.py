from pathlib import Path


# Project locations
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAASTAV_ROOT = PROJECT_ROOT.parent / "Fantasy-Premier-League"

DATA_ROOT = VAASTAV_ROOT / "data"


def main():
    print("FPL Dataset Builder")
    print("=" * 40)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Vaastav repo: {VAASTAV_ROOT}")
    print(f"Data directory: {DATA_ROOT}")

    if not VAASTAV_ROOT.exists():
        print("\nERROR: Vaastav repository not found.")
        print("Expected location:")
        print(VAASTAV_ROOT)
        return

    if not DATA_ROOT.exists():
        print("\nERROR: Vaastav data directory not found.")
        return

    print("\nVaastav repository found!")
    print("\nAvailable seasons:")

    for season in sorted(DATA_ROOT.iterdir()):
        if season.is_dir():
            print(f"  - {season.name}")


if __name__ == "__main__":
    main()