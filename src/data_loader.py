# This file loads raw CSVs, standardizes columns, and outputs “cleaned” DataFrames
# (or later, saves them to data/processed/).

"""
What data_loader.py does:

- Loads CSVs into pandas tables (DataFrames).
- Forces the match tables to use the exact 8 columns you know are correct
  (so later scripts don’t crash due to extra columns).
- Converts date into real dates, which is required for time-based training logic.
- Optionally loads "resolved" 2026 fixtures (placeholders replaced) from data/processed/.
"""

import os
import pandas as pd

from config import (
    DATA_PATHS,
    PROCESSED_PATHS,
    USE_RESOLVED_FIXTURES,
)

from src.utils import parse_match_date
from src.name_cleaning import standardize_matches, report_unmapped_teams


MATCH_COLS = [
    "date", "home_team", "away_team",
    "home_score", "away_score",
    "tournament", "country", "neutral",
]


def _ensure_match_cols(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """
    Ensure a DataFrame contains a required schema.

    Strategy:
    - If a required column is missing, create it as NA.
    - Then reorder to the standard column order.

    Impact:
    - Makes downstream feature builders robust to minor schema differences
      (especially common in fixtures templates).
    """
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[required_cols].copy()


def load_country_names() -> pd.DataFrame:
    """
    Load the name mapping table: original_name -> current_name.

    Impact:
    - Without this mapping, team name merges will fail downstream.
    """
    return pd.read_csv(DATA_PATHS["country_names"])


def load_all_matches(raw: bool = False) -> pd.DataFrame:
    """
    Load the main results dataset (1872–2026).

    If raw=False (default):
    - Parses date
    - Standardizes team names using countries_names.csv

    Impact:
    - Makes training data consistent so later feature merges work reliably.
    """
    df = pd.read_csv(DATA_PATHS["all_matches"])
    df = _ensure_match_cols(df, MATCH_COLS)
    df["date"] = parse_match_date(df["date"])

    if raw:
        return df

    country_names = load_country_names()
    df = standardize_matches(df, country_names)
    return df


def load_fixtures_2026(raw: bool = False) -> pd.DataFrame:
    """
    Load the 2026 World Cup fixtures template.

    If raw=False (default):
    - Parses date
    - Standardizes team names using countries_names.csv
    - Optionally loads the "resolved" fixtures from data/processed/ (placeholders replaced)

    Impact:
    - Ensures team names align with training data (same spelling).
    - Allows the pipeline to predict using real teams instead of placeholder strings.
    """
    # Choose which fixtures file to load:
    # - Raw fixtures: data/raw/.../all_matches_WC_2026.csv
    # - Resolved fixtures: data/processed/predictions/fixtures_2026_resolved.csv
    if USE_RESOLVED_FIXTURES and (not raw):
        resolved_path = PROCESSED_PATHS["fixtures_2026_resolved"]
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Resolved fixtures file not found: {resolved_path}. "
                "Run simulate_playoffs.py to generate it, or set USE_RESOLVED_FIXTURES=False in config.py."
            )
        fixtures_path = resolved_path
    else:
        fixtures_path = DATA_PATHS["fixtures_2026"]

    df = pd.read_csv(fixtures_path)
    df = _ensure_match_cols(df, MATCH_COLS)
    df["date"] = parse_match_date(df["date"])

    if raw:
        return df

    country_names = load_country_names()
    df = standardize_matches(df, country_names)
    return df


def load_wc_tables() -> dict:
    """
    Load World Cup historical tables (1930–2022).

    Impact:
    - These tables support player-history exploration and optional validations.
    """
    return {
        "goals": pd.read_csv(DATA_PATHS["wc_goals"]),
        "matches": pd.read_csv(DATA_PATHS["wc_matches"]),
        "player_appearances": pd.read_csv(DATA_PATHS["wc_player_appearances"]),
        "squads": pd.read_csv(DATA_PATHS["wc_squads"]),
        "teams": pd.read_csv(DATA_PATHS["wc_teams"]),
    }


def load_player_seasons() -> dict:
    """
    Load your 3 club-season player stats datasets.

    Impact:
    - These are the raw inputs for player_features.py -> team_features.py.
    """
    return {
        "23_24": pd.read_csv(DATA_PATHS["players_23_24"]),
        "24_25": pd.read_csv(DATA_PATHS["players_24_25"]),
        "25_26": pd.read_csv(DATA_PATHS["players_25_26"]),
    }


def test_loader() -> None:
    """
    Run from project root:
    python -m src.data_loader

    This is a sandbox check only (not required for the pipeline).
    """
    # Local imports so normal pipeline imports stay lightweight.
    from src.player_features import test_player_features
    from src.code_mapping import load_nation_code_map

    country_names = load_country_names()

    matches_raw = load_all_matches(raw=True)
    fixtures_raw = load_fixtures_2026(raw=True)

    print("all_matches:", matches_raw.shape)
    print("fixtures_2026 (raw file):", fixtures_raw.shape)

    matches_clean = standardize_matches(matches_raw, country_names)
    fixtures_clean = standardize_matches(fixtures_raw, country_names)

    unknown_after = report_unmapped_teams(matches_clean, country_names)
    print("unknown team names (AFTER standardization):", unknown_after.shape[0])
    print(unknown_after.head(50))

    wc = load_wc_tables()
    players = load_player_seasons()

    print("wc tables:", {k: v.shape for k, v in wc.items()})
    print("player seasons:", {k: v.shape for k, v in players.items()})

    print(matches_clean.head(3)[["date", "home_team", "away_team", "home_score", "away_score"]])

    # Run player-feature test here (so it only runs when you execute this file)
    test_player_features(players, min_minutes=300)

    nation_map = load_nation_code_map(DATA_PATHS["nation_code_map"])
    print("nation_code_map:", nation_map.shape)
    print(nation_map.head())

    # Optional: show which fixtures are being used when cleaned
    try:
        fixtures_clean_selected = load_fixtures_2026(raw=False)
        print("fixtures_2026 (selected/clean):", fixtures_clean_selected.shape)
        print("USE_RESOLVED_FIXTURES:", USE_RESOLVED_FIXTURES)
    except FileNotFoundError as e:
        print("fixtures_2026 (selected/clean) could not be loaded:", str(e))


if __name__ == "__main__":
    test_loader()