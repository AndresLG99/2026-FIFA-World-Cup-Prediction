"""
What player_features.py does:
- It takes 3 different “player tables” and turns them into one clean player table where every season has the same columns.
- It makes new “per 90 minutes” stats in a consistent way (so the model isn’t confused by different season formats).
- It filters out players who barely played, because those stats can be misleading and hurt predictions.
"""

import pandas as pd
import numpy as np

from src.utils import nation_to_code


BASE_COLS = [
    "Player", "Nation", "Pos", "Squad", "Comp",
    "Age", "Born", "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
    "xG", "npxG", "xAG", "npxG+xAG",
    "PrgC", "PrgP", "PrgR"
]

NUMERIC_COLS = [
    "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
    "xG", "npxG", "xAG", "npxG+xAG",
    "PrgC", "PrgP", "PrgR"
]


def keep_base_columns(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we know exist across every season.

    Impact:
    - Makes features consistent across 23/24, 24/25, 25/26.
    - Prevents the model from depending on columns that only exist in one season.
    """
    df = players_df.copy()
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[BASE_COLS].copy()


def coerce_numeric(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Force numeric columns to numeric dtype.

    Impact:
    - Prevents string math bugs (e.g., '10' / '2' issues).
    - Ensures filtering by minutes works deterministically.
    """
    df = players_df.copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def add_nation_code(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'Nation' like 'us USA' into a clean 3-letter code 'USA'.

    Impact:
    - This becomes the key you can group by to get national-team averages.
    """
    df = players_df.copy()
    df["nation_code"] = df["Nation"].apply(nation_to_code)
    df["nation_code"] = df["nation_code"].astype("string").str.strip()
    return df


def safe_per90(numerator: pd.Series, ninety: pd.Series) -> pd.Series:
    """
    Compute per-90 safely.
    If 90s is 0 or missing, return 0.

    Impact:
    - Prevents divide-by-zero and prevents crazy values for players with tiny minutes.
    """
    denom = ninety.replace(0, np.nan)
    out = numerator / denom
    return out.fillna(0.0)


def add_per90_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-90 columns consistently for all seasons.

    We compute per-90 from totals and '90s' (minutes / 90).
    """
    df = players_df.copy()

    df["gls_90_calc"] = safe_per90(df["Gls"], df["90s"])
    df["ast_90_calc"] = safe_per90(df["Ast"], df["90s"])
    df["xg_90_calc"] = safe_per90(df["xG"], df["90s"])
    df["xag_90_calc"] = safe_per90(df["xAG"], df["90s"])
    df["prgp_90_calc"] = safe_per90(df["PrgP"], df["90s"])
    df["prgc_90_calc"] = safe_per90(df["PrgC"], df["90s"])
    df["prgr_90_calc"] = safe_per90(df["PrgR"], df["90s"])

    return df


def filter_by_minutes(players_df: pd.DataFrame, min_minutes: int) -> pd.DataFrame:
    """
    Remove players with very low minutes.

    Impact:
    - Reduces noise: someone with 5 minutes and 1 goal would look unrealistically elite.
    """
    df = players_df.copy()
    return df[df["Min"] >= float(min_minutes)].copy()


def drop_missing_nation_code(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows without a valid nation_code.

    Impact:
    - Prevents later merges from producing missing team_name.
    - Keeps team aggregates clean.
    """
    df = players_df.copy()
    return df[df["nation_code"].notna() & (df["nation_code"] != "")].copy()


def standardize_one_season(players_df: pd.DataFrame, season_label: str, min_minutes: int) -> pd.DataFrame:
    """
    Full standardization for one season file.
    """
    df = keep_base_columns(players_df)
    df = coerce_numeric(df)
    df = add_nation_code(df)
    df = drop_missing_nation_code(df)
    df = add_per90_features(df)
    df = filter_by_minutes(df, min_minutes)
    df["season"] = season_label
    return df


def build_players_master(players_seasons: dict, min_minutes: int) -> pd.DataFrame:
    """
    Combine all seasons into one table.

    players_seasons example:
      {"23_24": df23, "24_25": df24, "25_26": df25}

    Output:
      One DataFrame with consistent columns + season label.
    """
    parts = []
    for season_label in sorted(players_seasons.keys()):
        parts.append(standardize_one_season(players_seasons[season_label], season_label, min_minutes))
    return pd.concat(parts, ignore_index=True)


def test_player_features(players_seasons: dict, min_minutes: int = 300) -> None:
    """
    Quick test prints shapes and a sample of the output.
    """
    master = build_players_master(players_seasons, min_minutes=min_minutes)
    print("players_master:", master.shape)
    print(master[["Player", "nation_code", "season", "Min", "gls_90_calc", "xg_90_calc"]].head(10))