"""
What each function “means”:

- build_country_mapping: makes a “dictionary” like a translator book from old names to new names.
- apply_country_mapping: uses the translator book to replace names in the match table so teams are spelled consistently.
- apply_manual_fixes: fixes the tiny leftovers the translator book doesn’t know yet.
- standardize_matches: a “do it all” button to clean a matches DataFrame.
- report_unmapped_teams: returns a table of names that still don’t match so they can be fixed before training.
"""

import pandas as pd

MATCH_TEAM_COLS = ["home_team", "away_team"]


def build_country_mapping(country_names_df: pd.DataFrame) -> dict:
    """
    Build a dictionary that maps old/alternate team names -> current team name.

    Example:
    - 'Czechoslovakia' -> 'Czechia'
    - 'Netherlands Antilles' -> 'Curaçao'

    Impact on project:
    - If names don't match, later merges (features, Elo, squads, etc.) will fail.
    """
    df = country_names_df.copy()

    df["original_name"] = df["original_name"].astype(str).str.strip()
    df["current_name"] = df["current_name"].astype(str).str.strip()

    name_map = dict(zip(df["original_name"], df["current_name"]))
    return name_map


def apply_manual_fixes(name_map: dict) -> dict:
    """
    Add extra hard-coded name fixes that are not present in countries_names.csv.

    Impact on project:
    - Helps resolve rare edge cases without editing the raw CSV.
    """
    manual = {
        "São Tomé and Príncipe": "São Tome and Principe",
        "São Tome and Principe": "São Tome and Principe",
        "Marshall Islands": "Marshall Islands",
        "USA": "United States",
    }

    merged = dict(name_map)
    merged.update(manual)
    return merged


def apply_country_mapping(
    matches_df: pd.DataFrame,
    name_map: dict,
    team_cols=MATCH_TEAM_COLS,
) -> pd.DataFrame:
    """
    Replace team names using name_map in selected columns.

    Impact on project:
    - Makes sure 'home_team' and 'away_team' match across datasets.
    - Reduces duplicates like 'West Germany' vs 'Germany'.
    """
    df = matches_df.copy()

    for col in team_cols:
        cleaned = df[col].astype(str).str.strip()
        df[col] = cleaned.map(name_map).fillna(cleaned)

    return df


def standardize_matches(matches_df: pd.DataFrame, country_names_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-stop function:
    1) Apply mapping table
    2) Apply manual fixes
    3) Strip whitespace

    Impact on project:
    - After this, team names are consistent, so feature-building and modeling work.
    """
    name_map = build_country_mapping(country_names_df)
    name_map = apply_manual_fixes(name_map)

    df = apply_country_mapping(matches_df, name_map, team_cols=MATCH_TEAM_COLS)

    for col in MATCH_TEAM_COLS:
        df[col] = df[col].astype(str).str.strip()

    return df


def report_unmapped_teams(matches_df: pd.DataFrame, country_names_df: pd.DataFrame) -> pd.DataFrame:
    """
    After standardization, a team name is 'valid' if it can appear as an output
    of the mapping dictionary (including manual additions).

    Returns:
    - DataFrame with one column: team_name
    """
    name_map = build_country_mapping(country_names_df)
    name_map = apply_manual_fixes(name_map)

    valid = set(name_map.values())
    teams = pd.unique(matches_df[MATCH_TEAM_COLS].values.ravel("K"))

    unknown = sorted([str(t).strip() for t in teams if str(t).strip() not in valid])
    return pd.DataFrame({"team_name": unknown})