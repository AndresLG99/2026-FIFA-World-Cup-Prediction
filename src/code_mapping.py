# This file creates/loads a CSV mapping so the pipeline can connect players to national teams.

"""
What code_mapping.py does:
- It creates a “translation sheet” from player codes to team names.
- This impacts the project outcome because without it, we can’t correctly aggregate player stats into team stats,
  so the model can’t learn “Germany’s players are strong” vs “Algeria’s players are weaker” using the player datasets.
"""

import pandas as pd


REQUIRED_COLS = ["nation_code", "team_name"]


def load_nation_code_map(path: str) -> pd.DataFrame:
    """
    Load the nation_code -> team_name mapping CSV and validate it.

    Impact on project:
    - This mapping is the bridge between player stats (nation codes like 'GER')
      and match fixtures (team names like 'Germany').
    - If the mapping is wrong, team features will be missing or assigned to the wrong team.
    """
    df = pd.read_csv(path, encoding="utf-8-sig").copy()  # Handles Excel BOM safely.

    # Validate required columns exist
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"nation_code_map is missing columns: {missing_cols}. Found: {df.columns.tolist()}")

    # Clean strings
    df["nation_code"] = df["nation_code"].astype(str).str.strip()
    df["team_name"] = df["team_name"].astype(str).str.strip()

    # Validate content
    if df["nation_code"].duplicated(keep=False).any():
        dupes = sorted(df.loc[df["nation_code"].duplicated(keep=False), "nation_code"].unique().tolist())
        raise ValueError(f"Duplicate nation_code values: {dupes[:20]}")

    if (df["nation_code"] == "").any():
        bad = df.loc[df["nation_code"] == "", "team_name"].tolist()
        raise ValueError(f"Blank nation_code rows exist (team_name examples): {bad[:10]}")

    if (df["team_name"] == "").any():
        bad = df.loc[df["team_name"] == "", "nation_code"].tolist()
        raise ValueError(f"Blank team_name for codes: {bad[:20]}")

    return df.reset_index(drop=True)


def loadnationcodemap(path: str) -> pd.DataFrame:
    """
    Backward-compatible alias.

    Why:
    - Some modules may still import loadnationcodemap from earlier iterations.
    - This avoids breaking the pipeline while refactoring naming consistency.
    """
    return load_nation_code_map(path)


def test_code_mapping(path: str) -> None:
    """
    Quick standalone check.
    """
    df = load_nation_code_map(path)
    print("nation_code_map:", df.shape)
    print(df.tail(10))


if __name__ == "__main__":
    # Optional standalone test (run from project root)
    from config import DATA_PATHS
    test_code_mapping(DATA_PATHS["nation_code_map"])