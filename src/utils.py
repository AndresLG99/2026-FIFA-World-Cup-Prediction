# This file holds tiny helper functions used everywhere (so you don’t repeat code in 5 different modules).

"""
What utils.py does:
- parse_match_date makes sure dates behave like dates (so you can filter by year).
- nation_to_code turns messy nationality text into a simple code you can group by.
- These two helpers strongly impact the outcome because bad date parsing or mismatched nationality keys
  will break merges and make the model learn wrong patterns.
"""

import pandas as pd


def parse_match_date(series: pd.Series) -> pd.Series:
    """
    Parse dates from BOTH formats used in this project:
    1) 'YYYY-MM-DD' (main dataset)
    2) 'DD-mon-YY'  (fixtures dataset, like 11-jun-26)

    Why two passes?
    - It prevents pandas from guessing formats row-by-row (which triggers warnings).
    - It makes the pipeline deterministic (same input -> same parsed date).

    Output:
    - A pandas datetime Series (NaT for invalid rows).
    """
    # Keep missing values missing (avoid "nan" strings).
    s = series.copy()
    s = s.where(~s.isna(), None)
    s = s.astype("string").str.strip()

    # Pass 1: ISO format from all_matches.csv (fast + no ambiguity)
    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")

    # Pass 2: fixtures format like '11-jun-26'
    mask = dt.isna()
    dt.loc[mask] = pd.to_datetime(s.loc[mask], errors="coerce", format="%d-%b-%y")

    return dt


def nation_to_code(nation_value: str) -> str | None:
    """
    Convert a FBref-style Nation field into a clean 3-letter nation code.

    Example:
    - "us USA" -> "USA"

    Impact:
    - This code becomes the join key between player stats and the nation_code_map.
    - If this is wrong, team aggregation will fail or map players to the wrong national team.
    """
    if pd.isna(nation_value):
        return None

    parts = str(nation_value).split()
    return parts[-1] if parts else None