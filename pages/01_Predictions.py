"""
pages/01_Predictions.py

Show model-based 90-minute expected goals predictions for 2026 fixtures.

Inputs:
- data/processed/predictions/predictions_fixtures_2026.csv
  (produced by: python -m src.predict) [file:727][file:726]

Outputs:
- Interactive table with filtering by team and basic sorting.
"""

import os

import pandas as pd
import streamlit as st

from config import DATA_PROCESSED_PREDICTIONS  # [file:726]


@st.cache_data
def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV with Streamlit caching for speed."""
    return pd.read_csv(path)


def _pick_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column in candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    st.title("Fixture Predictions")

    pred_path = os.path.join(DATA_PROCESSED_PREDICTIONS, "predictions_fixtures_2026.csv")
    if not os.path.exists(pred_path):
        st.error(f"Missing predictions file: {pred_path}")
        st.info("Generate it with: python -m src.predict")
        st.stop()

    df = _read_csv(pred_path)

    # Column name compatibility (in case your pipeline uses slightly different names).
    home_col = _pick_existing_col(df, ["home_team", "hometeam"])
    away_col = _pick_existing_col(df, ["away_team", "awayteam"])
    date_col = _pick_existing_col(df, ["date"])
    phg_col = _pick_existing_col(df, ["pred_home_goals", "predhomegoals"])
    pag_col = _pick_existing_col(df, ["pred_away_goals", "predawaygoals"])
    model_col = _pick_existing_col(df, ["model_used", "modelused"])

    if home_col is None or away_col is None:
        st.error("Predictions CSV is missing required team columns.")
        st.write("Columns found:", list(df.columns))
        st.stop()

    # Filters
    teams = sorted(set(df[home_col].astype(str)) | set(df[away_col].astype(str)))
    team = st.selectbox("Filter by team (optional)", ["(All)"] + teams)

    if team != "(All)":
        df = df[(df[home_col] == team) | (df[away_col] == team)].copy()

    # Reorder columns for nicer display (only if they exist).
    preferred = [c for c in [date_col, home_col, away_col, model_col, phg_col, pag_col] if c is not None]
    remaining = [c for c in df.columns if c not in preferred]
    df = df[preferred + remaining]

    st.caption("Tip: click a column header to sort.")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()