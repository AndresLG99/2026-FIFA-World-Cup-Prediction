# Predict scores for fixtures using the TWO-MODEL system.

"""
What predict.py does:

- Loads the 4 saved models from /models:
  - player_home_goals.joblib, player_away_goals.joblib
  - history_home_goals.joblib, history_away_goals.joblib
- Loads the feature column lists used during training (2 CSVs).
- Loads both fixture feature tables created by match_features.py:
  - model_table_player_fixtures_2026.csv
  - model_table_history_fixtures_2026.csv
- Produces one final predictions table:
  - expected home goals, expected away goals
  - which model was used (PLAYER vs HISTORY)
  - plus basic match identifiers (date, teams, etc.)

Notes:
- Knockout rounds in the FIFA template often contain placeholders like:
  "Group E Winners", "Match 74 Winner", "Group A/B/C/D 3rd Place"
  which are not real teams and should not be scored directly unless you simulate the bracket.
"""

import os
import pandas as pd
from joblib import load

from config import MODELS_DIR, PROCESSED_PATHS


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def load_feature_list(path: str) -> list[str]:
    """
    Load the feature column names used during training.
    """
    cols = pd.read_csv(path, header=None)[0].astype(str).tolist()
    return cols


def safe_align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Ensure dataframe has exactly the training-time feature columns.

    - Missing columns are added as 0.0
    - Extra columns are ignored
    """
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[feature_cols].astype(float)


def build_placeholder_mask(df_id: pd.DataFrame) -> pd.Series:
    """
    Identify non-team placeholders that appear in knockout-stage fixture templates.
    """
    home = df_id["home_team"].astype(str)
    away = df_id["away_team"].astype(str)

    is_placeholder = (
        home.str.startswith(("Group ", "Match "))
        | away.str.startswith(("Group ", "Match "))
        | home.str.contains("3rd Place", na=False)
        | away.str.contains("3rd Place", na=False)
        | home.str.contains("/", na=False)
        | away.str.contains("/", na=False)
    )
    return is_placeholder


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> None:
    # -----------------------
    # Load models + feature lists
    # -----------------------
    player_home = load(os.path.join(MODELS_DIR, "player_home_goals.joblib"))
    player_away = load(os.path.join(MODELS_DIR, "player_away_goals.joblib"))
    hist_home = load(os.path.join(MODELS_DIR, "history_home_goals.joblib"))
    hist_away = load(os.path.join(MODELS_DIR, "history_away_goals.joblib"))

    player_cols = load_feature_list(os.path.join(MODELS_DIR, "player_feature_cols.csv"))
    hist_cols = load_feature_list(os.path.join(MODELS_DIR, "history_feature_cols.csv"))

    # -----------------------
    # Load fixture feature tables
    # -----------------------
    fx_player = pd.read_csv(PROCESSED_PATHS["model_table_player_fixtures"])
    fx_hist = pd.read_csv(PROCESSED_PATHS["model_table_history_fixtures"])

    fx_player["date"] = pd.to_datetime(fx_player["date"], errors="coerce")
    fx_hist["date"] = pd.to_datetime(fx_hist["date"], errors="coerce")

    # Base identifiers (same rows/order as feature tables)
    out = fx_player[
        [
            "date", "home_team", "away_team", "tournament", "country", "neutral",
            "is_neutral", "is_home_host",
            "home_has_player_features", "away_has_player_features",
        ]
    ].copy()

    # -----------------------
    # Filter out placeholder fixtures (knockout template rows)
    # -----------------------
    is_placeholder = build_placeholder_mask(out)
    keep = ~is_placeholder

    out = out.loc[keep].copy()
    fx_player = fx_player.loc[keep].copy()
    fx_hist = fx_hist.loc[keep].copy()

    # -----------------------
    # Decide which model to use per fixture
    # -----------------------
    use_player = (out["home_has_player_features"] == 1) & (out["away_has_player_features"] == 1)

    out["model_used"] = "HISTORY"
    out.loc[use_player, "model_used"] = "PLAYER"

    out["pred_home_goals"] = pd.NA
    out["pred_away_goals"] = pd.NA

    # -----------------------
    # Predict with PLAYER model (eligible matches)
    # -----------------------
    if use_player.any():
        Xp = safe_align_features(fx_player.loc[use_player], player_cols)
        out.loc[use_player, "pred_home_goals"] = player_home.predict(Xp)
        out.loc[use_player, "pred_away_goals"] = player_away.predict(Xp)

    # -----------------------
    # Predict with HISTORY model (fallback matches)
    # -----------------------
    use_hist = ~use_player
    if use_hist.any():
        Xh = safe_align_features(fx_hist.loc[use_hist], hist_cols)
        out.loc[use_hist, "pred_home_goals"] = hist_home.predict(Xh)
        out.loc[use_hist, "pred_away_goals"] = hist_away.predict(Xh)

    out["pred_home_goals"] = out["pred_home_goals"].astype(float)
    out["pred_away_goals"] = out["pred_away_goals"].astype(float)

    # -----------------------
    # Save
    # -----------------------
    pred_path = PROCESSED_PATHS["predictions_fixtures_2026"]
    out.to_csv(pred_path, index=False)

    print("Saved predictions:", pred_path)
    print("Total fixtures in feature tables:", len(pd.read_csv(PROCESSED_PATHS["model_table_player_fixtures"])))
    print("Predicted fixtures (real teams only):", len(out))
    print("Filtered placeholders:", int(is_placeholder.sum()))
    print("Used PLAYER model:", int(use_player.sum()))
    print("Used HISTORY model:", int(use_hist.sum()))
    print(out.head(10))


if __name__ == "__main__":
    main()