# This file builds the final "model tables" for a two-model approach.

"""
What match_features.py does:

- It “glues” features onto matches, but it produces TWO different feature tables:

1) Player-feature tables (Model A input)
- Uses club player stats -> team averages (team_features.py)
- Best quality when available
- Only valid for teams that exist in the player datasets + mapping

2) Match-history tables (Model B input)
- Uses past international match results only (Elo + rolling form)
- Works for many more teams (fallback when player features are missing)

Why we do this:
- The player dataset covers only a subset of national teams.
- Historical matches contain many teams that are not covered by player data.
- The two-model approach lets you predict the full World Cup:
  - Use player model when both teams have player features
  - Otherwise, use history model as fallback
"""

import pandas as pd

from config import (
    DATA_PATHS,
    PROCESSED_PATHS,
    MIN_PLAYER_MINUTES,
    TRAIN_MAX_DATE,
    ROLLING_WINDOW,
    PREDICTION_ASOF_DATE,
)

from src.data_loader import load_all_matches, load_fixtures_2026, load_player_seasons
from src.player_features import build_players_master
from src.team_features import build_team_features
from src.code_mapping import load_nation_code_map
from src.history_features import build_history_features, build_team_state_asof


# Player-derived features that exist in team_features.py output
PLAYER_TEAM_FEATURE_COLS = [
    "gls_90_calc", "ast_90_calc", "xg_90_calc", "xag_90_calc",
    "prgp_90_calc", "prgc_90_calc", "prgr_90_calc",
]


def add_basic_match_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple match-level context features.

    Impact:
    - Gives the model context like neutral site and host advantage.
    - These are useful for both the player model and the history model.
    """
    out = df.copy()

    # Keep these flags numeric so models can use them directly.
    out["is_neutral"] = out["neutral"].astype(int)
    out["is_home_host"] = (out["home_team"] == out["country"]).astype(int)

    return out


# -------------------------------------------------------------------
# PLAYER MODEL TABLES (Model A)
# -------------------------------------------------------------------

def build_player_team_features_table() -> pd.DataFrame:
    """
    Build team-level features from the club player stats.

    Steps:
    1) Load 3 seasons of player stats
    2) Standardize them into players_master (per-90 + nation_code)
    3) Map nation_code -> team_name using nation_code_map.csv
    4) Aggregate players -> team (minutes-weighted averages)

    Output:
    - One row per team_name with columns in PLAYER_TEAM_FEATURE_COLS (+ n_players_used)
    """
    players = load_player_seasons()
    players_master = build_players_master(players, min_minutes=MIN_PLAYER_MINUTES)

    nation_map = load_nation_code_map(DATA_PATHS["nation_code_map"])
    team_features = build_team_features(players_master, nation_map)

    return team_features


def merge_player_team_features(matches_df: pd.DataFrame, team_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge player-derived team features onto matches twice:
    - home team -> home_*
    - away team -> away_*

    Adds:
    - home_has_player_features
    - away_has_player_features

    Impact:
    - Creates Model A inputs (player-driven team strength).
    - Flags tell the pipeline when Model A can be used safely.
    """
    df = matches_df.copy()
    tf = team_features_df.copy()

    # Home merge
    home_tf = tf.rename(columns={"team_name": "home_team"}).add_prefix("home_")
    home_tf = home_tf.rename(columns={"home_home_team": "home_team"})
    df = df.merge(home_tf, on="home_team", how="left")

    # Away merge
    away_tf = tf.rename(columns={"team_name": "away_team"}).add_prefix("away_")
    away_tf = away_tf.rename(columns={"away_away_team": "away_team"})
    df = df.merge(away_tf, on="away_team", how="left")

    # Availability flags (use xG as a “sentinel” feature)
    df["home_has_player_features"] = (~df["home_xg_90_calc"].isna()).astype(int)
    df["away_has_player_features"] = (~df["away_xg_90_calc"].isna()).astype(int)

    return df


def build_player_model_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Model A tables.

    Returns:
    - player_train:
      Only matches where BOTH teams have player features.
      (Avoids training the player model on missing/imputed team strengths.)

    - player_fixtures:
      All fixtures with player features where available.
      Flags indicate whether Model A can be used for a given fixture.
    """
    matches = load_all_matches(raw=False)
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches = add_basic_match_flags(matches)

    fixtures = load_fixtures_2026(raw=False)
    fixtures = add_basic_match_flags(fixtures)

    team_features = build_player_team_features_table()

    matches = merge_player_team_features(matches, team_features)
    fixtures = merge_player_team_features(fixtures, team_features)

    player_train = matches[
        (matches["home_has_player_features"] == 1) &
        (matches["away_has_player_features"] == 1)
    ].copy()

    player_fixtures = fixtures.copy()

    return player_train, player_fixtures


# -------------------------------------------------------------------
# HISTORY MODEL TABLES (Model B)
# -------------------------------------------------------------------

def _safe_fill_column(df: pd.DataFrame, col: str, fill_value, cast=None) -> pd.DataFrame:
    """
    Fill a column safely, creating it if missing.

    Impact:
    - Prevents KeyErrors when team-state merges produce missing columns.
    - Keeps feature tables schema-stable across refactors.
    """
    out = df.copy()
    if col not in out.columns:
        out[col] = fill_value
    out[col] = out[col].fillna(fill_value)
    if cast is not None:
        out[col] = out[col].astype(cast)
    return out


def build_history_model_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Model B tables (match-history features).

    Train table:
    - Uses pre-match Elo + rolling form computed sequentially over time.
    - Cut off at TRAIN_MAX_DATE to avoid leakage into evaluation folds.

    Fixtures table:
    - Uses team state computed as-of PREDICTION_ASOF_DATE (if provided),
      otherwise TRAIN_MAX_DATE + 1 day.
    - Attaches last-known Elo/form snapshot to each fixture team.

    IMPORTANT (column consistency):
    - The history TRAIN table has: elo_home_pre, elo_away_pre, elo_diff_pre
    - The fixtures merge creates: home_elo_pre, away_elo_pre (from team_state elo_pre)
    - We rename to match training feature names so predict/train align.
    """
    # -----------------------
    # Training history features
    # -----------------------
    matches = load_all_matches(raw=False)
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches = add_basic_match_flags(matches)

    hist_matches = build_history_features(matches)

    train_max = pd.to_datetime(TRAIN_MAX_DATE)
    hist_train = hist_matches[hist_matches["date"] <= train_max].copy()

    # -----------------------
    # Fixtures: attach team state snapshot
    # -----------------------
    if PREDICTION_ASOF_DATE:
        asof_date = pd.to_datetime(PREDICTION_ASOF_DATE)
    else:
        asof_date = train_max + pd.Timedelta(days=1)

    team_state = build_team_state_asof(matches, asof_date=asof_date)

    fixtures = load_fixtures_2026(raw=False)
    fixtures = add_basic_match_flags(fixtures)

    # Home merge
    home_state = team_state.rename(columns={"team_name": "home_team"}).add_prefix("home_")
    home_state = home_state.rename(columns={"home_home_team": "home_team"})
    fixtures = fixtures.merge(home_state, on="home_team", how="left")

    # Away merge
    away_state = team_state.rename(columns={"team_name": "away_team"}).add_prefix("away_")
    away_state = away_state.rename(columns={"away_away_team": "away_team"})
    fixtures = fixtures.merge(away_state, on="away_team", how="left")

    # Align Elo naming with training feature names
    fixtures = fixtures.rename(columns={
        "home_elo_pre": "elo_home_pre",
        "away_elo_pre": "elo_away_pre",
    })

    fixtures = _safe_fill_column(fixtures, "elo_home_pre", 1500.0, cast=float)
    fixtures = _safe_fill_column(fixtures, "elo_away_pre", 1500.0, cast=float)
    fixtures["elo_diff_pre"] = fixtures["elo_home_pre"] - fixtures["elo_away_pre"]

    # Align rolling form naming with training expectations.
    # team_state columns are: gf_rollN, ga_rollN, hist_n
    # after merge: home_gf_rollN, home_ga_rollN, home_hist_n (and away_ equivalents)
    for c in [
        f"home_gf_roll{ROLLING_WINDOW}",
        f"home_ga_roll{ROLLING_WINDOW}",
        f"away_gf_roll{ROLLING_WINDOW}",
        f"away_ga_roll{ROLLING_WINDOW}",
    ]:
        fixtures = _safe_fill_column(fixtures, c, 0.0, cast=float)

    for c in ["home_hist_n", "away_hist_n"]:
        fixtures = _safe_fill_column(fixtures, c, 0, cast=int)

    hist_fixtures = fixtures.copy()

    return hist_train, hist_fixtures


# -------------------------------------------------------------------
# MAIN TEST / SAVE
# -------------------------------------------------------------------

def test_match_features_two_model() -> None:
    """
    Smoke test that:
    - Builds both model tables
    - Prints shapes
    - Saves outputs to data/processed/
    """
    player_train, player_fixtures = build_player_model_tables()
    hist_train, hist_fixtures = build_history_model_tables()

    print("player_train:", player_train.shape)
    print("player_fixtures:", player_fixtures.shape)
    print("history_train:", hist_train.shape)
    print("history_fixtures:", hist_fixtures.shape)

    player_train.to_csv(PROCESSED_PATHS["model_table_player_train"], index=False)
    player_fixtures.to_csv(PROCESSED_PATHS["model_table_player_fixtures"], index=False)
    hist_train.to_csv(PROCESSED_PATHS["model_table_history_train"], index=False)
    hist_fixtures.to_csv(PROCESSED_PATHS["model_table_history_fixtures"], index=False)

    print("Saved player train:", PROCESSED_PATHS["model_table_player_train"])
    print("Saved player fixtures:", PROCESSED_PATHS["model_table_player_fixtures"])
    print("Saved history train:", PROCESSED_PATHS["model_table_history_train"])
    print("Saved history fixtures:", PROCESSED_PATHS["model_table_history_fixtures"])


if __name__ == "__main__":
    test_match_features_two_model()
