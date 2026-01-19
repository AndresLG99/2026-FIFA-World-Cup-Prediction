# This file simulates the 2026 qualification playoffs and resolves placeholder teams in the WC fixtures.

"""
What simulate_playoffs.py does:

- Predicts the play-in bracket (UEFA + Inter-confederation playoffs) and converts:
  - "UEFA Winner A/B/C/D"
  - "IC Winner 1/2"
  into real team names.

Why we do this:
- The WC fixture list contains placeholders the pipeline can't treat as real teams.
- Resolving placeholders first improves prediction precision (real team merges -> better features).

Notes / limitations:
- This script does not simulate extra time / penalties (ties resolved by rule).
- For more realism later, replace deterministic winner rule with Poisson sampling.
"""

import os
import pandas as pd
from joblib import load

from config import (
    DATA_PATHS,
    PROCESSED_PATHS,
    MODELS_DIR,
    MIN_PLAYER_MINUTES,
    ROLLING_WINDOW,
    PREDICTION_ASOF_DATE,
)

from src.data_loader import load_all_matches, load_fixtures_2026, load_player_seasons
from src.player_features import build_players_master
from src.team_features import build_team_features
from src.code_mapping import load_nation_code_map
from src.history_features import build_team_state_asof


# -------------------------------------------------------------------
# PLAYOFF BRACKET DEFINITION
# -------------------------------------------------------------------

def load_playoff_bracket() -> pd.DataFrame:
    """
    Define the playoff bracket (as per your diagram).

    Output columns:
    - game_id, home_team, away_team, slot_name
    where slot_name is the placeholder to replace in WC fixtures.
    """
    rows = [
        (1, "Italy", "Northern Ireland", None),
        (2, "Wales", "Bosnia and Herzegovina", None),
        (3, "Ukraine", "Sweden", None),
        (4, "Poland", "Albania", None),
        (5, "Turkey", "Romania", None),
        (6, "Slovakia", "Kosovo", None),
        (7, "Denmark", "North Macedonia", None),
        (8, "Czech Republic", "Republic of Ireland", None),
        (9, "Winner Game 1", "Winner Game 2", "UEFA Winner A"),
        (10, "Winner Game 3", "Winner Game 4", "UEFA Winner B"),
        (11, "Winner Game 5", "Winner Game 6", "UEFA Winner C"),
        (12, "Winner Game 7", "Winner Game 8", "UEFA Winner D"),
        (13, "New Caledonia", "Jamaica", None),
        (14, "DR Congo", "Winner Game 13", "IC Winner 1"),
        (15, "Bolivia", "Suriname", None),
        (16, "Iraq", "Winner Game 15", "IC Winner 2"),
    ]
    return pd.DataFrame(rows, columns=["game_id", "home_team", "away_team", "slot_name"])


def resolve_winner_placeholders(team_name: str, winners: dict[int, str]) -> str:
    """
    Convert strings like "Winner Game 7" into the actual winner team name.
    """
    if isinstance(team_name, str) and team_name.startswith("Winner Game "):
        ref = int(team_name.replace("Winner Game ", "").strip())
        return winners[ref]
    return team_name


# -------------------------------------------------------------------
# BASIC MATCH FLAGS (CONSISTENT WITH match_features.py)
# -------------------------------------------------------------------

def add_basic_match_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple match-level context features.
    For playoffs, assume neutral-site by default.
    """
    out = df.copy()
    out["neutral"] = True
    out["country"] = "Neutral"
    out["is_neutral"] = 1
    out["is_home_host"] = 0
    out["tournament"] = "Playoffs"
    out["date"] = pd.to_datetime(PREDICTION_ASOF_DATE)
    return out


# -------------------------------------------------------------------
# FEATURE BUILDING (PLAYER + HISTORY)
# -------------------------------------------------------------------

def build_player_team_features_table() -> pd.DataFrame:
    """
    Build team-level features from the club player stats.
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
    """
    df = matches_df.copy()
    tf = team_features_df.copy()

    home_tf = tf.rename(columns={"team_name": "home_team"}).add_prefix("home_")
    home_tf = home_tf.rename(columns={"home_home_team": "home_team"})
    df = df.merge(home_tf, on="home_team", how="left")

    away_tf = tf.rename(columns={"team_name": "away_team"}).add_prefix("away_")
    away_tf = away_tf.rename(columns={"away_away_team": "away_team"})
    df = df.merge(away_tf, on="away_team", how="left")

    df["home_has_player_features"] = (~df["home_xg_90_calc"].isna()).astype(int)
    df["away_has_player_features"] = (~df["away_xg_90_calc"].isna()).astype(int)
    return df


def merge_history_team_state(fixtures_df: pd.DataFrame, team_state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge history "team state" features onto fixtures twice:
    - home team -> home_*
    - away team -> away_*

    IMPORTANT:
    - Rename home_elo_pre/away_elo_pre -> elo_home_pre/elo_away_pre
    - Add elo_diff_pre
    - Fill rolling features and hist_n for missing history teams
    """
    df = fixtures_df.copy()
    ts = team_state_df.copy()

    home_state = ts.rename(columns={"team_name": "home_team"}).add_prefix("home_")
    home_state = home_state.rename(columns={"home_home_team": "home_team"})
    df = df.merge(home_state, on="home_team", how="left")

    away_state = ts.rename(columns={"team_name": "away_team"}).add_prefix("away_")
    away_state = away_state.rename(columns={"away_away_team": "away_team"})
    df = df.merge(away_state, on="away_team", how="left")

    df = df.rename(columns={
        "home_elo_pre": "elo_home_pre",
        "away_elo_pre": "elo_away_pre",
    })

    df["elo_home_pre"] = df["elo_home_pre"].fillna(1500.0)
    df["elo_away_pre"] = df["elo_away_pre"].fillna(1500.0)
    df["elo_diff_pre"] = df["elo_home_pre"] - df["elo_away_pre"]

    df[f"home_gf_roll{ROLLING_WINDOW}"] = df[f"home_gf_roll{ROLLING_WINDOW}"].fillna(0.0)
    df[f"home_ga_roll{ROLLING_WINDOW}"] = df[f"home_ga_roll{ROLLING_WINDOW}"].fillna(0.0)
    df[f"away_gf_roll{ROLLING_WINDOW}"] = df[f"away_gf_roll{ROLLING_WINDOW}"].fillna(0.0)
    df[f"away_ga_roll{ROLLING_WINDOW}"] = df[f"away_ga_roll{ROLLING_WINDOW}"].fillna(0.0)

    df["home_hist_n"] = df["home_hist_n"].fillna(0).astype(int)
    df["away_hist_n"] = df["away_hist_n"].fillna(0).astype(int)

    return df


# -------------------------------------------------------------------
# MODEL LOADING + PREDICTION HELPERS
# -------------------------------------------------------------------

def load_models_and_feature_cols() -> dict:
    """
    Load the trained models and feature column lists from /models.
    """
    out = {
        "player_home_model": load(os.path.join(MODELS_DIR, "player_home_goals.joblib")),
        "player_away_model": load(os.path.join(MODELS_DIR, "player_away_goals.joblib")),
        "history_home_model": load(os.path.join(MODELS_DIR, "history_home_goals.joblib")),
        "history_away_model": load(os.path.join(MODELS_DIR, "history_away_goals.joblib")),
        "player_feature_cols": pd.read_csv(os.path.join(MODELS_DIR, "player_feature_cols.csv"), header=None)[0].tolist(),
        "history_feature_cols": pd.read_csv(os.path.join(MODELS_DIR, "history_feature_cols.csv"), header=None)[0].tolist(),
    }
    return out


def pick_winner_deterministic(home_team: str, away_team: str, lam_home: float, lam_away: float) -> str:
    """
    Deterministic winner rule (stable for a placeholder-resolver).
    """
    return home_team if lam_home >= lam_away else away_team


def predict_expected_goals_one_match(one_match_df: pd.DataFrame, assets: dict, feature_source: str) -> tuple[float, float]:
    """
    Predict expected goals for a single match using either PLAYER or HISTORY assets.

    Parameters:
    - one_match_df: single-row DataFrame containing features
    - assets: models + feature column lists
    - feature_source: "PLAYER" or "HISTORY"
    """
    if feature_source == "PLAYER":
        X = one_match_df.reindex(columns=assets["player_feature_cols"], fill_value=0.0).astype(float)
        lam_home = float(assets["player_home_model"].predict(X)[0])
        lam_away = float(assets["player_away_model"].predict(X)[0])
        return lam_home, lam_away

    X = one_match_df.reindex(columns=assets["history_feature_cols"], fill_value=0.0).astype(float)
    lam_home = float(assets["history_home_model"].predict(X)[0])
    lam_away = float(assets["history_away_model"].predict(X)[0])
    return lam_home, lam_away


# -------------------------------------------------------------------
# MAIN BRACKET SIMULATION
# -------------------------------------------------------------------

def simulate_playoffs_and_build_slot_map() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Simulate the bracket sequentially and produce:
    1) A predictions table for all playoff games
    2) A slot_map dict: placeholder -> winning_team_name
    """
    assets = load_models_and_feature_cols()

    team_features = build_player_team_features_table()

    matches = load_all_matches(raw=False).dropna(subset=["home_score", "away_score"]).copy()
    asof_date = pd.to_datetime(PREDICTION_ASOF_DATE)
    team_state = build_team_state_asof(matches, asof_date=asof_date)

    bracket = add_basic_match_flags(load_playoff_bracket())

    winners: dict[int, str] = {}
    results: list[dict] = []

    for _, r in bracket.sort_values("game_id").iterrows():
        gid = int(r["game_id"])

        home_team = resolve_winner_placeholders(r["home_team"], winners)
        away_team = resolve_winner_placeholders(r["away_team"], winners)

        one = pd.DataFrame([{
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "slot_name": r["slot_name"],
            "date": r["date"],
            "tournament": r["tournament"],
            "country": r["country"],
            "neutral": r["neutral"],
            "is_neutral": r["is_neutral"],
            "is_home_host": r["is_home_host"],
        }])

        one_player = merge_player_team_features(one, team_features)
        one_hist = merge_history_team_state(one, team_state)

        use_player = (
            int(one_player.loc[0, "home_has_player_features"]) == 1
            and int(one_player.loc[0, "away_has_player_features"]) == 1
        )

        if use_player:
            lam_h, lam_a = predict_expected_goals_one_match(one_player, assets, feature_source="PLAYER")
            model_used = "PLAYER"
        else:
            lam_h, lam_a = predict_expected_goals_one_match(one_hist, assets, feature_source="HISTORY")
            model_used = "HISTORY"

        winner = pick_winner_deterministic(home_team, away_team, lam_h, lam_a)
        winners[gid] = winner

        results.append({
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "pred_home_goals": lam_h,
            "pred_away_goals": lam_a,
            "model_used": model_used,
            "winner": winner,
            "slot_name": r["slot_name"],
        })

    results_df = pd.DataFrame(results)

    slot_map = (
        results_df.dropna(subset=["slot_name"])
        .set_index("slot_name")["winner"]
        .to_dict()
    )

    return results_df, slot_map


def resolve_wc_fixtures(slot_map: dict[str, str]) -> pd.DataFrame:
    """
    Replace placeholder team names in the WC fixtures using slot_map.
    """
    fixtures = load_fixtures_2026(raw=False).copy()
    for slot, team in slot_map.items():
        fixtures["home_team"] = fixtures["home_team"].replace(slot, team)
        fixtures["away_team"] = fixtures["away_team"].replace(slot, team)
    return fixtures


# -------------------------------------------------------------------
# MAIN TEST / SAVE
# -------------------------------------------------------------------

def test_simulate_playoffs() -> None:
    """
    Smoke test that:
    - Simulates the playoffs
    - Saves playoff predictions
    - Saves resolved WC fixtures with placeholders replaced
    """
    playoff_predictions, slot_map = simulate_playoffs_and_build_slot_map()
    fixtures_resolved = resolve_wc_fixtures(slot_map)

    playoff_predictions.to_csv(PROCESSED_PATHS["playoff_predictions"], index=False)
    fixtures_resolved.to_csv(PROCESSED_PATHS["fixtures_2026_resolved"], index=False)

    print("Saved playoff predictions:", PROCESSED_PATHS["playoff_predictions"])
    print("Saved resolved fixtures:", PROCESSED_PATHS["fixtures_2026_resolved"])
    print("Resolved slots:", slot_map)


if __name__ == "__main__":
    test_simulate_playoffs()