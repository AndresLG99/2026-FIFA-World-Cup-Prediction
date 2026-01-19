# Simulate the full tournament (group stage -> knockout bracket) using the TWO-MODEL system.

"""
What simulate_tournament.py does:

- Loads the 4 saved models from /models:
  - player_home_goals.joblib, player_away_goals.joblib
  - history_home_goals.joblib, history_away_goals.joblib
- Builds reusable resources once:
  - player team features (from club stats)
  - history team_state snapshot as-of PREDICTION_ASOF_DATE
- Simulates:
  1) Group stage (72 matches)
  2) Group standings and qualifiers (top 2 + best 8 third-place)
  3) Knockout bracket (R32 -> FINAL), resolving placeholders

Outputs (saved using PROCESSED_PATHS):
- tournament_group_matches
- tournament_group_table
- tournament_knockout_matches
- tournament_summary
"""

import os
import re
import numpy as np
import pandas as pd
from joblib import load

from config import (
    MODELS_DIR,
    DATA_PATHS,
    PROCESSED_PATHS,
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
# CONSTANTS
# -------------------------------------------------------------------

RNG_SEED = 42  # overwritten by run_monte_carlo.py per simulation


# -------------------------------------------------------------------
# HELPERS (FEATURE ALIGNMENT + ROUTING)
# -------------------------------------------------------------------

def load_feature_list(path: str) -> list[str]:
    cols = pd.read_csv(path, header=None)[0].astype(str).tolist()
    return cols


def safe_align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[feature_cols].astype(float)


def add_basic_match_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_neutral"] = out["neutral"].astype(int)
    out["is_home_host"] = (out["home_team"] == out["country"]).astype(int)
    return out


def build_player_team_features_table() -> pd.DataFrame:
    players = load_player_seasons()
    players_master = build_players_master(players, min_minutes=MIN_PLAYER_MINUTES)
    nation_map = load_nation_code_map(DATA_PATHS["nation_code_map"])
    team_features = build_team_features(players_master, nation_map)
    return team_features


def merge_player_team_features(matches_df: pd.DataFrame, team_features_df: pd.DataFrame) -> pd.DataFrame:
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
    df = fixtures_df.copy()
    ts = team_state_df.copy()

    home_state = ts.rename(columns={"team_name": "home_team"}).add_prefix("home_")
    home_state = home_state.rename(columns={"home_home_team": "home_team"})
    df = df.merge(home_state, on="home_team", how="left")

    away_state = ts.rename(columns={"team_name": "away_team"}).add_prefix("away_")
    away_state = away_state.rename(columns={"away_away_team": "away_team"})
    df = df.merge(away_state, on="away_team", how="left")

    df = df.rename(columns={"home_elo_pre": "elo_home_pre", "away_elo_pre": "elo_away_pre"})

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


def predict_expected_goals_one_match(one_player: pd.DataFrame, one_hist: pd.DataFrame, assets: dict) -> tuple[float, float, str]:
    use_player = (
        int(one_player.loc[0, "home_has_player_features"]) == 1
        and int(one_player.loc[0, "away_has_player_features"]) == 1
    )

    if use_player:
        Xp = safe_align_features(one_player, assets["player_feature_cols"])
        lam_h = float(assets["player_home_model"].predict(Xp)[0])
        lam_a = float(assets["player_away_model"].predict(Xp)[0])
        return lam_h, lam_a, "PLAYER"

    Xh = safe_align_features(one_hist, assets["history_feature_cols"])
    lam_h = float(assets["history_home_model"].predict(Xh)[0])
    lam_a = float(assets["history_away_model"].predict(Xh)[0])
    return lam_h, lam_a, "HISTORY"


def sample_score(rng: np.random.Generator, lam_home: float, lam_away: float) -> tuple[int, int]:
    gh = int(rng.poisson(lam_home))
    ga = int(rng.poisson(lam_away))
    return gh, ga


def decide_knockout_winner(
    rng: np.random.Generator,
    home_team: str,
    away_team: str,
    gh: int,
    ga: int,
    lam_h: float,
    lam_a: float,
) -> tuple[str, str, str]:
    if gh > ga:
        return home_team, away_team, "REG"
    if ga > gh:
        return away_team, home_team, "REG"
    if lam_h > lam_a:
        return home_team, away_team, "ET/PEN"
    if lam_a > lam_h:
        return away_team, home_team, "ET/PEN"
    return (home_team, away_team, "ET/PEN") if int(rng.integers(0, 2)) == 0 else (away_team, home_team, "ET/PEN")


# -------------------------------------------------------------------
# GROUP INFERENCE + TABLES
# -------------------------------------------------------------------

def infer_groups_from_group_stage(group_fixtures: pd.DataFrame) -> dict[str, str]:
    teams = pd.unique(pd.concat([group_fixtures["home_team"], group_fixtures["away_team"]], ignore_index=True))
    parent = {t: t for t in teams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in group_fixtures.iterrows():
        union(r["home_team"], r["away_team"])

    comps: dict[str, list[str]] = {}
    for t in teams:
        comps.setdefault(find(t), []).append(t)

    first_pos: dict[str, int] = {}
    for i, r in group_fixtures.reset_index(drop=True).iterrows():
        for t in (r["home_team"], r["away_team"]):
            first_pos.setdefault(find(t), i)

    comp_list = sorted(comps.items(), key=lambda kv: first_pos.get(kv[0], 10**9))
    letters = list("ABCDEFGHIJKL")

    team_to_group: dict[str, str] = {}
    for idx, (_, comp_teams) in enumerate(comp_list):
        g = letters[idx]
        for t in sorted(comp_teams):
            team_to_group[t] = g

    return team_to_group


def compute_group_table(group_matches: pd.DataFrame, team_to_group: dict[str, str]) -> pd.DataFrame:
    rows = []
    for team, grp in team_to_group.items():
        rows.append({"group": grp, "team": team, "mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0})

    tab = pd.DataFrame(rows).set_index(["group", "team"])

    for _, r in group_matches.iterrows():
        g = team_to_group[r["home_team"]]
        ht, at = r["home_team"], r["away_team"]
        gh, ga = int(r["home_score_sim"]), int(r["away_score_sim"])

        tab.loc[(g, ht), "mp"] += 1
        tab.loc[(g, at), "mp"] += 1

        tab.loc[(g, ht), "gf"] += gh
        tab.loc[(g, ht), "ga"] += ga
        tab.loc[(g, at), "gf"] += ga
        tab.loc[(g, at), "ga"] += gh

        if gh > ga:
            tab.loc[(g, ht), "w"] += 1
            tab.loc[(g, at), "l"] += 1
            tab.loc[(g, ht), "pts"] += 3
        elif ga > gh:
            tab.loc[(g, at), "w"] += 1
            tab.loc[(g, ht), "l"] += 1
            tab.loc[(g, at), "pts"] += 3
        else:
            tab.loc[(g, ht), "d"] += 1
            tab.loc[(g, at), "d"] += 1
            tab.loc[(g, ht), "pts"] += 1
            tab.loc[(g, at), "pts"] += 1

    tab["gd"] = tab["gf"] - tab["ga"]
    tab = tab.reset_index()
    tab = tab.sort_values(by=["group", "pts", "gd", "gf", "team"], ascending=[True, False, False, False, True]).copy()
    tab["rank_in_group"] = tab.groupby("group").cumcount() + 1
    return tab


def select_qualifiers(group_table: pd.DataFrame) -> dict:
    winners = group_table[group_table["rank_in_group"] == 1].set_index("group")["team"].to_dict()
    runners = group_table[group_table["rank_in_group"] == 2].set_index("group")["team"].to_dict()

    third = group_table[group_table["rank_in_group"] == 3].copy()
    third = third.sort_values(by=["pts", "gd", "gf", "team"], ascending=[False, False, False, True]).copy()
    third["third_rank_overall"] = range(1, len(third) + 1)

    third_qual = third.head(8).copy()
    third_by_group = third_qual.set_index("group")["team"].to_dict()

    return {
        "winners": winners,
        "runners": runners,
        "third_qualifiers": third_qual,
        "third_by_group": third_by_group,
    }


# -------------------------------------------------------------------
# PLACEHOLDER RESOLUTION
# -------------------------------------------------------------------

def resolve_group_placeholder_simple(name: str, qualifiers: dict) -> str:
    if not isinstance(name, str):
        return name

    m = re.match(r"^Group ([A-L]) Winners$", name)
    if m:
        return qualifiers["winners"][m.group(1)]

    m = re.match(r"^Group ([A-L]) Runners Up$", name)
    if m:
        return qualifiers["runners"][m.group(1)]

    return name


def resolve_match_placeholder(name: str, results: dict[int, dict]) -> str:
    if not isinstance(name, str):
        return name

    m = re.match(r"^Match (\d+) Winner$", name)
    if m:
        mid = int(m.group(1))
        return results[mid]["winner"]

    m = re.match(r"^Match (\d+) Loser$", name)
    if m:
        mid = int(m.group(1))
        return results[mid]["loser"]

    return name


def parse_third_place_placeholder(name: str) -> list[str]:
    m = re.match(r"^Group ([A-L/]+) 3rd Place$", name)
    if not m:
        return []
    raw = m.group(1)
    return [c for c in raw if c in "ABCDEFGHIJKL"]


def assign_third_place_slots(r32_fixtures: pd.DataFrame, third_qualifiers: pd.DataFrame) -> dict[tuple[int, str], str]:
    teams = third_qualifiers[["group", "team", "third_rank_overall"]].to_dict(orient="records")

    slots = []
    for _, r in r32_fixtures.iterrows():
        mid = int(r["match_no"])
        if isinstance(r["home_team"], str) and "3rd Place" in r["home_team"]:
            slots.append({"match_no": mid, "side": "home", "allowed": parse_third_place_placeholder(r["home_team"])})
        if isinstance(r["away_team"], str) and "3rd Place" in r["away_team"]:
            slots.append({"match_no": mid, "side": "away", "allowed": parse_third_place_placeholder(r["away_team"])})

    slots = sorted(slots, key=lambda s: len(s["allowed"]))
    teams = sorted(teams, key=lambda t: int(t["third_rank_overall"]))

    used: set[str] = set()
    assignment: dict[tuple[int, str], str] = {}

    def backtrack(i: int) -> bool:
        if i >= len(slots):
            return True

        slot = slots[i]
        allowed = set(slot["allowed"])
        candidates = [t for t in teams if (t["team"] not in used) and (t["group"] in allowed)]

        for t in candidates:
            used.add(t["team"])
            assignment[(slot["match_no"], slot["side"])] = t["team"]
            if backtrack(i + 1):
                return True
            used.remove(t["team"])
            assignment.pop((slot["match_no"], slot["side"]), None)

        return False

    if not backtrack(0):
        raise ValueError("No valid assignment exists for third-place slots given this set of qualified third-place teams.")

    return assignment


def stage_from_match_no(match_no: int) -> str:
    if 1 <= match_no <= 72:
        return "GROUP"
    if 73 <= match_no <= 88:
        return "R32"
    if 89 <= match_no <= 96:
        return "R16"
    if 97 <= match_no <= 100:
        return "QF"
    if 101 <= match_no <= 102:
        return "SF"
    if match_no == 103:
        return "3P"
    if match_no == 104:
        return "FINAL"
    return "UNKNOWN"


# -------------------------------------------------------------------
# MAIN SIMULATION
# -------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    # Load models + feature lists
    assets = {
        "player_home_model": load(os.path.join(MODELS_DIR, "player_home_goals.joblib")),
        "player_away_model": load(os.path.join(MODELS_DIR, "player_away_goals.joblib")),
        "history_home_model": load(os.path.join(MODELS_DIR, "history_home_goals.joblib")),
        "history_away_model": load(os.path.join(MODELS_DIR, "history_away_goals.joblib")),
        "player_feature_cols": load_feature_list(os.path.join(MODELS_DIR, "player_feature_cols.csv")),
        "history_feature_cols": load_feature_list(os.path.join(MODELS_DIR, "history_feature_cols.csv")),
    }

    # Reusable resources
    team_features = build_player_team_features_table()

    matches_hist = load_all_matches(raw=False).dropna(subset=["home_score", "away_score"]).copy()
    asof_date = pd.to_datetime(PREDICTION_ASOF_DATE)
    team_state = build_team_state_asof(matches_hist, asof_date=asof_date)

    # Load fixtures
    fixtures = load_fixtures_2026(raw=False).copy().reset_index(drop=True)
    fixtures["match_no"] = fixtures.index + 1
    fixtures["stage"] = fixtures["match_no"].apply(stage_from_match_no)
    fixtures = add_basic_match_flags(fixtures)

    # Group-stage fixtures are the real-team rows (exclude template placeholders)
    is_template_placeholder = (
        fixtures["home_team"].astype(str).str.startswith(("Group ", "Match "))
        | fixtures["away_team"].astype(str).str.startswith(("Group ", "Match "))
        | fixtures["home_team"].astype(str).str.contains("3rd Place", na=False)
        | fixtures["away_team"].astype(str).str.contains("3rd Place", na=False)
        | fixtures["home_team"].astype(str).str.contains("/", na=False)
        | fixtures["away_team"].astype(str).str.contains("/", na=False)
    )

    group_fx = fixtures[~is_template_placeholder].copy()
    team_to_group = infer_groups_from_group_stage(group_fx)

    # -----------------------
    # GROUP SIMULATION
    # -----------------------
    group_rows = []

    for _, r in group_fx.iterrows():
        one = pd.DataFrame([{
            "date": r["date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "tournament": r["tournament"],
            "country": r["country"],
            "neutral": r["neutral"],
            "is_neutral": r["is_neutral"],
            "is_home_host": r["is_home_host"],
        }])

        one_p = merge_player_team_features(one, team_features)
        one_h = merge_history_team_state(one, team_state)

        lam_h, lam_a, model_used = predict_expected_goals_one_match(one_p, one_h, assets)
        gh, ga = sample_score(rng, lam_h, lam_a)

        group_rows.append({
            "match_no": int(r["match_no"]),
            "stage": "GROUP",
            "group": team_to_group[r["home_team"]],
            "date": r["date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "pred_home_goals": lam_h,
            "pred_away_goals": lam_a,
            "home_score_sim": gh,
            "away_score_sim": ga,
            "model_used": model_used,
        })

    group_results = pd.DataFrame(group_rows)
    group_table = compute_group_table(group_results, team_to_group)
    qualifiers = select_qualifiers(group_table)

    # -----------------------
    # KNOCKOUT SIMULATION
    # -----------------------
    results_by_match: dict[int, dict] = {}
    knockout_fx = fixtures[is_template_placeholder].copy().sort_values("match_no")

    r32_fx = knockout_fx[knockout_fx["stage"] == "R32"].copy()
    third_assignment = assign_third_place_slots(r32_fx, qualifiers["third_qualifiers"])

    knockout_rows = []

    for _, r in knockout_fx.iterrows():
        match_no = int(r["match_no"])
        stage = str(r["stage"])

        home = r["home_team"]
        away = r["away_team"]

        home = resolve_group_placeholder_simple(home, qualifiers)
        away = resolve_group_placeholder_simple(away, qualifiers)

        if stage == "R32":
            if isinstance(home, str) and "3rd Place" in home:
                home = third_assignment[(match_no, "home")]
            if isinstance(away, str) and "3rd Place" in away:
                away = third_assignment[(match_no, "away")]

        home = resolve_match_placeholder(home, results_by_match)
        away = resolve_match_placeholder(away, results_by_match)

        one = pd.DataFrame([{
            "date": r["date"],
            "home_team": home,
            "away_team": away,
            "tournament": r["tournament"],
            "country": r["country"],
            "neutral": r["neutral"],
            "is_neutral": r["is_neutral"],
            "is_home_host": r["is_home_host"],
        }])

        one_p = merge_player_team_features(one, team_features)
        one_h = merge_history_team_state(one, team_state)

        lam_h, lam_a, model_used = predict_expected_goals_one_match(one_p, one_h, assets)
        gh, ga = sample_score(rng, lam_h, lam_a)

        winner, loser, win_method = decide_knockout_winner(rng, str(home), str(away), gh, ga, lam_h, lam_a)

        results_by_match[match_no] = {
            "home_team": home,
            "away_team": away,
            "home_score_sim": gh,
            "away_score_sim": ga,
            "winner": winner,
            "loser": loser,
        }

        knockout_rows.append({
            "match_no": match_no,
            "stage": stage,
            "date": r["date"],
            "home_team": home,
            "away_team": away,
            "pred_home_goals": lam_h,
            "pred_away_goals": lam_a,
            "home_score_sim": gh,
            "away_score_sim": ga,
            "winner": winner,
            "win_method": win_method,
            "model_used": model_used,
        })

    knockout_results = pd.DataFrame(knockout_rows).sort_values("match_no")

    # -----------------------
    # SAVE OUTPUTS (NEW CONFIG)
    # -----------------------
    group_results.to_csv(PROCESSED_PATHS["tournament_group_matches"], index=False)
    group_table.to_csv(PROCESSED_PATHS["tournament_group_table"], index=False)
    knockout_results.to_csv(PROCESSED_PATHS["tournament_knockout_matches"], index=False)

    champion_row = knockout_results[knockout_results["stage"] == "FINAL"].tail(1)
    champion = champion_row["winner"].iloc[0] if len(champion_row) else None

    summary = pd.DataFrame([{
        "rng_seed": RNG_SEED,
        "asof_date": PREDICTION_ASOF_DATE,
        "champion": champion,
    }])
    summary.to_csv(PROCESSED_PATHS["tournament_summary"], index=False)

    print("Saved group matches:", PROCESSED_PATHS["tournament_group_matches"])
    print("Saved group table:", PROCESSED_PATHS["tournament_group_table"])
    print("Saved knockout matches:", PROCESSED_PATHS["tournament_knockout_matches"])
    print("Saved summary:", PROCESSED_PATHS["tournament_summary"])
    print("Champion:", champion)


if __name__ == "__main__":
    main()