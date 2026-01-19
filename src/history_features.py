# This module builds pre-match Elo + rolling “form” features from all_matches.csv,
# and it can also generate team states to attach to fixtures.

"""
What history_features.py does:

- Creates time-correct features: each match’s Elo/form values are computed before the match is played (avoids leakage).
- Uses a standard Elo expected-score formula.
- Builds "team states as-of a date" for attaching to fixtures (playoffs and tournament simulation).
"""

from collections import deque

import pandas as pd

from config import (
    ELO_BASE,
    ELO_K,
    ELO_HOME_ADV,
    ROLLING_WINDOW,
    TRAIN_MAX_DATE,
)

from src.data_loader import load_all_matches


def expected_score(ra: float, rb: float) -> float:
    """
    Elo expected score:

    E_A = 1 / (1 + 10^((R_B - R_A)/400))

    Impact:
    - Translates rating difference into an expected match result for Elo updates.
    """
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def build_history_features(
    matches_df: pd.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
    elo_k: float = ELO_K,
    elo_base: float = ELO_BASE,
    elo_home_adv: float = ELO_HOME_ADV,
) -> pd.DataFrame:
    """
    Build *pre-match* history features for each match.

    Key rule (anti-leakage):
    - Features for match i are computed using only matches < i (earlier in time).

    Outputs:
    - elo_home_pre, elo_away_pre, elo_diff_pre
    - Rolling means over last N matches (goals for/against):
      home_gf_rollN, home_ga_rollN, away_gf_rollN, away_ga_rollN
    - History counts:
      home_hist_n, away_hist_n

    Impact:
    - Creates a fallback feature set that covers many more teams than player stats.
    - Prevents “future” information from leaking into training.
    """
    df = matches_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Column names (single source of truth)
    gf_home_col = f"home_gf_roll{rolling_window}"
    ga_home_col = f"home_ga_roll{rolling_window}"
    gf_away_col = f"away_gf_roll{rolling_window}"
    ga_away_col = f"away_ga_roll{rolling_window}"

    # Team states
    elo: dict[str, float] = {}
    gf_hist: dict[str, deque] = {}
    ga_hist: dict[str, deque] = {}

    def get_team_state(team: str) -> tuple[float, float, float, int]:
        if team not in elo:
            elo[team] = float(elo_base)
        if team not in gf_hist:
            gf_hist[team] = deque(maxlen=rolling_window)
        if team not in ga_hist:
            ga_hist[team] = deque(maxlen=rolling_window)

        gf_mean = (sum(gf_hist[team]) / len(gf_hist[team])) if len(gf_hist[team]) else 0.0
        ga_mean = (sum(ga_hist[team]) / len(ga_hist[team])) if len(ga_hist[team]) else 0.0
        return float(elo[team]), float(gf_mean), float(ga_mean), int(len(gf_hist[team]))

    # Pre-match features we will fill
    elo_home_pre, elo_away_pre, elo_diff_pre = [], [], []
    home_gf, home_ga, away_gf, away_ga = [], [], [], []
    home_hist_n, away_hist_n = [], []

    for _, r in df.iterrows():
        ht = str(r["home_team"])
        at = str(r["away_team"])
        neutral = bool(r["neutral"])

        elo_h, gf_h, ga_h, n_h = get_team_state(ht)
        elo_a, gf_a, ga_a, n_a = get_team_state(at)

        # Apply a home-advantage offset only when not neutral
        elo_h_adj = elo_h + (0.0 if neutral else float(elo_home_adv))
        elo_a_adj = elo_a

        elo_home_pre.append(elo_h_adj)
        elo_away_pre.append(elo_a_adj)
        elo_diff_pre.append(elo_h_adj - elo_a_adj)

        home_gf.append(gf_h)
        home_ga.append(ga_h)
        away_gf.append(gf_a)
        away_ga.append(ga_a)

        home_hist_n.append(n_h)
        away_hist_n.append(n_a)

        # Update states AFTER we record pre-match features
        hs = float(r["home_score"])
        aws = float(r["away_score"])

        # Elo update uses win/draw/loss result
        if hs > aws:
            s_h, s_a = 1.0, 0.0
        elif hs < aws:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        e_h = expected_score(elo_h_adj, elo_a_adj)
        e_a = 1.0 - e_h

        elo[ht] = elo_h + float(elo_k) * (s_h - e_h)
        elo[at] = elo_a + float(elo_k) * (s_a - e_a)

        # Rolling histories update (team-centric)
        gf_hist[ht].append(hs)
        ga_hist[ht].append(aws)
        gf_hist[at].append(aws)
        ga_hist[at].append(hs)

    out = df.copy()
    out["elo_home_pre"] = elo_home_pre
    out["elo_away_pre"] = elo_away_pre
    out["elo_diff_pre"] = elo_diff_pre

    out[gf_home_col] = home_gf
    out[ga_home_col] = home_ga
    out[gf_away_col] = away_gf
    out[ga_away_col] = away_ga

    out["home_hist_n"] = home_hist_n
    out["away_hist_n"] = away_hist_n

    return out


def build_team_state_asof(
    history_df: pd.DataFrame,
    asof_date: pd.Timestamp,
    rolling_window: int = ROLLING_WINDOW,
    elo_k: float = ELO_K,
    elo_base: float = ELO_BASE,
    elo_home_adv: float = ELO_HOME_ADV,
) -> pd.DataFrame:
    """
    Compute each team's Elo + rolling means using matches strictly before asof_date.

    Output columns:
    - team_name
    - elo_pre
    - gf_rollN, ga_rollN
    - hist_n

    Impact:
    - Produces the “last known state” features needed to score fixtures in the future.
    """
    df = history_df[history_df["date"] < asof_date].copy()
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}
    gf_hist: dict[str, deque] = {}
    ga_hist: dict[str, deque] = {}

    def init(team: str) -> None:
        if team not in elo:
            elo[team] = float(elo_base)
        if team not in gf_hist:
            gf_hist[team] = deque(maxlen=rolling_window)
        if team not in ga_hist:
            ga_hist[team] = deque(maxlen=rolling_window)

    for _, r in df.iterrows():
        ht = str(r["home_team"])
        at = str(r["away_team"])
        neutral = bool(r["neutral"])

        init(ht)
        init(at)

        elo_h = float(elo[ht])
        elo_a = float(elo[at])

        elo_h_adj = elo_h + (0.0 if neutral else float(elo_home_adv))
        elo_a_adj = elo_a

        hs = float(r["home_score"])
        aws = float(r["away_score"])

        if hs > aws:
            s_h, s_a = 1.0, 0.0
        elif hs < aws:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        e_h = expected_score(elo_h_adj, elo_a_adj)
        e_a = 1.0 - e_h

        elo[ht] = elo_h + float(elo_k) * (s_h - e_h)
        elo[at] = elo_a + float(elo_k) * (s_a - e_a)

        gf_hist[ht].append(hs)
        ga_hist[ht].append(aws)
        gf_hist[at].append(aws)
        ga_hist[at].append(hs)

    rows = []
    for team in sorted(elo.keys()):
        gf_mean = (sum(gf_hist[team]) / len(gf_hist[team])) if len(gf_hist[team]) else 0.0
        ga_mean = (sum(ga_hist[team]) / len(ga_hist[team])) if len(ga_hist[team]) else 0.0
        rows.append(
            {
                "team_name": team,
                "elo_pre": float(elo[team]),
                f"gf_roll{rolling_window}": float(gf_mean),
                f"ga_roll{rolling_window}": float(ga_mean),
                "hist_n": int(len(gf_hist[team])),
            }
        )

    return pd.DataFrame(rows)


def test_history_features() -> None:
    """
    Quick smoke test to verify:
    - Feature columns are created as expected
    - Team state snapshot builds without errors
    """
    matches = load_all_matches(raw=False)
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()

    hist = build_history_features(matches)
    print("history_features table:", hist.shape)
    print(hist[["date", "home_team", "away_team", "elo_diff_pre"]].head(5))

    cutoff = pd.to_datetime(TRAIN_MAX_DATE) + pd.Timedelta(days=1)
    team_state = build_team_state_asof(matches, asof_date=cutoff)

    print("team_state_asof:", team_state.shape)
    print(team_state.head(10))


if __name__ == "__main__":
    test_history_features()