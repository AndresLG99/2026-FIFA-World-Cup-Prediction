from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from config import MODELS_DIR, PROCESSED_PATHS


# -------------------------
# helpers
# -------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


def load_feature_list(path: str) -> list[str]:
    return pd.read_csv(path, header=None)[0].astype(str).tolist()


def safe_align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[feature_cols].astype(float)


def poisson_win_probs(lam_home: float, lam_away: float, n: int = 20000, seed: int = 7) -> dict:
    """
    Estimate win/draw probabilities by sampling scorelines from Poisson(lam).
    Returns probabilities for home win / draw / away win.
    """
    rng = np.random.default_rng(seed)
    gh = rng.poisson(lam_home, size=n)
    ga = rng.poisson(lam_away, size=n)
    home_win = float(np.mean(gh > ga))
    draw = float(np.mean(gh == ga))
    away_win = float(np.mean(gh < ga))
    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def fmt(x: float, nd: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def pick_winner_from_probs(home_team: str, away_team: str, probs: dict) -> tuple[str, str]:
    """
    Decide winner for UI purposes.
    - If one side has higher win prob -> winner
    - If equal within tiny epsilon -> "Too close to call"
    """
    eps = 1e-6
    if probs["home_win"] > probs["away_win"] + eps:
        return home_team, "By win probability"
    if probs["away_win"] > probs["home_win"] + eps:
        return away_team, "By win probability"
    return "Too close to call", "Nearly equal win probability"


def extract_team_list(fx: pd.DataFrame) -> list[str]:
    return sorted(set(fx["home_team"].astype(str)).union(set(fx["away_team"].astype(str))))


def find_fixture_row(fx: pd.DataFrame, home: str, away: str, is_neutral: int) -> pd.DataFrame:
    """
    Find the single fixture row in fx matching home/away and is_neutral.
    """
    m = (
        (fx["home_team"].astype(str) == str(home))
        & (fx["away_team"].astype(str) == str(away))
        & (fx["is_neutral"].fillna(0).astype(int) == int(is_neutral))
    )
    return fx.loc[m].head(1).copy()


def build_feature_compare_table(one_player: pd.DataFrame, one_hist: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    side in {"home", "away"}.
    Shows a small curated set of features if present.
    """
    prefix = f"{side}_"

    def get(df: pd.DataFrame, col: str):
        c = prefix + col
        return df.iloc[0][c] if c in df.columns else pd.NA

    rows = []

    # PLAYER-derived (from match_features.py player team features) [file:739]
    for col, label in [
        ("xg_90_calc", "xG/90 (players)"),
        ("ga_90_calc", "GA/90 (players)"),
        ("shots_90_calc", "Shots/90 (players)"),
        ("sot_90_calc", "SoT/90 (players)"),
        ("n_players_used", "Players used (n)"),
    ]:
        v = get(one_player, col)
        if pd.notna(v):
            rows.append((label, float(v) if isinstance(v, (int, float, np.number)) else v))

    # HISTORY-derived (from match_features.py history team state) [file:739]
    # Note: in fixtures table they should be named like elo_home_pre in training,
    # but in fixtures created by match_features.py they are renamed to elohomepre/eloawaypre. [file:739]
    # We'll try both patterns.
    hist_map = [
        ("elohomepre", "Elo (history)"),
        ("eloawaypre", "Elo (history)"),
        ("home_gf_roll10", "GF last 10 (history)"),
        ("home_ga_roll10", "GA last 10 (history)"),
        ("away_gf_roll10", "GF last 10 (history)"),
        ("away_ga_roll10", "GA last 10 (history)"),
        ("home_hist_n", "History games (n)"),
        ("away_hist_n", "History games (n)"),
    ]

    # choose per side
    wanted = []
    if side == "home":
        wanted = ["elohomepre", "home_gf_roll10", "home_ga_roll10", "home_hist_n"]
    else:
        wanted = ["eloawaypre", "away_gf_roll10", "away_ga_roll10", "away_hist_n"]

    for col in wanted:
        if col in one_hist.columns:
            v = one_hist.iloc[0][col]
            if pd.notna(v):
                label = dict(hist_map).get(col, col)
                rows.append((label, float(v) if isinstance(v, (int, float, np.number)) else v))

    out = pd.DataFrame(rows, columns=["Feature", "Value"])
    if not out.empty:
        out["Value"] = out["Value"].map(lambda z: fmt(z, 3) if isinstance(z, (int, float, np.number)) else str(z))
    return out


# -------------------------
# page
# -------------------------
st.set_page_config(page_title="Simulation", layout="wide")
st.title("Simulation")

st.markdown(
    "Choose two teams and this page will display key input features plus a predicted matchup outcome."
)

# Load fixture feature tables produced by match_features.py / pipeline [file:739][file:727]
fx_player = safe_read_csv(PROCESSED_PATHS["model_table_player_fixtures"])
fx_hist = safe_read_csv(PROCESSED_PATHS["model_table_history_fixtures"])

# Ensure is_neutral exists
if "is_neutral" not in fx_player.columns and "neutral" in fx_player.columns:
    fx_player["is_neutral"] = fx_player["neutral"].astype(int)
if "is_neutral" not in fx_hist.columns and "neutral" in fx_hist.columns:
    fx_hist["is_neutral"] = fx_hist["neutral"].astype(int)

teams = extract_team_list(fx_player)

c1, c2, c3 = st.columns([2.0, 2.0, 1.2], gap="small")
with c1:
    team_a = st.selectbox("Team A", teams, index=0)
with c2:
    team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)
with c3:
    neutral_site = st.checkbox("Neutral site", value=True)

if team_a == team_b:
    st.warning("Please choose two different teams.")
    st.stop()

is_neutral = 1 if neutral_site else 0

# ---- find the correct feature rows for this pairing ----
one_player = find_fixture_row(fx_player, team_a, team_b, is_neutral)
one_hist = find_fixture_row(fx_hist, team_a, team_b, is_neutral)

# If the exact pairing is not in fixtures, try reverse pairing.
# If reverse exists, we will compute prediction for reverse and then swap back.
swapped = False
if one_player.empty or one_hist.empty:
    one_player_rev = find_fixture_row(fx_player, team_b, team_a, is_neutral)
    one_hist_rev = find_fixture_row(fx_hist, team_b, team_a, is_neutral)
    if (not one_player_rev.empty) and (not one_hist_rev.empty):
        one_player = one_player_rev
        one_hist = one_hist_rev
        swapped = True
    else:
        st.error(
            "This exact matchup is not present in the 2026 fixtures feature tables, "
            "so the page cannot fetch the correct engineered features for it.\n\n"
            "Pick a matchup that exists in the fixtures schedule, or we can upgrade this page "
            "to rebuild features on-the-fly (slower but supports any pairing)."
        )
        st.stop()

# If we used reversed rows, we must interpret them correctly:
# The row currently represents (home=team_b, away=team_a). We want (home=team_a, away=team_b).
if swapped:
    # swap perspective by swapping team labels and swapping predicted outputs after inference.
    home_team = team_b
    away_team = team_a
else:
    home_team = team_a
    away_team = team_b

# Determine routing: PLAYER only if both have player features [file:727][file:739]
home_has = int(one_player.iloc[0].get("home_has_player_features", 0))
away_has = int(one_player.iloc[0].get("away_has_player_features", 0))
use_player = (home_has == 1) and (away_has == 1)

# Load models + feature lists [file:727]
player_cols = load_feature_list(os.path.join(MODELS_DIR, "player_feature_cols.csv"))
hist_cols = load_feature_list(os.path.join(MODELS_DIR, "history_feature_cols.csv"))

player_home = load(os.path.join(MODELS_DIR, "player_home_goals.joblib"))
player_away = load(os.path.join(MODELS_DIR, "player_away_goals.joblib"))
hist_home = load(os.path.join(MODELS_DIR, "history_home_goals.joblib"))
hist_away = load(os.path.join(MODELS_DIR, "history_away_goals.joblib"))

# Predict expected goals for the row we actually have (home_team/away_team)
if use_player:
    X = safe_align_features(one_player, player_cols)
    lam_home = float(player_home.predict(X)[0])
    lam_away = float(player_away.predict(X)[0])
    model_used = "PLAYER"
else:
    X = safe_align_features(one_hist, hist_cols)
    lam_home = float(hist_home.predict(X)[0])
    lam_away = float(hist_away.predict(X)[0])
    model_used = "HISTORY"

# If we used reversed fixture row, swap the lambdas back so output is for (team_a vs team_b)
if swapped:
    lam_home, lam_away = lam_away, lam_home
    model_used = model_used  # unchanged

# Estimate win probabilities via Poisson sampling (removes "home wins ties" bias) [file:733]
probs = poisson_win_probs(lam_home, lam_away, n=15000, seed=11)
winner, win_note = pick_winner_from_probs(team_a, team_b, probs)

# ---- styled result banner ----
st.markdown(
    """
    <style>
      .sim-banner{
        margin: 14px 0 18px 0;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.16);
        background: linear-gradient(90deg, rgba(99,102,241,0.18), rgba(255,255,255,0.03));
        box-shadow: 0 10px 26px rgba(0,0,0,0.30);
        text-align: center;
      }
      .sim-small{ opacity: 0.80; font-size: 0.95rem; }
      .sim-scoreline{ font-size: 1.35rem; font-weight: 950; margin-top: 6px; }
      .sim-winner{ font-size: 2.0rem; font-weight: 1000; margin-top: 8px; }
      .sim-pill{
        display:inline-block;
        margin-top: 8px;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.04);
        font-size: 0.85rem;
        opacity: 0.9;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="sim-banner">
      <div class="sim-small">
        Expected goals (Poisson means) · Model used: <b>{model_used}</b> · Neutral: {bool(neutral_site)}
      </div>
      <div class="sim-scoreline">{team_a} {lam_home:.2f} — {lam_away:.2f} {team_b}</div>
      <div class="sim-winner">Winner: {winner}</div>
      <div class="sim-pill">{win_note} · P(win): {100*max(probs["home_win"], probs["away_win"]):.1f}% · P(draw): {100*probs["draw"]:.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- feature comparison ----
st.subheader("Feature comparison", divider="gray")

left, right = st.columns(2, gap="large")

with left:
    st.markdown(f"### {team_a}")
    # We want features for team_a as home; if swapped, team_a was away in the fetched row.
    side = "home" if not swapped else "away"
    comp_a = build_feature_compare_table(one_player, one_hist, side=side)
    st.dataframe(comp_a, use_container_width=True, hide_index=True)

with right:
    st.markdown(f"### {team_b}")
    side = "away" if not swapped else "home"
    comp_b = build_feature_compare_table(one_player, one_hist, side=side)
    st.dataframe(comp_b, use_container_width=True, hide_index=True)

with st.expander("Debug details"):
    st.write(
        {
            "neutral_site": neutral_site,
            "is_neutral": is_neutral,
            "used_fixture_row_swapped": swapped,
            "home_has_player_features": home_has,
            "away_has_player_features": away_has,
            "model_used": model_used,
            "lam_home(team_a)": lam_home,
            "lam_away(team_b)": lam_away,
            "p_home_win(team_a)": probs["home_win"],
            "p_draw": probs["draw"],
            "p_away_win(team_b)": probs["away_win"],
        }
    )

st.caption("This page pulls engineered features from the prebuilt 2026 fixtures tables and uses the same PLAYER/HISTORY routing.")