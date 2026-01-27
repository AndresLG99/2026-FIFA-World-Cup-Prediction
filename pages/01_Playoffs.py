from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from config import PROCESSED_PATHS


# -----------------------------
# helpers
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


def coerce_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def normalize_playoffs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your simulate_playoffs.py writes columns like:
    gameid, hometeam, awayteam, predhomegoals, predawaygoals, modelused, winner, slotname [file:728]
    This normalizes them to underscore style used in the app:
    game_id, home_team, away_team, pred_home_goals, pred_away_goals, model_used, winner, slot_name
    """
    rename_map = {
        "gameid": "game_id",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "predhomegoals": "pred_home_goals",
        "predawaygoals": "pred_away_goals",
        "modelused": "model_used",
        "slotname": "slot_name",
    }
    out = df.copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def render_match_row(
    home_team: str,
    away_team: str,
    pred_home_goals: float | None,
    pred_away_goals: float | None,
) -> None:
    # 7 cols: margins + (team, xg, vs, xg, team) + margin
    left_margin, c1, c2, c3, c4, c5, right_margin = st.columns(
        [0.6, 2.2, 1.0, 0.6, 1.0, 2.2, 0.6],
        gap="small",
    )

    home_xg = f"{float(pred_home_goals):.2f}" if pred_home_goals is not None and pd.notna(pred_home_goals) else ""
    away_xg = f"{float(pred_away_goals):.2f}" if pred_away_goals is not None and pd.notna(pred_away_goals) else ""

    with c1:
        st.markdown(f"<div style='text-align:left;'>{home_team}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div style='text-align:center; opacity:0.85;'>xG {home_xg}</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div style='text-align:center;'>vs</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"<div style='text-align:center; opacity:0.85;'>xG {away_xg}</div>", unsafe_allow_html=True)

    with c5:
        st.markdown(f"<div style='text-align:right;'>{away_team}</div>", unsafe_allow_html=True)



def render_meta_row(
    winner: str,
    model_used: str,
    pred_home_goals: float | None,
    pred_away_goals: float | None,
) -> None:
    xg_text = ""
    if pred_home_goals is not None and pred_away_goals is not None:
        if pd.notna(pred_home_goals) and pd.notna(pred_away_goals):
            xg_text = f" — xG: {float(pred_home_goals):.2f}–{float(pred_away_goals):.2f}"

    st.markdown(
        f"""
        <div style="text-align:center; opacity:0.85; font-size:0.95rem; margin-bottom:0.6rem;">
            Winner: <span style="font-weight:700;">{winner}</span> · Model: {model_used}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_resolved_slot_line(slot_name: str, team_name: str) -> None:
    # Center the block on the page, but left-align the text inside it.
    left_margin, content_col, right_margin = st.columns([2.0, 3.0, 2.0], gap="small")

    with content_col:
        st.markdown(
            f"""
            <div style="text-align:left; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                        font-size:1.05rem; line-height:1.6;">
              <span style="display:inline-block; width: 14ch; font-weight:800;">{slot_name}</span>
              <span style="display:inline-block; width: 3ch; text-align:center;">→</span>
              <span style="display:inline-block; width: 18ch;">{team_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# page config
# -----------------------------
st.set_page_config(
    page_title="Playoffs",
    layout="wide",
)

st.title("Playoffs")

st.markdown(
    """
These are the qualification playoff games used to resolve placeholders like **UEFA Winner A–D** and **IC Winner 1–2** in the 2026 fixtures.
"""
)

# -----------------------------
# load data
# -----------------------------
playoff_predictions_path = PROCESSED_PATHS["playoff_predictions"]
df = safe_read_csv(playoff_predictions_path)
df = coerce_date(df, "date")
df = normalize_playoffs_columns(df)

required_cols = ["game_id", "home_team", "away_team", "winner", "model_used"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"`{playoff_predictions_path}` is missing required columns: {missing_cols}")
    st.stop()

df = df.copy()
df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
df = df.sort_values(["game_id"], na_position="last")

# -----------------------------
# resolved slots (UEFA Winner A–D, IC Winner 1–2)
# -----------------------------
if "slot_name" in df.columns:
    slot_df = df.dropna(subset=["slot_name"]).copy()
else:
    slot_df = pd.DataFrame()

if not slot_df.empty:
    st.subheader("Resolved slots", divider="gray")

    for _, r in slot_df.iterrows():
        render_resolved_slot_line(str(r["slot_name"]), str(r["winner"]))

st.subheader("Playoff games", divider="gray")

# -----------------------------
# render all games
# -----------------------------
for _, r in df.iterrows():
    game_id = r["game_id"]
    home_team = str(r["home_team"])
    away_team = str(r["away_team"])
    winner = str(r["winner"])
    model_used = str(r["model_used"])

    pred_home_goals = r["pred_home_goals"] if "pred_home_goals" in df.columns else None
    pred_away_goals = r["pred_away_goals"] if "pred_away_goals" in df.columns else None

    st.markdown(
        f"<div style='text-align:center; font-weight:800; margin-top:0.75rem;'>Game {game_id}</div>",
        unsafe_allow_html=True,
    )

    render_match_row(home_team, away_team, pred_home_goals, pred_away_goals)
    render_meta_row(winner, model_used, pred_home_goals, pred_away_goals)

st.caption("Source: processed predictions output generated by simulate_playoffs.py.")
