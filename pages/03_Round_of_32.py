from __future__ import annotations

import os
import pandas as pd
import streamlit as st


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


def render_match_row_score(home_team: str, away_team: str, home_score: int | None, away_score: int | None) -> None:
    left_margin, c1, c2, c3, c4, c5, right_margin = st.columns(
        [0.6, 2.2, 0.8, 0.6, 0.8, 2.2, 0.6],
        gap="small",
    )

    hs = "" if home_score is None or pd.isna(home_score) else str(int(home_score))
    a_s = "" if away_score is None or pd.isna(away_score) else str(int(away_score))

    with c1:
        st.markdown(f"<div style='text-align:left;'>{home_team}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='text-align:center; opacity:0.9; font-weight:700;'>{hs}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div style='text-align:center;'>-</div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div style='text-align:center; opacity:0.9; font-weight:700;'>{a_s}</div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div style='text-align:right;'>{away_team}</div>", unsafe_allow_html=True)


st.set_page_config(page_title="Round of 32", layout="wide")
st.title("Round of 32")

knockout_path = os.path.join("data", "processed", "tournament_single_run", "tournament_knockout_matches.csv")
df = coerce_date(safe_read_csv(knockout_path), "date")
df = df[df["stage"] == "R32"].copy()  # [file:804]

if "match_no" in df.columns:
    df = df.sort_values(["match_no"])
elif "date" in df.columns:
    df = df.sort_values(["date"])

# ---- filters (same method) ----
f1, f2, f3 = st.columns([1.2, 1.2, 2.6], gap="small")

with f1:
    player_model = st.checkbox("PLAYER model", value=False)
with f2:
    history_model = st.checkbox("HISTORY model", value=False)
with f3:
    spacer, input_col = st.columns([1.6, 1.0], gap="small")
    with input_col:
        search_team = st.text_input(
            label="Search team",
            value="",
            placeholder="Search team",
            label_visibility="collapsed",
        ).strip()

if "model_used" in df.columns:
    if player_model and not history_model:
        df = df[df["model_used"] == "PLAYER"]
    elif history_model and not player_model:
        df = df[df["model_used"] == "HISTORY"]

if search_team:
    ht = df["home_team"].astype(str)
    at = df["away_team"].astype(str)
    df = df[ht.str.contains(search_team, case=False, na=False) | at.str.contains(search_team, case=False, na=False)].copy()

# ---- render ----
for _, r in df.iterrows():
    bits = []
    if pd.notna(r.get("match_no", None)):
        bits.append(f"Match {int(r['match_no'])}")
    if pd.notna(r.get("date", None)):
        bits.append(pd.to_datetime(r["date"]).strftime("%Y-%m-%d"))
    label = " · ".join(bits) if bits else "Match"

    st.markdown(
        f"<div style='text-align:center; font-weight:800; margin-top:0.75rem;'>{label}</div>",
        unsafe_allow_html=True,
    )

    render_match_row_score(
        home_team=str(r.get("home_team", "")),
        away_team=str(r.get("away_team", "")),
        home_score=r.get("home_score_sim", None),
        away_score=r.get("away_score_sim", None),
    )

    winner = str(r.get("winner", ""))
    win_method = str(r.get("win_method", ""))
    model_used = str(r.get("model_used", ""))

    st.markdown(
        f"<div style='text-align:center; opacity:0.85; font-size:0.95rem; margin-bottom:0.6rem;'>"
        f"Winner: <span style='font-weight:700;'>{winner}</span> · {win_method} · Model: {model_used}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.caption("Source: tournament_single_run/tournament_knockout_matches.csv")