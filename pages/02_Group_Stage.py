from __future__ import annotations

import os
import pandas as pd
import streamlit as st


# ---------- helpers ----------
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


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Supports both styles, then standardizes to underscore naming.
    rename_map = {
        "matchno": "match_no",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "predhomegoals": "pred_home_goals",
        "predawaygoals": "pred_away_goals",
        "homescoresim": "home_score_sim",
        "awayscoresim": "away_score_sim",
        "modelused": "model_used",
        "winmethod": "win_method",
        "rankingroup": "rank_in_group",
    }
    out = df.copy()
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def render_match_row_score(home_team: str, away_team: str, home_score: int | None, away_score: int | None) -> None:
    left_margin, c1, c2, c3, c4, c5, right_margin = st.columns(
        [0.6, 2.2, 0.8, 0.6, 0.8, 2.2, 0.6],
        gap="small",
    )

    hs = "" if home_score is None else str(int(home_score))
    a_s = "" if away_score is None else str(int(away_score))

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


def render_match_meta(scoreline: str, model_used: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center; opacity:0.85; font-size:0.95rem; margin-bottom:0.6rem;">
            Score: <span style="font-weight:700;">{scoreline}</span> · Model: {model_used}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- page ----------
st.set_page_config(page_title="Group Stage", layout="wide")
st.title("Group Stage")

st.markdown(
    "This page shows one simulated tournament run: group standings and the 72 group matches."
)

# These files are produced by simulate_tournament.py [file:733]
group_table_path = os.path.join("data", "processed", "tournament_single_run", "tournament_group_table.csv")
group_matches_path = os.path.join("data", "processed", "tournament_single_run", "tournament_group_matches.csv")

group_table = normalize_cols(safe_read_csv(group_table_path))
group_matches = normalize_cols(coerce_date(safe_read_csv(group_matches_path), "date"))

# ---------- standings ----------
st.subheader("Standings", divider="gray")

# Expect: columns like group, team, pts, gd, gf, rank_in_group (or enough to sort) [file:733]
standings = group_table.copy()

# Ensure ordering inside each group
sort_cols = [c for c in ["group", "rank_in_group", "pts", "gd", "gf", "team"] if c in standings.columns]
if "group" in standings.columns:
    if "rank_in_group" in standings.columns:
        standings = standings.sort_values(["group", "rank_in_group", "team"], ascending=[True, True, True])
    else:
        # fallback sort similar to your simulation logic [file:733]
        standings = standings.sort_values(
            [c for c in ["group", "pts", "gd", "gf", "team"] if c in standings.columns],
            ascending=[True, False, False, False, True][: len([c for c in ["group", "pts", "gd", "gf", "team"] if c in standings.columns])],
        )

groups = sorted([g for g in standings["group"].dropna().unique()]) if "group" in standings.columns else []

medals = {1: "🥇", 2: "🥈", 3: "🥉"}

# Render 4 columns × 3 rows (12 groups A–L)
n_cols = 4
for row_start in range(0, len(groups), n_cols):
    row_groups = groups[row_start : row_start + n_cols]
    cols = st.columns(n_cols, gap="large")

    for i, g in enumerate(row_groups):
        with cols[i]:
            st.markdown(f"#### Group {g}")

            gdf = standings[standings["group"] == g].copy()

            # Build a simple ranked list
            lines = []
            for j, (_, r) in enumerate(gdf.iterrows(), start=1):
                medal = medals.get(j, "")
                team = str(r.get("team", ""))
                lines.append(f"{medal} {team}".strip())

            st.markdown("\n".join([f"- {x}" for x in lines]))


# ---------- matches ----------
st.subheader("Matches", divider="gray")

matches = group_matches.copy()

# nice ordering
if "match_no" in matches.columns:
    matches = matches.sort_values(["match_no"])
elif "date" in matches.columns:
    matches = matches.sort_values(["date"])

f1, f2, f3 = st.columns([1.2, 1.2, 2.6], gap="small")

with f1:
    player_model = st.checkbox("PLAYER model", value=False)
with f2:
    history_model = st.checkbox("HISTORY model", value=False)
with f3:
    spacer, input_col = st.columns([1.6, 1.0], gap="small")  # pushes the input to the right, keeps it shorter
    with input_col:
        search_team = st.text_input(
            label="Search team",
            value="",
            placeholder="Search team",
            label_visibility="collapsed",
        ).strip()


# model filter behavior:
# - if neither checked => show all
# - if one checked => show only that
# - if both checked => show all (same as neither)
if "model_used" in matches.columns:
    if player_model and not history_model:
        matches = matches[matches["model_used"] == "PLAYER"]
    elif history_model and not player_model:
        matches = matches[matches["model_used"] == "HISTORY"]

# team search
if search_team:
    ht = matches["home_team"].astype(str)
    at = matches["away_team"].astype(str)
    matches = matches[ht.str.contains(search_team, case=False, na=False) | at.str.contains(search_team, case=False, na=False)]


# render list
for _, r in matches.iterrows():
    bits = []
    if pd.notna(r.get("match_no", None)):
        bits.append(f"Match {int(r['match_no'])}")
    if pd.notna(r.get("group", None)):
        bits.append(f"Group {r['group']}")
    if pd.notna(r.get("date", None)):
        bits.append(pd.to_datetime(r["date"]).strftime("%Y-%m-%d"))

    match_label = " · ".join(bits) if bits else "Match"

    st.markdown(
        f"<div style='text-align:center; font-weight:800; margin-top:0.75rem;'>{match_label}</div>",
        unsafe_allow_html=True,
    )

    home_team = str(r.get("home_team", ""))
    away_team = str(r.get("away_team", ""))

    home_score = r.get("home_score_sim", None)
    away_score = r.get("away_score_sim", None)

    render_match_row_score(home_team, away_team, home_score, away_score)

    model_used = str(r.get("model_used", ""))
    st.markdown(
        f"<div style='text-align:center; opacity:0.85; font-size:0.95rem; margin-bottom:0.6rem;'>Model: {model_used}</div>",
        unsafe_allow_html=True,
    )

st.caption("Source: tournament_single_run outputs from simulate_tournament.py.")