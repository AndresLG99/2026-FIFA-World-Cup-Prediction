from __future__ import annotations
import os
import pandas as pd
import streamlit as st
import altair as alt


# Helpers
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)

def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100*x:.1f}%"

# Page setup
st.set_page_config(page_title="Figures and Tables", layout="wide")
st.title("Figures and Tables")


# Section 1: Top 15 Teams - Round Progression Probabilities
st.header("Top 15 Teams – Round Progression Probabilities")
st.markdown(
    """
This bar chart shows the probability of the top 15 countries reaching a selected tournament round,
based on Monte Carlo simulations.
"""
)

# Load reach probabilities
reach_path = os.path.join("data", "processed", "monte_carlo", "mc_reach_round_odds.csv")
df_reach = safe_read_csv(reach_path)

round_cols = ["reached_R32", "reached_R16", "reached_QF", "reached_SF", "reached_FINAL"]
if "team" not in df_reach.columns or any(c not in df_reach.columns for c in round_cols):
    st.error(f"The CSV must contain 'team' and {round_cols}")
    st.stop()

round_map = {
    "Round of 32": "reached_R32",
    "Round of 16": "reached_R16",
    "Quarterfinals": "reached_QF",
    "Semifinals": "reached_SF",
    "Final": "reached_FINAL",
}

selected_round = st.selectbox("Select tournament round to visualize", list(round_map.keys()))
round_col = round_map[selected_round]

# Prepare top 15
df_top = df_reach[["team", round_col]].copy()
df_top = df_top.rename(columns={round_col: "probability"})
df_top = df_top.sort_values("probability", ascending=False).head(15)

# Bar chart
chart_top = (
    alt.Chart(df_top)
    .mark_bar()
    .encode(
        x=alt.X("team:N", sort=None, title="Team"),
        y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("probability:Q", title="Probability", format=".1%")],
        color=alt.Color("probability:Q", scale=alt.Scale(scheme="blues"), legend=None),
    )
    .properties(width=1000, height=500)
)
st.altair_chart(chart_top, use_container_width=True)

# Table
st.subheader(f"Top 15 Teams Probabilities for {selected_round}")
df_top_display = df_top.copy()
df_top_display["probability"] = df_top_display["probability"].apply(pct)
st.dataframe(df_top_display, use_container_width=True)


# Section 2: Probabilities Trajectory per Team
st.header("Probability Trajectory per Team")
st.markdown(
    """
This line chart shows the probability of each top 15 team advancing through each tournament round.
It visualizes how likely each team is to progress to the later stages.
"""
)

# Prepare trajectory top 15 by Final probability
df_traj = df_reach[["team"] + round_cols].copy()
df_traj = df_traj.rename(columns={
    "reached_R32": "R32",
    "reached_R16": "R16",
    "reached_QF": "QF",
    "reached_SF": "SF",
    "reached_FINAL": "Final"
})
# Top 15 by Final probability
df_traj_top = df_traj.sort_values("Final", ascending=False).head(15)

# Melt for line chart
df_traj_melt = df_traj_top.melt(id_vars="team", var_name="Round", value_name="Probability")

# Line chart
chart_traj = (
    alt.Chart(df_traj_melt)
    .mark_line(point=True)
    .encode(
        x=alt.X("Round:N", sort=["R32","R16","QF","SF","Final"], title="Tournament Round"),
        y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), title="Probability"),
        color=alt.Color("team:N", legend=alt.Legend(title="Team")),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("Probability:Q", title="Probability", format=".1%"), "Round:N"]
    )
    .properties(width=1000, height=500)
)
st.altair_chart(chart_traj, use_container_width=True)


# Section 3: Upset / Surprise Potential
st.header("Upset / Surprise Potential")
st.markdown(
    """
Top 10 teams with the highest chance of advancing relative to their historical strength.
This highlights potential “dark horses” in the tournament.
"""
)

# Simple surprise metric: Final probability / Final Elo (if Elo available)
# For now, just use Final probability as placeholder
df_surprise = df_reach[["team", "reached_FINAL"]].copy()
df_surprise = df_surprise.rename(columns={"reached_FINAL": "Final_Prob"})
df_surprise = df_surprise.sort_values("Final_Prob", ascending=False).head(10)

# Bar chart
chart_surprise = (
    alt.Chart(df_surprise)
    .mark_bar()
    .encode(
        x=alt.X("team:N", title="Team"),
        y=alt.Y("Final_Prob:Q", axis=alt.Axis(format="%"), title="Probability to reach Final"),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("Final_Prob:Q", title="Probability", format=".1%")],
        color=alt.Color("Final_Prob:Q", scale=alt.Scale(scheme="blues"), legend=None),
    )
    .properties(width=1000, height=400)
)
st.altair_chart(chart_surprise, use_container_width=True)

st.caption("Data based on Monte Carlo simulations for the 2026 FIFA World Cup.")
