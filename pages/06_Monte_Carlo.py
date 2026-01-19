from __future__ import annotations

import os
import pandas as pd
import streamlit as st


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


def pct(x: float) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):.1f}%"


st.set_page_config(page_title="Monte Carlo", layout="wide")
st.title("Monte Carlo")

st.markdown(
    """
A Monte Carlo simulation estimates probabilities by running the same process many times with randomness, then counting how often each outcome happens.  
In this project, each “run” simulates a full tournament; aggregating thousands of runs gives estimated odds like “win the tournament” or “reach the semifinal”.
"""
)

base = os.path.join("data", "processed", "monte_carlo")

champion_path = os.path.join(base, "mc_champion_odds.csv")
reach_path = os.path.join(base, "mc_reach_round_odds.csv")
group_exp_path = os.path.join(base, "mc_group_stage_expectations.csv")
runs_path = os.path.join(base, "mc_run_summaries.csv")

df_champ = safe_read_csv(champion_path)
df_reach = safe_read_csv(reach_path)
df_group = safe_read_csv(group_exp_path)
df_runs = safe_read_csv(runs_path)

# ------------------ Run summaries ------------------
st.subheader("Simulation run summaries", divider="gray")

with st.expander("What is a run summary?"):
    st.markdown(
        """
Each row is one full tournament simulation:
- `rng_seed`: Random seed used for that run.
- `asof_date`: Snapshot date of the underlying inputs used to run the simulation.
- `champion`: The champion in that specific run.
- `n_wins`: Number of Monte Carlo runs where this team won the tournament.
- `p_win`: Estimated probability of winning the tournament = `n_wins / total_runs`.
        """
    )

# Expect columns: rng_seed, asof_date, champion
n_runs = len(df_runs)  # [file:800]
unique_champs = df_runs["champion"].nunique() if "champion" in df_runs.columns else 0  # [file:800]

m1, m2 = st.columns(2)
with m1:
    st.metric("Runs", n_runs)
with m2:
    st.metric("Unique champions", unique_champs)

# Top champions from runs (should match df_champ, but derived from raw runs)
if "champion" in df_runs.columns:
    champ_counts = df_runs["champion"].value_counts().rename_axis("champion").reset_index(name="n_wins")
    champ_counts["p_win"] = champ_counts["n_wins"] / n_runs
    champ_counts = champ_counts.head(12).copy()
    champ_counts["p_win"] = champ_counts["p_win"].map(pct)

    st.dataframe(champ_counts, use_container_width=True, hide_index=True)

# ------------------ Champion odds ------------------
st.subheader("Champion odds", divider="gray")

# Optional simple bar chart (Streamlit native)
df_champ_chart = df_champ.sort_values("p_win", ascending=False).head(12).copy()
df_champ_chart = df_champ_chart.set_index("champion")[["p_win"]]
st.bar_chart(df_champ_chart)

# ------------------ Reach round odds ------------------
st.subheader("Reach-round odds", divider="gray")

with st.expander("What do these columns mean?"):
    st.markdown(
        """
- `team`: Team name.
- `reached_R32`, `reached_R16`, `reached_QF`, `reached_SF`, `reached_FINAL`: Fraction of runs where the team reached that round (so 0.582 ≈ 58.2%).
- `p_win`: Fraction of runs where the team won the tournament.
        """
    )

# Expect columns: team, reached_R32, reached_R16, reached_QF, reached_SF, reached_FINAL, p_win
reach_cols = ["reached_QF", "reached_SF", "reached_FINAL", "p_win"]
reach_cols = [c for c in reach_cols if c in df_reach.columns]

reach_view = df_reach.copy()  # [file:798]
if "p_win" in reach_view.columns:
    reach_view = reach_view.sort_values("p_win", ascending=False)

reach_view = reach_view[["team"] + reach_cols].head(25).copy()
for c in reach_cols:
    reach_view[c] = reach_view[c].map(pct)

st.dataframe(reach_view, use_container_width=True, hide_index=True)



st.caption("Sources: mc_champion_odds.csv, mc_reach_round_odds.csv, mc_group_stage_expectations.csv, mc_run_summaries.csv.")