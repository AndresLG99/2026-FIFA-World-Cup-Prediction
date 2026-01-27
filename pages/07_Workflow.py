from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Workflow", layout="wide")

st.title("Project workflow")

tab_map, tab_details = st.tabs(["Map", "Details"])

def render_workflow_media() -> bool:
    """
    Render a workflow image/PDF if present in docs/.
    Returns True when a media asset was rendered.
    """
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    candidates = [
        Path("imgs/workflow.png"),
        Path("imgs/workflow.jpg"),
        Path("imgs/workflow.jpeg"),
        Path("imgs/workflow.webp"),
        Path("imgs/workflow.pdf"),
        Path("docs/workflow.png"),
        Path("docs/workflow.jpg"),
        Path("docs/workflow.jpeg"),
        Path("docs/workflow.webp"),
        Path("docs/workflow.pdf"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".pdf":
            pdf_bytes = path.read_bytes()
            encoded = base64.b64encode(pdf_bytes).decode("ascii")
            st.markdown(
                f"""
<div style="max-width:450px; margin:0 auto;">
  <iframe
    src="data:application/pdf;base64,{encoded}"
    width="100%"
    height="310"
    style="border:1px solid #e2e8f0; border-radius:12px;"
  ></iframe>
</div>
""",
                unsafe_allow_html=True,
            )
            return True
        mime = mime_map.get(path.suffix.lower(), "image/png")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        st.markdown(
            f"""
<div style="max-width:450px; margin:0 auto;">
  <img
    src="data:{mime};base64,{encoded}"
    style="width:100%; height:auto; display:block;"
    alt="Workflow diagram"
  />
</div>
""",
            unsafe_allow_html=True,
        )
        return True
    return False


with tab_map:
    st.subheader("Pipeline map")
    st.markdown(
        """
This diagram summarizes the full data flow for the project.
It shows how raw sources become features, model inputs, predictions, and tournament odds.
"""
    )
    st.markdown(
        """
**How to read it**
- Inputs: historical matches, player stats, and country mappings.
- Core processing: loading, cleaning, and feature engineering.
- Modeling: training and inference for fixtures.
- Simulation: playoffs, full tournament, and Monte Carlo odds.
"""
    )

    if not render_workflow_media():
        st.warning(
            "Workflow image not found. Add a file at imgs/workflow.png (or .jpg/.webp/.pdf)."
        )

with tab_details:
    st.subheader("Module descriptions")

    modules = [
        {
            "title": "config.py",
            "role": (
                "Settings hub and folder map. Defines paths, flags, and the as-of "
                "date for snapshots."
            ),
            "inputs": ["No user input (project constants)."],
            "outputs": [
                "Creates output folders when needed (processed, models, reports).",
            ],
        },
        {
            "title": "utils.py",
            "role": "Shared helpers (date parsing and simple cleanup).",
            "inputs": [
                "parse_match_date(series)",
                "nation_to_code(nation_value)",
            ],
            "outputs": [
                "Parsed date column",
                "Standardized country code",
            ],
        },
        {
            "title": "name_cleaning.py",
            "role": "Standardizes team names for reliable merges.",
            "inputs": [
                "DataFrame with home_team and away_team",
                "Name map (countries_names.csv)",
            ],
            "outputs": [
                "DataFrame with standardized names",
                "Optional report of unmapped names",
            ],
        },
        {
            "title": "data_loader.py",
            "role": "Loads raw CSVs, standardizes columns, and applies name_cleaning.",
            "inputs": [
                "CSVs in config.DATA_PATHS (matches, fixtures, players, country map)",
                "Flags (raw=True/False) when calling loaders",
            ],
            "outputs": [
                "load_all_matches()",
                "load_fixtures_2026()",
                "load_wc_tables()",
                "load_player_seasons()",
            ],
        },
        {
            "title": "player_features.py",
            "role": "Combines player seasons and calculates per-90 stats.",
            "inputs": [
                "Dict {season: df} from load_player_seasons()",
                "min_minutes (config)",
            ],
            "outputs": ["players_master (unified player features)."],
        },
        {
            "title": "code_mapping.py",
            "role": "Links nationality code to team name.",
            "inputs": ["nation_code_map.csv"],
            "outputs": ["Mapping DataFrame nation_code -> team_name."],
        },
        {
            "title": "team_features.py",
            "role": "Aggregates player stats into team-level features.",
            "inputs": ["players_master", "nation_code_map"],
            "outputs": ["team_features with weighted means + n_players_used."],
        },
        {
            "title": "history_features.py",
            "role": "Builds Elo and recent form from historical matches.",
            "inputs": ["clean matches", "Elo and window params (config)"],
            "outputs": [
                "build_history_features(...)",
                "build_team_state_asof(...)",
            ],
        },
        {
            "title": "match_features.py",
            "role": "Builds final training and fixture tables.",
            "inputs": [
                "matches + 2026 fixtures",
                "player seasons + nation mapping + history features",
            ],
            "outputs": [
                "model_table_player_train.csv",
                "model_table_player_fixtures_2026.csv",
                "model_table_history_train.csv",
                "model_table_history_fixtures_2026.csv",
            ],
        },
        {
            "title": "train.py",
            "role": "Trains 4 Poisson models (home/away, player/history).",
            "inputs": ["training tables generated in match_features.py"],
            "outputs": [
                "saved models in /models",
                "feature lists for inference",
            ],
        },
        {
            "title": "predict.py",
            "role": "Predicts fixture xG using the best available model.",
            "inputs": [
                "saved models + feature lists",
                "fixture tables in /processed",
            ],
            "outputs": ["predictions_fixtures_2026.csv"],
        },
        {
            "title": "simulate_playoffs.py",
            "role": "Resolves playoff placeholders into final fixtures.",
            "inputs": [
                "models + features + team snapshots",
                "bracket defined in the script",
            ],
            "outputs": [
                "playoff_predictions.csv",
                "fixtures_2026_resolved.csv",
            ],
        },
        {
            "title": "simulate_tournament.py",
            "role": "Simulates a full World Cup (groups and knockouts).",
            "inputs": [
                "models + features + team snapshots",
                "fixtures (ideally resolved)",
            ],
            "outputs": [
                "tournament_group_matches.csv",
                "tournament_group_table.csv",
                "tournament_knockout_matches.csv",
                "tournament_summary.csv",
            ],
        },
        {
            "title": "run_monte_carlo.py",
            "role": "Repeats the tournament many times to produce odds.",
            "inputs": [
                "args: n_sims, seed, freq",
                "uses simulate_tournament.main() each run",
            ],
            "outputs": [
                "mc_champion_odds.csv",
                "mc_reach_round_odds.csv",
                "mc_group_stage_expectations.csv",
                "mc_round_matchup_odds.csv",
                "mc_run_summaries.csv",
            ],
        },
    ]

    for m in modules:
        with st.expander(m["title"], expanded=False):
            st.markdown(f"**Role:** {m['role']}")
            st.markdown("**Input:**")
            for item in m["inputs"]:
                st.write(f"- {item}")
            st.markdown("**Output:**")
            for item in m["outputs"]:
                st.write(f"- {item}")
