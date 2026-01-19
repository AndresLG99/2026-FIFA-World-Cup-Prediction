# 2026 FIFA World Cup Prediction (Monte Carlo)

This project predicts 90-minute expected goals for international fixtures using a two-model approach (player-feature model + match-history fallback), then simulates the expanded 2026 FIFA World Cup format via Monte Carlo to estimate team advancement and title odds.

## Project overview

The pipeline builds two Poisson regression models (home-goals and away-goals for each feature set) and uses simple routing: use the PLAYER model only when both teams have player features, otherwise fall back to the HISTORY model.

Tournament simulation samples match scores from Poisson distributions parameterized by predicted expected goals, then applies group/knockout rules for the 12-group 2026 format (including best third-place teams and a 32-team knockout).

## Repository structure

- `src/` — data loading, feature engineering, training, prediction, simulation, reporting.
- `data/raw/` — input datasets (match results, World Cup tables, player seasons, fixtures template, mapping tables).
- `data/processed/` — generated artifacts used by later steps (feature tables, predictions, simulation outputs, Monte Carlo aggregates).
- `models/` — serialized models (`*.joblib`) and training-time feature column lists (`*.csv`).
- `reports/` — report-ready tables and figures used for communication and the Streamlit app.

## Data sources

Full bibliography: `docs/data_sources.md`.

## Setup

### 1) Create and activate an environment

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Ensure the data files exist

This project expects the raw CSVs to match the paths defined in `config.py` (see `DATAPATHS`).

At minimum, ensure these exist:

- `data/raw/main/all_matches.csv`
- `data/raw/fixtures/all_matches_WC_2026.csv`
- `data/raw/players_data-2011_2022/` (World Cup tables)
- `data/raw/players_data-2023_2026/` (player seasons 23/24, 24/25, 25/26)
- `data/raw/main/countries_names.csv` and `data/raw/main/nation_code_map.csv`

## How to run (end-to-end)

Run commands from the repository root.

### Step 1 — Build feature tables

```bash
python -m src.match_features
```

This produces the processed model tables used for training and for 2026 fixture prediction (player-based and history-based).

### Step 2 — Train models

```bash
python -m src.train
```

This writes four trained Poisson regression models (player/home, player/away, history/home, history/away) into `models/`, plus the corresponding feature column lists.

### Step 3 — Predict expected goals for 2026 fixtures

```bash
python -m src.predict
```

This writes `data/processed/predictions_fixtures_2026.csv` and reports how many fixtures were routed to PLAYER vs HISTORY (placeholders are filtered).

### Step 4 — Simulate a single 2026 tournament

```bash
python -m src.simulate_tournament
```

This writes (to `data/processed/`):

- `tournament_group_matches.csv`
- `tournament_group_table.csv`
- `tournament_knockout_matches.csv`
- `tournament_summary.csv`

### Step 5 — Monte Carlo aggregation (N tournament simulations)

```bash
python  python -m src.run_monte_carlo --n_sims 1000 --seed_start 1 --progress_every 25
```

This repeatedly calls the single-tournament simulator and aggregates results into:

- `data/processed/mc_run_summaries.csv`
- `data/processed/mc_champion_odds.csv`
- `data/processed/mc_reach_round_odds.csv`
- `data/processed/mc_group_stage_expectations.csv`
- `data/processed/mc_round_matchup_odds.csv`

### Step 6 — Build report tables and figures

```bash
python -m src.metrics.build_report_tables
python -m src.reports.make_figures
```

These generate tables/figures under `reports/` for inclusion in the Streamlit app and final write-up.

### Step 7 — Launch the Streamlit app

```bash
streamlit run Home.py
```

## Notes and design choices

- The prediction scripts avoid scoring template placeholders like “Group A Winners”, “Match 74 Winner”, or “... 3rd Place” because those are not real teams until the bracket is simulated.
- For knockout matches that end level in simulated 90 minutes, the simulator resolves ties using expected goals (and then a coin flip if expected goals are equal), as a lightweight approximation of extra time/penalties.
- Third-place advancement and Round-of-32 slotting is handled via backtracking constraint satisfaction so the bracket assignment remains valid under the official template constraints.

## Reproducibility

- Tournament simulation randomness is controlled by a seed (`RNGSEED`) inside `src/simulate_tournament.py`, and Monte Carlo iterates seeds to produce multiple runs.
- All key filesystem paths are centralized in `config.py` to reduce hard-coded paths across modules.
