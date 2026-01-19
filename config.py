# Single source of truth for paths and project settings (prevents hard-coded locations).

"""
What config.py does:
- Think of config.py like a map that tells every script where the files live.
- If a user has their folders slightly different, they only fix config.py, not 6 scripts.
- It improves reproducibility and prevents “it works on my laptop” issues.
"""

import os

# Root folder = the folder where config.py is located (robust across run locations).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data folders
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")

# Processed subfolders
DATA_PROCESSED_MODEL_TABLES = os.path.join(DATA_PROCESSED, "model_tables")
DATA_PROCESSED_MONTE_CARLO = os.path.join(DATA_PROCESSED, "monte_carlo")
DATA_PROCESSED_PREDICTIONS = os.path.join(DATA_PROCESSED, "predictions")
DATA_PROCESSED_TOURNAMENT_SINGLE_RUN = os.path.join(DATA_PROCESSED, "tournament_single_run")

# Output folders
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# Create output folders if they don't exist
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_PROCESSED_MODEL_TABLES, exist_ok=True)
os.makedirs(DATA_PROCESSED_MONTE_CARLO, exist_ok=True)
os.makedirs(DATA_PROCESSED_PREDICTIONS, exist_ok=True)
os.makedirs(DATA_PROCESSED_TOURNAMENT_SINGLE_RUN, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# === Raw CSV paths ===
DATA_PATHS = {
    # International results
    "all_matches": os.path.join(DATA_RAW, "main", "all_matches.csv"),
    "fixtures_2026": os.path.join(DATA_RAW, "fixtures", "all_matches_WC_2026.csv"),
    "country_names": os.path.join(DATA_RAW, "main", "countries_names.csv"),

    # World Cup historical (1930–2022)
    "wc_goals": os.path.join(DATA_RAW, "players_data-2011_2022", "goals.csv"),
    "wc_matches": os.path.join(DATA_RAW, "players_data-2011_2022", "matches.csv"),
    "wc_player_appearances": os.path.join(DATA_RAW, "players_data-2011_2022", "player_appearances.csv"),
    "wc_squads": os.path.join(DATA_RAW, "players_data-2011_2022", "squads.csv"),
    "wc_teams": os.path.join(DATA_RAW, "players_data-2011_2022", "teams.csv"),

    # Player stats (club seasons)
    "players_23_24": os.path.join(DATA_RAW, "players_data-2023_2026", "player_data-2023_2024.csv"),
    "players_24_25": os.path.join(DATA_RAW, "players_data-2023_2026", "players_data-2024_2025.csv"),
    "players_25_26": os.path.join(DATA_RAW, "players_data-2023_2026", "players_data-2025_2026.csv"),

    # Code mapping
    "nation_code_map": os.path.join(DATA_RAW, "main", "nation_code_map.csv"),
}

# === Processed outputs ===
PROCESSED_PATHS = {
    # Player-derived model tables
    "model_table_player_train": os.path.join(DATA_PROCESSED_MODEL_TABLES, "model_table_player_train.csv"),
    "model_table_player_fixtures": os.path.join(DATA_PROCESSED_MODEL_TABLES, "model_table_player_fixtures_2026.csv"),

    # Match-history-derived model tables
    "model_table_history_train": os.path.join(DATA_PROCESSED_MODEL_TABLES, "model_table_history_train.csv"),
    "model_table_history_fixtures": os.path.join(DATA_PROCESSED_MODEL_TABLES, "model_table_history_fixtures_2026.csv"),

    # Match predictions
    "fixtures_2026_resolved": os.path.join(DATA_PROCESSED_PREDICTIONS, "fixtures_2026_resolved.csv"),
    "predictions_fixtures_2026": os.path.join(DATA_PROCESSED_PREDICTIONS, "predictions_fixtures_2026.csv"),

    # Playoff bracket outputs
    "playoff_predictions": os.path.join(DATA_PROCESSED_PREDICTIONS, "playoff_predictions.csv"),

    # Monte Carlo outputs
    "mc_run_summaries": os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_run_summaries.csv"),
    "mc_champion_odds": os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_champion_odds.csv"),
    "mc_reach_round_odds": os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_reach_round_odds.csv"),
    "mc_round_matchup_odds": os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_round_matchup_odds.csv"),
    "mc_group_stage_expectations": os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_group_stage_expectations.csv"),

    # Tournament single-run outputs
    "tournament_group_matches": os.path.join(DATA_PROCESSED_TOURNAMENT_SINGLE_RUN, "tournament_group_matches.csv"),
    "tournament_group_table": os.path.join(DATA_PROCESSED_TOURNAMENT_SINGLE_RUN, "tournament_group_table.csv"),
    "tournament_knockout_matches": os.path.join(DATA_PROCESSED_TOURNAMENT_SINGLE_RUN, "tournament_knockout_matches.csv"),
    "tournament_summary": os.path.join(DATA_PROCESSED_TOURNAMENT_SINGLE_RUN, "tournament_summary.csv"),
}

# === Modeling settings ===
WC_TEST_YEARS = [2014, 2018, 2022]
PREDICT_90_MIN_ONLY = True

# Avoid leakage when evaluating (train on <= this date, test after)
TRAIN_MAX_DATE = "2022-12-31"

# Player feature building
MIN_PLAYER_MINUTES = 300

# === History features settings (fallback model) ===
ELO_BASE = 1500
ELO_K = 20
ELO_HOME_ADV = 0
ROLLING_WINDOW = 10

# === Hybrid routing ===
USE_PLAYER_MODEL_WHEN_BOTH_HAVE_FEATURES = True

# === Prediction settings ===
PREDICTION_ASOF_DATE = "2026-01-15"
USE_RESOLVED_FIXTURES = True