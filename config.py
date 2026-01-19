# Paths, hyperparameters, WC years

import os

# ROOT DIRECTORY (where config.py lives)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# EXACT PATHS
DATA_PATHS = {
    # Main international matches (Daily Updates Kaggle)
    "daily_updates": os.path.join(DATA_RAW, "main", "all_matches.csv"),

    # GitHub World Cup (2011-2022 blocks)
    "wc_matches": os.path.join(DATA_RAW, "players_data-2011_2022", "matches.csv"),
    "wc_goals": os.path.join(DATA_RAW, "players_data-2011_2022", "goals.csv"),
    "wc_appearances": os.path.join(DATA_RAW, "players_data-2011_2022", "player_appearances.csv"),
    "wc_squads": os.path.join(DATA_RAW, "players_data-2011_2022", "squads.csv"),
    "wc_teams": os.path.join(DATA_RAW, "players_data-2011_2022", "teams.csv"),

    # Recent player stats (2023-2026 for fixtures prediction)
    "players_23_24": os.path.join(DATA_RAW, "players_data-2023_2026", "player_data-2023_2024.csv"),
    "players_24_25": os.path.join(DATA_RAW, "players_data-2023_2026", "players_data-2024_2025.csv"),
    "players_25_26": os.path.join(DATA_RAW, "players_data-2023_2026", "players_data-2025_2026.csv"),

    # 2026 World Cup fixtures (your manual creation)
    "fixtures_2026": os.path.join(DATA_RAW, "fixtures", "all_matches_WC_2026.csv")
}

# World Cup validation blocks (train → test WC year)
WC_BLOCKS = [
    {"train_start": 2011, "train_end": 2014, "test_wc": 2014},
    {"train_start": 2015, "train_end": 2018, "test_wc": 2018},
    {"train_start": 2019, "train_end": 2022, "test_wc": 2022}
]

# Poisson model hyperparameters
POISSON_CONFIG = {
    "alpha": 1.0,  # L2 regularization
    "max_iter": 1000,
    "fit_intercept": True
}

# Expected feature names (auto-generated)
NUMERIC_FEATURES = [
    "home_goals90", "away_goals90",
    "home_xG90", "away_xG90",
    "home_advantage", "squad_strength"
]
CATEGORICAL_FEATURES = ["home_team", "away_team"]

# Output directories
PROCESSED_PATHS = {
    "master_features": os.path.join(DATA_PROCESSED, "master_features.parquet"),
    "block_results": os.path.join(DATA_PROCESSED, "block_evaluation.csv"),
    "predictions_2026": os.path.join(DATA_PROCESSED, "2026_predictions.csv")
}

# Model save paths (in src/models/ after training)
MODEL_PATHS = {
    "poisson_home": os.path.join(ROOT_DIR, "src", "models", "poisson_home.pkl"),
    "poisson_away": os.path.join(ROOT_DIR, "src", "models", "poisson_away.pkl")
}

FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
