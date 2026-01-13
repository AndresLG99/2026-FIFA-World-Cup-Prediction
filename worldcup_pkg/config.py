# Configuration and paths
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Dataset paths (update these after downloading CSVs)
WORLD_CUP_PATH = os.path.join(DATA_DIR, "raw", "worldcup_matches.csv")
INTERNATIONAL_PATH = os.path.join(DATA_DIR, "raw", "international_results.csv")
PLAYER_STATS_PATH = os.path.join(DATA_DIR, "raw", "player_stats_2025_26.csv")

# Modeling parameters
N_LAST_GAMES = 10       # for form features
ELO_K = 30              # Elo update factor
TEST_YEAR = 2022        # validate on this World Cup
N_SIMULATIONS = 10000   # tournament sims

# 2026 groups (simplified - use real ones when available)
GROUPS_2026 = {
    'A': ['Host1', 'Team2', 'Team3', 'Team4'],
    'B': ['Brazil', 'Team6', 'Team7', 'Team8'],
    # Add full groups...
}
