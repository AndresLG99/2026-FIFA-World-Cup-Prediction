# read CVs / APIs

import pandas as pd
from .config import WORLD_CUP_PATH, INTERNATIONAL_PATH, PLAYER_STATS_PATH

def load_worldcup_data():
    """Load World Cup matches."""
    df = pd.read_csv(WORLD_CUP_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['home_team'] = df['home_team'].str.strip().str.lower()
    df['away_team'] = df['away_team'].str.strip().str.lower()
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
    df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
    return df.sort_values('date')

def load_international_data():
    """Load all international matches."""
    df = pd.read_csv(INTERNATIONAL_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['home_team'] = df['home_team'].str.strip().str.lower()
    df['away_team'] = df['away_team'].str.strip().str.lower()
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
    return df.sort_values('date')

def load_player_stats():
    """Load 2025-26 player stats."""
    df = pd.read_csv(PLAYER_STATS_PATH)
    # Assume columns: player, nationality, goals_90, xG_90, assists_90, etc.
    df['nationality'] = df['nationality'].str.strip().str.lower()
    return df.groupby('nationality').agg({
        'goals_90': ['mean', 'median'],
        'xG_90': 'mean',
        'assists_90': 'mean',
        # Add more
    }).reset_index()
