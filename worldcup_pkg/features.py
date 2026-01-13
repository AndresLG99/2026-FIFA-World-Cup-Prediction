# team + player features

import pandas as pd
import numpy as np
from .config import N_LAST_GAMES
from .data_loading import load_international_data


def compute_team_stats(matches_df, team, date_cutoff, n_games=N_LAST_GAMES):
    """Compute rolling stats for a team up to date_cutoff."""
    team_matches = matches_df[
        ((matches_df['home_team'] == team) | (matches_df['away_team'] == team)) &
        (matches_df['date'] < date_cutoff)
        ].tail(n_games)

    if len(team_matches) == 0:
        return pd.Series({'win_rate': 0.5, 'goal_diff_avg': 0})

    is_home = team_matches['home_team'] == team
    goals_scored = np.where(is_home, team_matches['home_goals'], team_matches['away_goals'])
    goals_conceded = np.where(is_home, team_matches['away_goals'], team_matches['home_goals'])
    results = np.where(goals_scored > goals_conceded, 1, np.where(goals_scored < goals_conceded, 0, 0.5))

    return pd.Series({
        'win_rate': results.mean(),
        'goal_diff_avg': (goals_scored - goals_conceded).mean()
    })


def build_match_features(worldcup_df, international_df):
    """Build features for each World Cup match."""
    features = []
    for _, match in worldcup_df.iterrows():
        cutoff = match['date']
        home_stats = compute_team_stats(international_df, match['home_team'], cutoff)
        away_stats = compute_team_stats(international_df, match['away_team'], cutoff)

        feat_row = {
            'home_win_rate': home_stats['win_rate'],
            'away_win_rate': away_stats['win_rate'],
            'home_goal_diff': home_stats['goal_diff_avg'],
            'away_goal_diff': away_stats['goal_diff_avg'],
            'form_diff': home_stats['win_rate'] - away_stats['win_rate'],
            'strength_diff': home_stats['goal_diff_avg'] - away_stats['goal_diff_avg']
        }
        feat_row.update(match[['home_team', 'away_team', 'date', 'home_win']].to_dict())
        features.append(feat_row)

    return pd.DataFrame(features).dropna()
