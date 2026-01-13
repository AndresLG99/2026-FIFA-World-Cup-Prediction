# tournament simulation

import numpy as np
from .config import N_SIMULATIONS, GROUPS_2026
from .models import FEATURE_COLS


def simulate_world_cup(best_model, group_fixtures):
    """Simulate full tournament."""
    winners = np.zeros(N_SIMULATIONS, dtype=object)

    for sim in range(N_SIMULATIONS):
        # Simplified group stage
        group_results = {}
        for group, teams in GROUPS_2026.items():
            standings = simulate_group(best_model, teams)
            group_results[group] = standings.iloc[0]['team']  # winner

        # Simplified knockout (round of 16 -> final)
        finalists = list(group_results.values())
        champ = simulate_knockout(best_model, finalists)
        winners[sim] = champ

    # Compute probabilities
    unique, counts = np.unique(winners, return_counts=True)
    probs = dict(zip(unique, counts / N_SIMULATIONS))
    return probs


def simulate_group(model, teams):
    """Simulate one group."""
    # Placeholder: generate fake fixtures and results
    # In reality: build actual fixtures, predict, compute standings
    standings = pd.DataFrame({'team': teams})
    standings['points'] = np.random.uniform(3, 9, len(teams))
    return standings.sort_values('points', ascending=False)
