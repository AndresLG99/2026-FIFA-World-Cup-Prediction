# Processed data (generated outputs)

This folder contains **generated** (derived) datasets produced by the pipeline.
These files can be re-created by rerunning the scripts in `src/` using the paths defined in `config.py`.

## Folder overview

- `model_tables/`
  - Final feature tables used for training and for generating fixture predictions.
  - Outputs are intended to be stable “inputs” for modeling scripts.

- `predictions/`
  - “Latest” prediction outputs produced by the prediction pipeline.
  - Safe to overwrite on reruns.

- `monte_carlo/`
  - “Latest” Monte Carlo simulation outputs.
  - Safe to overwrite on reruns (e.g., rerun with different number of simulations).

- `tournament_single_run/`
  - Outputs from a single end-to-end tournament simulation (group + knockout).
  - Safe to overwrite on reruns.

## Files produced (current)

### model_tables/
- `model_table_player_train.csv`: Player-based training table.
- `model_table_player_fixtures_2026.csv`: Player-based fixtures table for 2026 predictions.
- `model_table_history_train.csv`: History/ELO-based training table.
- `model_table_history_fixtures_2026.csv`: History/ELO-based fixtures table for 2026 predictions.

### predictions/
- `fixtures_2026_resolved.csv`: Fixtures after any resolution/cleanup steps.
- `predictions_fixtures_2026.csv`: Match-level predictions for 2026 fixtures.
- `playoff_predictions.csv`: Knockout/playoff bracket predictions.

### monte_carlo/
- `mc_run_summaries.csv`: Run-level summary statistics for Monte Carlo simulations.
- `mc_champion_odds.csv`: Title-winning odds by team.
- `mc_reach_round_odds.csv`: Probability of reaching each round.
- `mc_round_matchup_odds.csv`: Matchup odds by round (if applicable).
- `mc_group_stage_expectations.csv`: Expected group-stage performance metrics.

### tournament_single_run/
- `tournament_group_matches.csv`: Simulated group-stage match results.
- `tournament_group_table.csv`: Final group standings.
- `tournament_knockout_matches.csv`: Simulated knockout match results.
- `tournament_summary.csv`: High-level summary of the tournament run.

## Reproducibility notes

- Files here are *not* edited manually.
- If an output looks wrong, update the upstream script in `src/` and rerun.
- If you change folder names or add outputs, update `config.py` and then update this README.

## Naming conventions

- All output filenames use snake_case.
- Year-specific outputs include the year (e.g., `_2026`) where relevant.
