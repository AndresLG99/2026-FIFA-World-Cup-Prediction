"""
Run Monte Carlo tournament simulations and aggregate probabilities.

Inputs:
- Calls src.simulate_tournament.main() which writes these files each run:
  - PROCESSED_PATHS["tournament_summary"]
  - PROCESSED_PATHS["tournament_group_table"]
  - PROCESSED_PATHS["tournament_knockout_matches"]

Outputs:
- PROCESSED_PATHS["mc_run_summaries"]
- PROCESSED_PATHS["mc_champion_odds"]
- PROCESSED_PATHS["mc_reach_round_odds"]
- PROCESSED_PATHS["mc_group_stage_expectations"]
- PROCESSED_PATHS["mc_round_matchup_odds"]
"""

import argparse
import time
import pandas as pd

from config import PROCESSED_PATHS
import src.simulate_tournament as sim


def read_one_run_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(PROCESSED_PATHS["tournament_summary"])
    group_table = pd.read_csv(PROCESSED_PATHS["tournament_group_table"])
    knockout = pd.read_csv(PROCESSED_PATHS["tournament_knockout_matches"])
    return summary, group_table, knockout


def compute_reach_round_rows(knockout: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    def teams_in_stage(stage: str) -> set[str]:
        df = knockout[knockout["stage"] == stage]
        if df.empty:
            return set()
        return set(df["home_team"].astype(str)) | set(df["away_team"].astype(str))

    r32 = teams_in_stage("R32")
    r16 = teams_in_stage("R16")
    qf = teams_in_stage("QF")
    sf = teams_in_stage("SF")
    final = teams_in_stage("FINAL")

    final_winner = None
    df_final = knockout[knockout["stage"] == "FINAL"]
    if not df_final.empty:
        final_winner = str(df_final.iloc[-1]["winner"])

    all_teams = sorted(r32 | r16 | qf | sf | final)

    rows = []
    for t in all_teams:
        rows.append(
            {
                "rng_seed": rng_seed,
                "team": t,
                "reached_R32": int(t in r32),
                "reached_R16": int(t in r16),
                "reached_QF": int(t in qf),
                "reached_SF": int(t in sf),
                "reached_FINAL": int(t in final),
                "won_FINAL": int(final_winner == t),
            }
        )

    return pd.DataFrame(rows)


def compute_group_stage_rows(group_table: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    df = group_table.copy()
    df["rng_seed"] = rng_seed
    keep = ["rng_seed", "group", "team", "pts", "gd", "gf", "ga", "rank_in_group"]
    return df[keep]


def compute_matchup_rows(knockout: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    """
    One row per knockout match with an order-invariant team pair.
    """
    df = knockout.copy()
    df = df[df["stage"].isin(["R32", "R16", "QF", "SF", "FINAL", "3P"])].copy()

    t1 = df["home_team"].astype(str)
    t2 = df["away_team"].astype(str)

    team1 = t1.where(t1 <= t2, t2)
    team2 = t2.where(t1 <= t2, t1)

    out = pd.DataFrame(
        {
            "rng_seed": rng_seed,
            "stage": df["stage"].astype(str),
            "team1": team1,
            "team2": team2,
        }
    )
    return out


def aggregate_outputs(
    summaries: pd.DataFrame,
    reach_rows: pd.DataFrame,
    group_rows: pd.DataFrame,
    matchup_rows: pd.DataFrame,
    n_sims: int,
) -> None:
    summaries.to_csv(PROCESSED_PATHS["mc_run_summaries"], index=False)

    champ = summaries.groupby("champion").size().reset_index(name="n_wins")
    champ["p_win"] = champ["n_wins"] / float(n_sims)
    champ = champ.sort_values(["p_win", "champion"], ascending=[False, True])
    champ.to_csv(PROCESSED_PATHS["mc_champion_odds"], index=False)

    reach = (
        reach_rows.groupby("team")[["reached_R32", "reached_R16", "reached_QF", "reached_SF", "reached_FINAL", "won_FINAL"]]
        .mean()
        .reset_index()
        .rename(columns={"won_FINAL": "p_win"})
        .sort_values(["p_win", "team"], ascending=[False, True])
    )
    reach.to_csv(PROCESSED_PATHS["mc_reach_round_odds"], index=False)

    group_exp = (
        group_rows.groupby("team")[["pts", "gd", "gf", "ga", "rank_in_group"]]
        .mean()
        .reset_index()
        .sort_values(["pts", "gd", "gf", "team"], ascending=[False, False, False, True])
    )
    group_exp.to_csv(PROCESSED_PATHS["mc_group_stage_expectations"], index=False)

    matchups = matchup_rows.groupby(["stage", "team1", "team2"]).size().reset_index(name="n_occurrences")
    matchups["p_occurs"] = matchups["n_occurrences"] / float(n_sims)
    matchups = matchups.sort_values(["stage", "p_occurs", "team1", "team2"], ascending=[True, False, True, True])
    matchups.to_csv(PROCESSED_PATHS["mc_round_matchup_odds"], index=False)

    print("Saved:", PROCESSED_PATHS["mc_run_summaries"])
    print("Saved:", PROCESSED_PATHS["mc_champion_odds"])
    print("Saved:", PROCESSED_PATHS["mc_reach_round_odds"])
    print("Saved:", PROCESSED_PATHS["mc_group_stage_expectations"])
    print("Saved:", PROCESSED_PATHS["mc_round_matchup_odds"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims", type=int, default=200)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    t0 = time.time()

    summaries_all = []
    reach_all = []
    group_all = []
    matchup_all = []

    for i in range(args.n_sims):
        seed = args.seed_start + i

        # Set seed for the tournament module (it uses a module-level RNG_SEED).
        sim.RNG_SEED = seed
        sim.main()

        summary, group_table, knockout = read_one_run_outputs()

        # Validate seed consistency (prevents silent mismatch bugs).
        if int(summary.loc[0, "rng_seed"]) != int(seed):
            raise ValueError(f"Seed mismatch: summary has {summary.loc[0,'rng_seed']} but loop seed is {seed}")

        summaries_all.append(summary)
        reach_all.append(compute_reach_round_rows(knockout, rng_seed=seed))
        group_all.append(compute_group_stage_rows(group_table, rng_seed=seed))
        matchup_all.append(compute_matchup_rows(knockout, rng_seed=seed))

        if (i + 1) % args.progress_every == 0:
            dt = time.time() - t0
            print(f"Completed {i+1}/{args.n_sims} sims in {dt:.1f}s")

    summaries = pd.concat(summaries_all, ignore_index=True)
    reach_rows = pd.concat(reach_all, ignore_index=True)
    group_rows = pd.concat(group_all, ignore_index=True)
    matchup_rows = pd.concat(matchup_all, ignore_index=True)

    aggregate_outputs(summaries, reach_rows, group_rows, matchup_rows, n_sims=args.n_sims)


if __name__ == "__main__":
    main()