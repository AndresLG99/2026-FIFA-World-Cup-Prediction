"""
metrics/build_report_tables.py

Build plot-ready report tables under:
- reports/tables/   (inputs to plotting)
- reports/metrics/  (raw evaluation outputs)

Design goal:
Plot scripts should ONLY read from reports/tables and write to reports/figures.
"""

import os
from datetime import datetime

import pandas as pd
from joblib import load

from config import (
    REPORTSDIR,
    MODELSDIR,
    DATAPROCESSEDMONTECARLO,
    PROCESSEDPATHS,
    WCTESTYEARS,
)
from src.metrics.regression import mae


def _ensure_report_dirs() -> dict:
    """Create the report subfolders if missing."""
    tables_dir = os.path.join(REPORTSDIR, "tables")
    metrics_dir = os.path.join(REPORTSDIR, "metrics")
    figures_dir = os.path.join(REPORTSDIR, "figures")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return {"tables": tables_dir, "metrics": metrics_dir, "figures": figures_dir}


def _load_feature_list(path: str) -> list[str]:
    """Load the feature column list saved during training."""
    return pd.read_csv(path, header=None)[0].astype(str).tolist()


def _evaluate_world_cup_mae(train_df: pd.DataFrame, feature_cols: list[str], home_model, away_model, label: str) -> pd.DataFrame:
    """
    Evaluate MAE for World Cup matches by year.
    Mirrors the logic in train.py but produces a clean table for plotting.
    """
    if "tournament" not in train_df.columns:
        return pd.DataFrame(columns=["model", "year", "n", "mae_home", "mae_away", "mae_total"])

    wc = train_df[train_df["tournament"].astype(str).str.strip().eq("World Cup")].copy()
    wc["year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year

    rows = []
    for y in WCTESTYEARS:
        test = wc[wc["year"] == y].copy()
        if test.empty:
            continue

        X = test.reindex(columns=feature_cols, fill_value=0.0).astype(float)
        y_home = test["homescore"].astype(float).values
        y_away = test["awayscore"].astype(float).values

        pred_home = home_model.predict(X)
        pred_away = away_model.predict(X)

        rows.append(
            {
                "model": label,
                "year": int(y),
                "n": int(test.shape[0]),
                "mae_home": mae(y_home, pred_home),
                "mae_away": mae(y_away, pred_away),
                "mae_total": mae(y_home + y_away, pred_home + pred_away),
            }
        )

    return pd.DataFrame(rows)


def build_tables() -> None:
    dirs = _ensure_report_dirs()

    # -----------------------
    # 1) Model metrics table (World Cups)
    # -----------------------
    player_home = load(os.path.join(MODELSDIR, "playerhomegoals.joblib"))
    player_away = load(os.path.join(MODELSDIR, "playerawaygoals.joblib"))
    hist_home = load(os.path.join(MODELSDIR, "historyhomegoals.joblib"))
    hist_away = load(os.path.join(MODELSDIR, "historyawaygoals.joblib"))

    player_cols = _load_feature_list(os.path.join(MODELSDIR, "playerfeaturecols.csv"))
    hist_cols = _load_feature_list(os.path.join(MODELSDIR, "historyfeaturecols.csv"))

    player_df = pd.read_csv(PROCESSEDPATHS["modeltableplayertrain"])
    hist_df = pd.read_csv(PROCESSEDPATHS["modeltablehistorytrain"])

    player_wc = _evaluate_world_cup_mae(player_df, player_cols, player_home, player_away, label="PLAYER")
    hist_wc = _evaluate_world_cup_mae(hist_df, hist_cols, hist_home, hist_away, label="HISTORY")

    model_mae = pd.concat([player_wc, hist_wc], ignore_index=True).sort_values(["model", "year"])
    model_mae_path_metrics = os.path.join(dirs["metrics"], "model_metrics_worldcups.csv")
    model_mae.to_csv(model_mae_path_metrics, index=False)

    model_mae_path_tables = os.path.join(dirs["tables"], "model_mae_worldcups.csv")
    model_mae.to_csv(model_mae_path_tables, index=False)

    # -----------------------
    # 2) Plot-ready Monte Carlo tables
    # -----------------------
    champ = pd.read_csv(os.path.join(DATAPROCESSEDMONTECARLO, "mc_champion_odds.csv"))
    champ = champ.sort_values("pwin", ascending=False).head(15).copy()
    champ.to_csv(os.path.join(dirs["tables"], "champion_odds_top15.csv"), index=False)

    reach = pd.read_csv(os.path.join(DATAPROCESSEDMONTECARLO, "mc_reach_round_odds.csv"))
    # Focus list: host nations + a few top favorites (edit anytime)
    focus = {"Canada", "United States", "Mexico", "Brazil", "Argentina", "France", "England", "Germany", "Spain"}
    reach_focus = reach[reach["team"].isin(focus)].copy()
    reach_focus.to_csv(os.path.join(dirs["tables"], "reach_round_odds_focus.csv"), index=False)

    group_exp = pd.read_csv(os.path.join(DATAPROCESSEDMONTECARLO, "mc_group_stage_expectations.csv"))
    group_top = group_exp.sort_values("pts", ascending=False).head(15).copy()
    group_top.to_csv(os.path.join(dirs["tables"], "group_stage_expectations_top15.csv"), index=False)

    # -----------------------
    # 3) Optional manifest (helps debugging / reproducibility)
    # -----------------------
    manifest = pd.DataFrame(
        [
            {"item": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
            {"item": "champion_rows", "value": int(champ.shape[0])},
            {"item": "reach_focus_rows", "value": int(reach_focus.shape[0])},
            {"item": "group_top_rows", "value": int(group_top.shape[0])},
            {"item": "model_mae_rows", "value": int(model_mae.shape[0])},
        ]
    )
    manifest.to_csv(os.path.join(dirs["metrics"], "metrics_manifest.csv"), index=False)


def main() -> None:
    build_tables()
    print(f"Saved report tables to: {os.path.join(REPORTSDIR, 'tables')}")
    print(f"Saved metrics outputs to: {os.path.join(REPORTSDIR, 'metrics')}")


if __name__ == "__main__":
    main()