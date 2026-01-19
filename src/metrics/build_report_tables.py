"""
src/metrics/build_report_tables.py

Build plot-ready report tables under:
- reports/tables/   (inputs to plotting)
- reports/metrics/  (raw evaluation outputs)

Design goal:
Plot scripts should ONLY read from reports/tables and write to reports/figures.

This version is aligned with the current pipeline:
- train.py saves models as:
  - player_home_goals.joblib
  - player_away_goals.joblib
  - history_home_goals.joblib
  - history_away_goals.joblib
  - plus feature lists: player_feature_cols.csv, history_feature_cols.csv [file:734]
- match_features/train tables use: home_score / away_score columns [file:739][file:734]
- Monte Carlo outputs are written by run_monte_carlo.py as:
  - mc_champion_odds.csv  (columns: champion, p_win)
  - mc_reach_round_odds.csv (columns include: team, p_win)
  - mc_group_stage_expectations.csv (columns include: team, pts, ...) [file:729]
"""

import os
from datetime import datetime

import pandas as pd
from joblib import load

from config import (
    REPORTS_DIR,
    MODELS_DIR,
    DATA_PROCESSED_MONTE_CARLO,
    PROCESSED_PATHS,
    WC_TEST_YEARS,
)


# -----------------------
# Helpers
# -----------------------

def _ensure_report_dirs() -> dict:
    """Create the report subfolders if missing."""
    tables_dir = os.path.join(REPORTS_DIR, "tables")
    metrics_dir = os.path.join(REPORTS_DIR, "metrics")
    figures_dir = os.path.join(REPORTS_DIR, "figures")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return {"tables": tables_dir, "metrics": metrics_dir, "figures": figures_dir}


def _load_feature_list(path: str) -> list[str]:
    """Load the feature column list saved during training."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature list not found: {path}")
    return pd.read_csv(path, header=None)[0].astype(str).tolist()


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    """
    Read a CSV if it exists; otherwise return None.

    Why:
    - Monte Carlo might still be running (outputs not written yet).
    - Keeps the report build step from crashing.
    """
    if not os.path.exists(path):
        print(f"[build_report_tables] Missing file (skipping): {path}")
        return None
    return pd.read_csv(path)


def _mae(y_true, y_pred) -> float:
    """Mean Absolute Error without extra dependencies."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    return float((y_true - y_pred).abs().mean())


def _evaluate_world_cup_mae(
    df: pd.DataFrame,
    feature_cols: list[str],
    home_model,
    away_model,
    label: str,
) -> pd.DataFrame:
    """
    Evaluate MAE for World Cup matches by year.

    Inputs:
    - df: training table produced by match_features.py (player or history)
    - feature_cols: list saved during training
    - home_model, away_model: fitted models

    Output columns:
    - model, year, n, mae_home, mae_away, mae_total
    """
    outcols = ["model", "year", "n", "mae_home", "mae_away", "mae_total"]

    if "tournament" not in df.columns:
        return pd.DataFrame(columns=outcols)

    wc = df[df["tournament"].astype(str).str.strip().eq("World Cup")].copy()
    if wc.empty:
        return pd.DataFrame(columns=outcols)

    wc["year"] = pd.to_datetime(wc["date"], errors="coerce").dt.year

    rows = []
    for y in WC_TEST_YEARS:
        test = wc[wc["year"] == y].copy()
        if test.empty:
            continue

        # Align feature columns to exactly what the model expects.
        X = test.reindex(columns=feature_cols, fill_value=0.0).astype(float)

        # IMPORTANT: current canonical target names
        y_home = test["home_score"].astype(float).values
        y_away = test["away_score"].astype(float).values

        pred_home = home_model.predict(X)
        pred_away = away_model.predict(X)

        rows.append(
            {
                "model": label,
                "year": int(y),
                "n": int(test.shape[0]),
                "mae_home": _mae(y_home, pred_home),
                "mae_away": _mae(y_away, pred_away),
                "mae_total": _mae(y_home + y_away, pred_home + pred_away),
            }
        )

    return pd.DataFrame(rows, columns=outcols)


def _normalize_probability_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize probability column name to `p_win`.

    run_monte_carlo.py writes `p_win` (not `pwin`). [file:729]
    """
    out = df.copy()
    if "pwin" in out.columns and "p_win" not in out.columns:
        out = out.rename(columns={"pwin": "p_win"})
    if "poccurs" in out.columns and "p_occurs" not in out.columns:
        out = out.rename(columns={"poccurs": "p_occurs"})
    return out


# -----------------------
# Main build
# -----------------------

def build_tables() -> None:
    dirs = _ensure_report_dirs()

    # -----------------------
    # 1) Model metrics table (World Cups)
    # -----------------------
    player_home_path = os.path.join(MODELS_DIR, "player_home_goals.joblib")
    player_away_path = os.path.join(MODELS_DIR, "player_away_goals.joblib")
    hist_home_path = os.path.join(MODELS_DIR, "history_home_goals.joblib")
    hist_away_path = os.path.join(MODELS_DIR, "history_away_goals.joblib")

    if not all(os.path.exists(p) for p in [player_home_path, player_away_path, hist_home_path, hist_away_path]):
        missing = [p for p in [player_home_path, player_away_path, hist_home_path, hist_away_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing trained model files: {missing}")

    player_home = load(player_home_path)
    player_away = load(player_away_path)
    hist_home = load(hist_home_path)
    hist_away = load(hist_away_path)

    player_cols = _load_feature_list(os.path.join(MODELS_DIR, "player_feature_cols.csv"))
    hist_cols = _load_feature_list(os.path.join(MODELS_DIR, "history_feature_cols.csv"))

    player_df = pd.read_csv(PROCESSED_PATHS["model_table_player_train"])
    hist_df = pd.read_csv(PROCESSED_PATHS["model_table_history_train"])

    player_wc = _evaluate_world_cup_mae(player_df, player_cols, player_home, player_away, label="PLAYER")
    hist_wc = _evaluate_world_cup_mae(hist_df, hist_cols, hist_home, hist_away, label="HISTORY")

    model_mae = pd.concat([player_wc, hist_wc], ignore_index=True)
    if not model_mae.empty:
        model_mae = model_mae.sort_values(["model", "year"]).reset_index(drop=True)

    model_mae.to_csv(os.path.join(dirs["metrics"], "model_metrics_worldcups.csv"), index=False)
    model_mae.to_csv(os.path.join(dirs["tables"], "model_mae_worldcups.csv"), index=False)

    # -----------------------
    # 2) Plot-ready Monte Carlo tables
    # -----------------------
    # run_monte_carlo.py outputs these names. [file:729]
    champ_path = os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_champion_odds.csv")
    reach_path = os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_reach_round_odds.csv")
    group_path = os.path.join(DATA_PROCESSED_MONTE_CARLO, "mc_group_stage_expectations.csv")

    champ = _safe_read_csv(champ_path)
    champ_focus = pd.DataFrame()

    if champ is not None:
        champ = _normalize_probability_column(champ)
        # Expected columns from run_monte_carlo.py: champion, p_win [file:729]
        if "p_win" in champ.columns and "champion" in champ.columns:
            champ_focus = champ.sort_values(["p_win", "champion"], ascending=[False, True]).head(15).copy()
            champ_focus.to_csv(os.path.join(dirs["tables"], "champion_odds_top15.csv"), index=False)
        else:
            print(f"[build_report_tables] champion table missing columns. Have: {champ.columns.tolist()}")

    reach = _safe_read_csv(reach_path)
    reach_focus = pd.DataFrame()

    if reach is not None:
        reach = _normalize_probability_column(reach)
        # Expected: team + reachedR16.. + p_win (renamed from wonFINAL) [file:729]
        if "team" in reach.columns:
            focus = {
                "Canada", "United States", "Mexico",
                "Brazil", "Argentina", "France", "England", "Germany", "Spain",
            }
            reach_focus = reach[reach["team"].astype(str).isin(focus)].copy()
            reach_focus.to_csv(os.path.join(dirs["tables"], "reach_round_odds_focus.csv"), index=False)
        else:
            print(f"[build_report_tables] reach table missing 'team'. Have: {reach.columns.tolist()}")

    group_exp = _safe_read_csv(group_path)
    group_top = pd.DataFrame()

    if group_exp is not None:
        # Expected: team, pts, gd, gf, ga, rank_in_group [file:729]
        if "pts" in group_exp.columns and "team" in group_exp.columns:
            group_top = group_exp.sort_values(["pts", "gd", "gf", "team"], ascending=[False, False, False, True]).head(15).copy()
            group_top.to_csv(os.path.join(dirs["tables"], "group_stage_expectations_top15.csv"), index=False)
        else:
            print(f"[build_report_tables] group expectations missing columns. Have: {group_exp.columns.tolist()}")

    # -----------------------
    # 3) Optional manifest (debugging / reproducibility)
    # -----------------------
    manifest = pd.DataFrame(
        [
            {"item": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
            {"item": "model_mae_rows", "value": int(model_mae.shape[0])},
            {"item": "champion_rows", "value": int(champ.shape[0]) if isinstance(champ, pd.DataFrame) else 0},
            {"item": "champion_top15_rows", "value": int(champ_focus.shape[0]) if isinstance(champ_focus, pd.DataFrame) else 0},
            {"item": "reach_focus_rows", "value": int(reach_focus.shape[0]) if isinstance(reach_focus, pd.DataFrame) else 0},
            {"item": "group_top_rows", "value": int(group_top.shape[0]) if isinstance(group_top, pd.DataFrame) else 0},
        ]
    )
    manifest.to_csv(os.path.join(dirs["metrics"], "metrics_manifest.csv"), index=False)


def main() -> None:
    build_tables()
    print(f"Saved report tables to: {os.path.join(REPORTS_DIR, 'tables')}")
    print(f"Saved metrics outputs to: {os.path.join(REPORTS_DIR, 'metrics')}")


if __name__ == "__main__":
    main()