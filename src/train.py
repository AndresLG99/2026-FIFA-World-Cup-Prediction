# Train the TWO-MODEL system:
# - Model A: player-features (only when both teams have player features)
# - Model B: match-history features (fallback when player features are missing)

"""
What train.py does:

- Loads the processed model tables created by match_features.py.
- Trains 4 models:
  1) Player model -> home goals
  2) Player model -> away goals
  3) History model -> home goals
  4) History model -> away goals
- Saves those models to disk so prediction scripts can reuse them.
- Optionally evaluates on World Cup years (2014/2018/2022).
"""

import os
import pandas as pd

from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

from config import (
    MODELS_DIR,
    PROCESSED_PATHS,
    WC_TEST_YEARS,
)


# -----------------------
# Utility helpers
# -----------------------

def pick_feature_columns(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    """
    Select numeric feature columns automatically, excluding obvious non-features.

    Impact:
    - Keeps train.py resilient if new features are added later.
    """
    feature_cols: list[str] = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    return feature_cols


def fit_poisson_model(train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> Pipeline:
    """
    Fit a Poisson regression model (good for count targets like goals).

    Notes:
    - StandardScaler is used to stabilize optimization when features have different scales.
    """
    X = train_df[feature_cols].astype(float)
    y = train_df[target_col].astype(float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", PoissonRegressor(alpha=0.01, max_iter=2000)),
        ]
    )
    model.fit(X, y)
    return model


def evaluate_mae(model: Pipeline, df: pd.DataFrame, feature_cols: list[str], target_col: str) -> float:
    """
    Evaluate Mean Absolute Error (MAE) on a dataset.
    """
    X = df[feature_cols].astype(float)
    y_true = df[target_col].astype(float)
    y_pred = model.predict(X)
    return float(mean_absolute_error(y_true, y_pred))


# -----------------------
# Training: Player model
# -----------------------

def train_player_models() -> dict:
    """
    Train Model A (player-features) on model_table_player_train.csv.

    Player training table should already be filtered so both teams have player features.
    """
    df = pd.read_csv(PROCESSED_PATHS["model_table_player_train"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    drop_cols = [
        "date", "home_team", "away_team", "tournament", "country",
        "home_score", "away_score", "neutral",
    ]

    feature_cols = pick_feature_columns(df, drop_cols)

    home_model = fit_poisson_model(df, feature_cols, "home_score")
    away_model = fit_poisson_model(df, feature_cols, "away_score")

    return {
        "feature_cols": feature_cols,
        "home_model": home_model,
        "away_model": away_model,
        "train_rows": int(df.shape[0]),
    }


# -----------------------
# Training: History model
# -----------------------

def train_history_models() -> dict:
    """
    Train Model B (history-features) on model_table_history_train.csv.
    """
    df = pd.read_csv(PROCESSED_PATHS["model_table_history_train"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    drop_cols = [
        "date", "home_team", "away_team", "tournament", "country",
        "home_score", "away_score", "neutral",
    ]

    feature_cols = pick_feature_columns(df, drop_cols)

    home_model = fit_poisson_model(df, feature_cols, "home_score")
    away_model = fit_poisson_model(df, feature_cols, "away_score")

    return {
        "feature_cols": feature_cols,
        "home_model": home_model,
        "away_model": away_model,
        "train_rows": int(df.shape[0]),
    }


# -----------------------
# Evaluation on World Cups
# -----------------------

def eval_on_world_cups(
    model_home: Pipeline,
    model_away: Pipeline,
    feature_cols: list[str],
    df_all: pd.DataFrame,
    label: str,
) -> None:
    """
    Evaluate a model on World Cup matches by year.

    This is a lightweight sanity-check metric printout (MAE).
    """
    if "tournament" not in df_all.columns:
        print(f"[{label}] No tournament column; skipping WC eval.")
        return

    wc_df = df_all[df_all["tournament"].astype(str).str.strip().eq("World Cup")].copy()
    wc_df["year"] = pd.to_datetime(wc_df["date"], errors="coerce").dt.year

    for y in WC_TEST_YEARS:
        test = wc_df[wc_df["year"] == y].copy()
        if test.empty:
            print(f"[{label}] WC {y}: no rows found.")
            continue

        mae_h = evaluate_mae(model_home, test, feature_cols, "home_score")
        mae_a = evaluate_mae(model_away, test, feature_cols, "away_score")

        total_true = (test["home_score"].astype(float) + test["away_score"].astype(float))
        total_pred = model_home.predict(test[feature_cols].astype(float)) + model_away.predict(test[feature_cols].astype(float))
        mae_tot = float(mean_absolute_error(total_true, total_pred))

        print(f"[{label}] WC {y} | n={len(test)} | MAE home={mae_h:.3f} away={mae_a:.3f} total={mae_tot:.3f}")


# -----------------------
# Main runner
# -----------------------

def main() -> None:
    """
    End-to-end train + save.

    Output files saved under /models:
    - player_home_goals.joblib
    - player_away_goals.joblib
    - history_home_goals.joblib
    - history_away_goals.joblib
    - plus feature column lists as CSV
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    player = train_player_models()
    dump(player["home_model"], os.path.join(MODELS_DIR, "player_home_goals.joblib"))
    dump(player["away_model"], os.path.join(MODELS_DIR, "player_away_goals.joblib"))
    pd.Series(player["feature_cols"]).to_csv(
        os.path.join(MODELS_DIR, "player_feature_cols.csv"),
        index=False,
        header=False,
    )
    print(f"Saved player models. Train rows: {player['train_rows']}")

    history = train_history_models()
    dump(history["home_model"], os.path.join(MODELS_DIR, "history_home_goals.joblib"))
    dump(history["away_model"], os.path.join(MODELS_DIR, "history_away_goals.joblib"))
    pd.Series(history["feature_cols"]).to_csv(
        os.path.join(MODELS_DIR, "history_feature_cols.csv"),
        index=False,
        header=False,
    )
    print(f"Saved history models. Train rows: {history['train_rows']}")

    # Optional evaluation prints
    player_df = pd.read_csv(PROCESSED_PATHS["model_table_player_train"])
    history_df = pd.read_csv(PROCESSED_PATHS["model_table_history_train"])
    eval_on_world_cups(player["home_model"], player["away_model"], player["feature_cols"], player_df, label="PLAYER")
    eval_on_world_cups(history["home_model"], history["away_model"], history["feature_cols"], history_df, label="HISTORY")


if __name__ == "__main__":
    main()