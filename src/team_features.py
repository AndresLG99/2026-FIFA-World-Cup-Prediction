# This file turns players_master (many rows per player) into team_features (one row per national team),
# which you will later merge onto matches as home_* and away_*.

"""
What team_features.py does:
- A “player table” is like one card per player.
- A “team table” is like one card per team.
- This script stacks all the player cards for a team, then makes one “average card” for the team,
  giving more importance to players who played more minutes.
"""

import pandas as pd


PER90_COLS = [
    "gls_90_calc", "ast_90_calc", "xg_90_calc", "xag_90_calc",
    "prgp_90_calc", "prgc_90_calc", "prgr_90_calc",
]


def minutes_weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "Min") -> float:
    """
    Compute a minutes-weighted mean for one numeric column.

    Impact:
    - Players with more minutes influence the team “average card” more than fringe players.
    - Makes team strength estimates more stable than a simple mean.
    """
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    x = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).astype(float)

    if float(w.sum()) == 0.0:
        return 0.0

    return float((x * w).sum() / w.sum())


def build_team_features(players_master: pd.DataFrame, nation_code_map: pd.DataFrame) -> pd.DataFrame:
    """
    Convert players_master (many rows) into team_features (one row per national team).

    Steps:
    1) Validate required columns exist.
    2) Merge nation_code -> team_name using the mapping CSV.
    3) Drop rows where team_name is missing (cannot map player to a national team).
    4) Aggregate per team using minutes-weighted averages.

    Impact:
    - Produces the team-level inputs that feed the PLAYER model feature tables.
    - If this aggregation is wrong, Model A will either miss teams or learn the wrong strength signals.
    """
    # Validate inputs
    required_player_cols = ["nation_code", "Min"] + PER90_COLS
    missing = [c for c in required_player_cols if c not in players_master.columns]
    if missing:
        raise ValueError(f"players_master is missing columns: {missing}")

    required_map_cols = {"nation_code", "team_name"}
    if not required_map_cols.issubset(set(nation_code_map.columns)):
        raise ValueError("nation_code_map must have columns: nation_code, team_name")

    df = players_master.copy()
    m = nation_code_map.copy()

    # Clean keys
    df["nation_code"] = df["nation_code"].astype("string").str.strip()
    m["nation_code"] = m["nation_code"].astype("string").str.strip()
    m["team_name"] = m["team_name"].astype("string").str.strip()

    # Attach team_name using the mapping CSV
    df = df.merge(m, on="nation_code", how="left")

    # Keep only rows where team_name is known
    df = df[df["team_name"].notna() & (df["team_name"] != "")].copy()

    rows = []
    for team_name, g in df.groupby("team_name"):
        row = {"team_name": str(team_name)}
        row["n_rows_used"] = int(g.shape[0])

        # Optional: track unique players (helps explain results)
        if "Player" in g.columns:
            row["n_unique_players"] = int(g["Player"].astype(str).nunique())
        else:
            row["n_unique_players"] = pd.NA

        for col in PER90_COLS:
            row[col] = minutes_weighted_mean(g, col, weight_col="Min")

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("team_name").reset_index(drop=True)
    return out


def test_team_features(players_master: pd.DataFrame, nation_code_map: pd.DataFrame) -> None:
    """
    Quick test prints shapes and a sample of the output.
    """
    team_df = build_team_features(players_master, nation_code_map)
    print("team_features:", team_df.shape)
    sort_col = "n_unique_players" if "n_unique_players" in team_df.columns else "n_rows_used"
    print(team_df.sort_values(sort_col, ascending=False).head(10))


if __name__ == "__main__":
    # Local test runner
    from config import DATA_PATHS, MIN_PLAYER_MINUTES
    from src.data_loader import load_player_seasons
    from src.player_features import build_players_master
    from src.code_mapping import load_nation_code_map

    players = load_player_seasons()
    players_master = build_players_master(players, min_minutes=MIN_PLAYER_MINUTES)
    nation_map = load_nation_code_map(DATA_PATHS["nation_code_map"])

    test_team_features(players_master, nation_map)