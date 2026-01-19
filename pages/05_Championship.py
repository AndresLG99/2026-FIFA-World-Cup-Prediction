from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return pd.read_csv(path)


def coerce_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def match_card(r: pd.Series) -> str:
    match_no = r.get("match_no", "")
    date = r.get("date", "")
    if pd.notna(date):
        date = pd.to_datetime(date).strftime("%Y-%m-%d")
    else:
        date = ""

    home_team = str(r.get("home_team", ""))
    away_team = str(r.get("away_team", ""))

    hs = r.get("home_score_sim", None)
    a_s = r.get("away_score_sim", None)
    hs = "" if hs is None or pd.isna(hs) else str(int(hs))
    a_s = "" if a_s is None or pd.isna(a_s) else str(int(a_s))

    winner = str(r.get("winner", ""))
    win_method = str(r.get("win_method", ""))
    model_used = str(r.get("model_used", ""))

    header_bits = []
    if pd.notna(match_no) and str(match_no) != "":
        try:
            header_bits.append(f"Match {int(match_no)}")
        except Exception:
            header_bits.append(f"Match {match_no}")
    if date:
        header_bits.append(date)

    header_txt = " · ".join(header_bits)

    return f"""
    <div class="match-card">
      <div class="match-header">{header_txt}</div>

      <div class="team-row">
        <div class="team-name">{home_team}</div>
        <div class="score">{hs}</div>
      </div>

      <div class="team-row">
        <div class="team-name">{away_team}</div>
        <div class="score">{a_s}</div>
      </div>

      <div class="match-footer">
        Winner: <span class="winner">{winner}</span> · {win_method} · Model: {model_used}
      </div>
    </div>
    """


st.set_page_config(page_title="Championship", layout="wide")
st.title("Championship")

summary_path = os.path.join("data", "processed", "tournament_single_run", "tournament_summary.csv")
knockout_path = os.path.join("data", "processed", "tournament_single_run", "tournament_knockout_matches.csv")

summary = safe_read_csv(summary_path)
knockout = coerce_date(safe_read_csv(knockout_path), "date")

champion = str(summary.loc[0, "champion"]) if ("champion" in summary.columns and len(summary) > 0) else "—"

# ---- filters (same method: short search right, placeholder only) ----
st.subheader("Bracket", divider="gray")

df = knockout.copy()

stage_order = [("QF", "Quarterfinals"), ("SF", "Semifinals"), ("FINAL", "Final")]

round_cols_html = []
for stage_code, stage_title in stage_order:
    sdf = df[df["stage"] == stage_code].copy()
    if "match_no" in sdf.columns:
        sdf = sdf.sort_values(["match_no"])
    cards_html = "".join(match_card(r) for _, r in sdf.iterrows())

    round_cols_html.append(
        f"""
        <div class="round-col">
          <div class="round-title">{stage_title}</div>
          {cards_html if cards_html else '<div class="empty-col">No matches</div>'}
        </div>
        """
    )

# ---- One self-contained HTML doc for components.html ----
html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --card-bg: rgba(255,255,255,0.03);
    --card-border: rgba(255,255,255,0.14);
    --text-dim: rgba(255,255,255,0.75);
    --line: rgba(255,255,255,0.18);
  }}

  body {{
    margin: 0;
    padding: 0;
    background: transparent;
    color: white;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }}

  .bracket-wrap {{
    display: flex;
    gap: 26px;
    align-items: flex-start;
    justify-content: center;
    width: fit-content;
    margin: 0 auto;
    padding: 8px 4px 4px 4px;
  }}

  .round-col {{
    width: 270px;
    position: relative;
  }}

  .round-title {{
    text-align: center;
    font-weight: 900;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
    opacity: 0.95;
  }}

  /* --- decorative connectors between columns --- */
  .round-col::after {{
    content: "";
    position: absolute;
    top: 46px;                 /* starts below title */
    right: -13px;              /* halfway into the gap */
    width: 13px;
    height: calc(100% - 60px);
    border-top: 2px solid var(--line);
    border-right: 2px solid var(--line);
    border-bottom: 2px solid var(--line);
    border-radius: 8px;
    opacity: 0.7;
  }}
  .round-col:last-child::after {{
    display: none;
  }}

  .match-card {{
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 10px 10px 8px 10px;
    margin-bottom: 12px;
    background: var(--card-bg);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    position: relative;
  }}

  /* Connector stub from each card to the next column */
  .match-card::after {{
    content: "";
    position: absolute;
    top: 50%;
    right: -18px;
    width: 18px;
    border-top: 2px solid var(--line);
    opacity: 0.7;
  }}

  /* Don’t draw stubs on last column */
  .round-col:last-child .match-card::after {{
    display: none;
  }}

  .match-header {{
    font-size: 0.78rem;
    opacity: 0.75;
    margin-bottom: 8px;
    text-align: center;
  }}

  .team-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 2px;
  }}

  .team-name {{
    width: 205px;
    text-align: left;
    font-weight: 650;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .score {{
    width: 34px;
    text-align: right;
    font-weight: 900;
    opacity: 0.95;
  }}

  .match-footer {{
    font-size: 0.78rem;
    opacity: 0.78;
    margin-top: 8px;
    text-align: center;
  }}

  .winner {{
    font-weight: 900;
  }}

  .empty-col {{
    opacity: 0.6;
    text-align: center;
    padding: 8px 0;
  }}

  .champion-banner {{
    margin: 14px auto 0 auto;
    text-align: center;
    padding: 14px 10px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.18);
    background: linear-gradient(90deg, rgba(255,215,0,0.14), rgba(255,255,255,0.03));
    box-shadow: 0 10px 26px rgba(0,0,0,0.30);
  }}

  .champion-title {{
    font-size: 0.95rem;
    letter-spacing: 0.10em;
    font-weight: 900;
    opacity: 0.85;
  }}

  .champion-name {{
    font-size: 2.2rem;
    font-weight: 1000;
    margin-top: 6px;
  }}
</style>
</head>

<body>
  <div class="bracket-wrap">
    {''.join(round_cols_html)}
  </div>

  <div class="champion-banner">
    <div class="champion-title">CHAMPION</div>
    <div class="champion-name">{champion}</div>
  </div>
</body>
</html>
"""

# height: large enough for R32 column (16 cards) + banner; tweak if needed
components.html(html_doc, height=1050, scrolling=True)

st.caption("Sources: tournament_single_run/tournament_knockout_matches.csv and tournament_summary.csv.")