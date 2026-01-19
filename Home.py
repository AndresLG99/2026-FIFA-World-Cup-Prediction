from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

from config import PROCESSED_PATHS


# -----------------------------
# helpers
# -----------------------------
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


def format_date(d: pd.Timestamp) -> str:
    if pd.isna(d):
        return "Unknown date"
    return d.strftime("%a, %b %d, %Y")


def to_int_no_decimals(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round(0).astype("Int64")


def country_label_html(country: str) -> str:
    """
    Create the colored country label exactly as requested.
    If country isn't one of USA/Mexico/Canada, fall back to plain text.
    """
    c = (country or "").strip()
    if c in {"United States", "USA", "U.S.A.", "US"}:
        return (
            "<span>"
            "<span style='color:#1e3a8a; font-weight:800;'>U</span>"
            "<span style='color:#ffffff; font-weight:800;'>S</span>"
            "<span style='color:#dc2626; font-weight:800;'>A</span>"
            "</span>"
        )

    if c == "Mexico":
        return (
            "<span>"
            "<span style='color:#16a34a; font-weight:800;'>ME</span>"
            "<span style='color:#ffffff; font-weight:800;'>XI</span>"
            "<span style='color:#dc2626; font-weight:800;'>CO</span>"
            "</span>"
        )

    if c in {"Canada", "Canda"}:  # tolerate your spelling
        return (
            "<span>"
            "<span style='color:#ffffff; font-weight:800;'>CA</span>"
            "<span style='color:#dc2626; font-weight:800;'>NA</span>"
            "<span style='color:#ffffff; font-weight:800;'>DA</span>"
            "</span>"
        )

    return f"<span style='font-weight:800;'>{c}</span>"


def day_country_header(date_value: pd.Timestamp, country: str) -> None:
    date_text = format_date(date_value)
    country_html = country_label_html(country)

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top: 0.75rem; margin-bottom: 1.25rem;">
          <div style="font-size:2.0rem; font-weight:800; line-height:1.1;">{date_text}</div>
          <div style="font-size:1.2rem; font-weight:800; line-height:1.1;">{country_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_match_row(home_team: str, away_team: str) -> None:
    # Smaller side margins + tighter team/vs/team widths
    left_margin, left_col, mid_col, right_col, right_margin = st.columns(
        [0.6, 2.4, 0.6, 2.4, 0.6],
        gap="small",
    )

    with left_col:
        st.markdown(f"<div style='text-align:left;'>{home_team}</div>", unsafe_allow_html=True)
    with mid_col:
        st.markdown("<div style='text-align:center;'>vs</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown(f"<div style='text-align:right;'>{away_team}</div>", unsafe_allow_html=True)


# -----------------------------
# page config
# -----------------------------
st.set_page_config(
    page_title="Home",
    layout="wide",
)

# -----------------------------
# title + intro
# -----------------------------
st.title("2026 FIFA World Cup Prediction")

st.markdown(
    """
The **FIFA World Cup** is the premier international men’s football tournament, contested by national teams from FIFA’s member associations.  
The **2026 FIFA World Cup** will be hosted by **Canada, Mexico, and the United States**, and it will be the first edition expanded to **48 teams**.
"""
)

# -----------------------------
# venues (TV-style, grouped by country, NO IMAGES)
# -----------------------------
st.header("Stadiums")
st.caption("TV-style venue list. Expand a stadium to see details.")

venues_path = os.path.join("data", "raw", "main", "stadiums_2026.csv")
venues_df = safe_read_csv(venues_path)

required_cols = {"stadium", "city", "country"}
missing_cols = required_cols - set(venues_df.columns)
if missing_cols:
    st.error(f"`{venues_path}` is missing columns: {sorted(missing_cols)}")
    st.stop()

venues_df = venues_df.copy()
venues_df["country"] = venues_df["country"].astype(str)
venues_df["city"] = venues_df["city"].astype(str)
venues_df["stadium"] = venues_df["stadium"].astype(str)

country_options = sorted(venues_df["country"].dropna().unique().tolist())
selected_countries = st.multiselect(
    "Filter by country",
    options=country_options,
    default=country_options,
)

filtered_venues_df = venues_df[venues_df["country"].isin(selected_countries)].copy()
filtered_venues_df = filtered_venues_df.sort_values(["country", "city", "stadium"], na_position="last")

for country, country_group in filtered_venues_df.groupby("country", dropna=False):
    country_label = "Unknown country" if pd.isna(country) else str(country)
    st.subheader(country_label, divider="gray")

    for _, r in country_group.iterrows():
        stadium = r["stadium"]
        city = r["city"]

        with st.expander(f"{stadium} — {city}", expanded=False):
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.write("**Stadium:**", stadium)
                st.write("**City:**", city)
            with col_right:
                st.write("**Country:**", country_label)

# -----------------------------
# tv_schedule (resolved fixtures)
# -----------------------------
st.header("Schedule")

fixtures_path = PROCESSED_PATHS["fixtures_2026_resolved"]
fixtures_df = safe_read_csv(fixtures_path)
fixtures_df = coerce_date(fixtures_df, "date")

required_fixture_cols = ["date", "home_team", "away_team"]
missing_fixture_cols = [c for c in required_fixture_cols if c not in fixtures_df.columns]
if missing_fixture_cols:
    st.error(f"`{fixtures_path}` is missing required columns: {missing_fixture_cols}")
    st.stop()

# Optional scores (not displayed per your request, but kept available)
has_scores = ("home_score" in fixtures_df.columns) and ("away_score" in fixtures_df.columns)
if has_scores:
    fixtures_df["home_score"] = to_int_no_decimals(fixtures_df["home_score"])
    fixtures_df["away_score"] = to_int_no_decimals(fixtures_df["away_score"])

min_date = fixtures_df["date"].min()
max_date = fixtures_df["date"].max()

if not pd.isna(min_date) and not pd.isna(max_date):
    date_range = st.slider(
        "Filter by date range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM-DD",
    )
    schedule_df = fixtures_df[
        (fixtures_df["date"] >= date_range[0]) & (fixtures_df["date"] <= date_range[1])
    ].copy()
else:
    st.warning("Dates could not be parsed from fixtures; showing unsorted list.")
    schedule_df = fixtures_df.copy()

# sort for stable TV feel
sort_cols = [c for c in ["date", "country"] if c in schedule_df.columns]
if sort_cols:
    schedule_df = schedule_df.sort_values(sort_cols, na_position="last")

days = list(schedule_df["date"].dt.date.dropna().unique())

if not days:
    st.dataframe(
        schedule_df[["date", "home_team", "away_team"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    for day in days:
        day_df = schedule_df[schedule_df["date"].dt.date == day].copy()

        # Group by match host country within the day (Mexico/Canada/USA blocks)
        if "country" in day_df.columns:
            for country, country_group in day_df.groupby("country", dropna=False):
                country_label = "Unknown country" if pd.isna(country) else str(country)

                # date + country on same line
                day_country_header(pd.Timestamp(day), country_label)

                for _, r in country_group.iterrows():
                    render_match_row(str(r["home_team"]), str(r["away_team"]))

                st.divider()
        else:
            # If no country column exists, still show date header
            day_country_header(pd.Timestamp(day), "")
            for _, r in day_df.iterrows():
                render_match_row(str(r["home_team"]), str(r["away_team"]))

st.caption("Schedule shown from the resolved fixtures file produced by the playoff resolver.")