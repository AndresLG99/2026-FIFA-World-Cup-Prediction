"""
pages/03_Figures_and_Tables.py

Browse report figures and plot-ready tables created by the reporting pipeline.

Inputs:
- reports/figures/*.png  (produced by: python -m src.reports.make_figures)
- reports/tables/*.csv   (produced by: python -m src.metrics.build_report_tables) [file:726]

Outputs:
- Figure viewer (PNG)
- Table viewer (CSV) to support transparency/reproducibility.
"""

import os

import pandas as pd
import streamlit as st

from config import REPORTS_DIR  # [file:726]


@st.cache_data
def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV with Streamlit caching for speed."""
    return pd.read_csv(path)


def main() -> None:
    st.title("Figures & Tables")

    figs_dir = os.path.join(REPORTS_DIR, "figures")
    tables_dir = os.path.join(REPORTS_DIR, "tables")

    st.subheader("Figures (PNG)")
    if not os.path.exists(figs_dir):
        st.info("No figures folder found yet. Generate with: python -m src.reports.make_figures")
    else:
        pngs = sorted([p for p in os.listdir(figs_dir) if p.lower().endswith(".png")])
        if not pngs:
            st.info("No PNGs found. Generate with: python -m src.reports.make_figures")
        else:
            pick = st.selectbox("Select a figure", pngs)
            st.image(os.path.join(figs_dir, pick), use_container_width=True)

    st.subheader("Plot-ready tables (CSV)")
    if not os.path.exists(tables_dir):
        st.info("No tables folder found yet. Generate with: python -m src.metrics.build_report_tables")
    else:
        csvs = sorted([p for p in os.listdir(tables_dir) if p.lower().endswith(".csv")])
        if not csvs:
            st.info("No CSV tables found. Generate with: python -m src.metrics.build_report_tables")
        else:
            t = st.selectbox("Select a table", csvs)
            st.dataframe(_read_csv(os.path.join(tables_dir, t)), use_container_width=True)


if __name__ == "__main__":
    main()