"""
app.py

Streamlit multipage entrypoint (Home page).

Design goal:
- Keep app.py minimal.
- All “real” pages live under /pages and are auto-discovered by Streamlit.

Inputs:
- src/app/style.css (optional)
- Data assets are loaded inside individual pages.

Outputs:
- Streamlit UI (home page + navigation sidebar handled by Streamlit).
"""

import os
from pathlib import Path

import streamlit as st


def _load_css(css_path: str) -> None:
    """Load a CSS file into the Streamlit app if it exists."""
    if os.path.exists(css_path):
        css = Path(css_path).read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="2026 World Cup Prediction", layout="wide")
    _load_css(os.path.join("src", "app", "style.css"))

    st.title("2026 FIFA World Cup Prediction")
    st.write(
        "Use the sidebar to navigate through the pages: "
        "Predictions, Monte Carlo, and Figures & Tables."
    )


if __name__ == "__main__":
    main()
