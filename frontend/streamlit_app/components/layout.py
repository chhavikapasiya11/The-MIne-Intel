"""Composable UI helpers used across the Streamlit surface."""

from __future__ import annotations

import streamlit as st

import config


def render_page_header() -> None:
    """Render a simple header with context."""
    st.title(config.APP_TITLE)
    st.caption(config.APP_DESCRIPTION)

    st.markdown(
        """
        #### Input parameters
        - **CMRR** – Coal Mine Roof Rating (0-100). Higher values indicate stronger strata.
        - **PRSUP** – Percentage of the installed roof support currently carrying load.
        - **Depth of cover** – Overburden thickness measured in meters.
        - **Intersection diagonal** – Crosscut diagonal distance in meters.
        - **Mining height** – Height of the extracted seam in meters.

        #### Prediction
        The model estimates the expected roof fall rate for the supplied readings. Use the
        value to prioritize inspections and supplemental support.
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render sidebar content with concise instructions."""
    st.sidebar.header("Quick steps")
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <p>1. Confirm the backend API URL if it is not running locally.</p>
            <p>2. Enter the latest readings from the working face.</p>
            <p>3. Press <strong>Predict roof fall rate</strong> to fetch the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(prediction: float) -> None:
    """Display the prediction output."""
    st.success("Prediction ready")
    st.metric("Estimated roof fall rate", f"{prediction:.{config.DEFAULT_PRECISION}f}")
    st.caption(
        "Values are returned in the same units as the training label "
        "(`roof_fall_rate`)."
    )


def render_error(message: str) -> None:
    """Display a consistent error block."""
    st.error(message)

