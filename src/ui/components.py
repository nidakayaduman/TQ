"""Reusable Streamlit UI helpers."""

from __future__ import annotations

import streamlit as st

from ..constants import DISCLAIMER


def render_disclaimer() -> None:
    st.warning(DISCLAIMER)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    render_disclaimer()

