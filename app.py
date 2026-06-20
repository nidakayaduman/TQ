"""Tender IQ Agentic Bid Advisor - Turkish Streamlit dashboard."""

from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.advisor.fallback_advisor import build_fallback_advisor
from src.advisor.forbidden_claim_detector import detect_forbidden_claims
from src.advisor.output_validator import validate_advisor_output
from src.advisor.prompt_builder import build_advisor_prompt
from src.constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from src.evaluation.backtest_runner import actual_rank_percentile, run_backtest
from src.evaluation.baseline_models import baseline_predictions
from src.evaluation.expert_review import expert_review_template
from src.evaluation.metrics import optimizer_metrics, price_corridor_metrics
from src.evaluation.segment_metrics import segment_level_metrics
from src.feature_masking import mask_actual_result_fields
from src.leakage_audit import audit_pre_reveal_input
from src.model_card import generate_model_card
from src.optimizer.recommendation_engine import rank_scenarios
from src.optimizer.scenario_generator import generate_candidate_scenarios
from src.optimizer.scenario_scorer import score_scenario
from src.optimizer.scenario_validator import validate_scenario
from src.reporting.audit_log import write_audit_event
from src.reporting.export_csv import dataframe_to_csv_bytes
from src.retrieval import RetrievalEngine, retrieval_quality
from src.schema import normalize_schema, schema_quality_summary, validate_schema
from src.split_strategy import temporal_split
from src.validation import validate_data_quality

ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "data" / "x_ilac_synthetic_tenders_2021_2025.csv"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

TURKISH_WARNING = (
    "Bu sistem gerçek kazanma olasılığı hesaplamaz. Veri yalnızca kazanılmış ihalelerden oluşur. "
    "Skor; geçmiş kazanılmış ihale profiline benzerlik, fiyat bandı uyumu, marj/risk dengesi ve "
    "model güvenini birlikte gösterir."
)

PWIN_PROXY_EXPLANATION = (
    "Bu MVP’de gerçek P-Win hesaplanmaz. Bunun yerine P-Win yerine kullanılabilecek bir karar destek göstergesi "
    "olarak Kazanılmış Profil Uyum Skoru kullanılır. Bu skor; emsal benzerlik, K-Means başarı profili, "
    "Isolation Forest uygunluğu, fiyat bandı uyumu, marj/risk dengesi ve model güveninden beslenir."
)

PAGE_NAMES = [
    "Ana Sayfa",
    "Veri Seti ve Kalite Kontrol",
    "Metodoloji",
    "Test İhalesi Simülatörü",
    "Senaryo Analizi",
    "Gerçek Sonuçla Karşılaştır",
    "Backtest Sonuçları",
    "Benzer İhaleler",
    "AI Danışman",
    "Raporlar ve Kontroller",
]

LEGACY_PAGE_LABELS_FOR_TESTS = ["Veri Yükleme ve Kalite Kontrol", "Raporlar ve Audit"]

st.set_page_config(
    page_title="Tender IQ Agentic Bid Advisor",
    page_icon="TI",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #fbfcff;
                --surface: rgba(255, 255, 255, 0.86);
                --surface-strong: #ffffff;
                --line: rgba(31, 41, 55, 0.10);
                --line-soft: rgba(99, 102, 241, 0.12);
                --text: #111827;
                --muted: #6b7280;
                --primary: #151827;
                --accent: #6d6be8;
                --accent-soft: #eef0ff;
                --blue: #5b7cfa;
                --cyan: #7ab8db;
                --purple: #8b7cf6;
                --green: #4f9d7a;
                --amber: #b8873d;
                --red: #c65d5d;
                --shadow: 0 24px 70px rgba(17, 24, 39, 0.08);
                --soft-shadow: 0 12px 36px rgba(17, 24, 39, 0.06);
            }
            .stApp, .app-bg {
                background:
                    radial-gradient(ellipse at 28% 3%, rgba(132, 118, 255, 0.22), transparent 36%),
                    radial-gradient(ellipse at 78% 12%, rgba(125, 196, 232, 0.18), transparent 34%),
                    linear-gradient(180deg, #ffffff 0%, #fbfcff 42%, #f7f9ff 100%);
                color: var(--text);
            }
            [data-testid='stHeader'] { background: transparent; }
            .block-container { max-width: 1320px; padding-top: 1.5rem; padding-bottom: 3.4rem; }
            [data-testid='stSidebar'] {
                background: rgba(255, 255, 255, 0.90);
                border-right: 1px solid var(--line);
                box-shadow: 10px 0 34px rgba(17, 24, 39, 0.035);
            }
            [data-testid='stSidebar'] .stRadio label { color: var(--primary); font-weight: 540; }
            [data-testid='stVerticalBlockBorderWrapper'] {
                background: var(--surface-strong);
                border: 1px solid var(--line);
                border-radius: 18px;
                box-shadow: var(--soft-shadow);
            }
            div[data-testid='stDataFrame'] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid var(--line);
                box-shadow: 0 10px 24px rgba(17, 24, 39, 0.04);
            }
            .brand-mark {
                width: 42px; height: 42px; display: grid; place-items: center;
                border-radius: 14px; color: var(--primary); font-weight: 820; letter-spacing: 0.02em;
                background: linear-gradient(135deg, #ffffff, #f0f2ff);
                border: 1px solid var(--line-soft);
                box-shadow: 0 14px 30px rgba(99, 102, 241, 0.10);
                margin-bottom: 0.85rem;
            }
            .sidebar-title { color: var(--primary); font-size: 1.04rem; font-weight: 760; letter-spacing: 0; }
            .sidebar-subtitle { color: var(--muted); font-size: 0.78rem; font-weight: 520; margin-top: 0.2rem; }
            .sidebar-status-stack { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.85rem 0 1rem; }
            .sidebar-note {
                color: var(--muted); font-size: 0.78rem; line-height: 1.45;
                padding: 0.8rem; border: 1px solid var(--line); border-radius: 16px;
                background: rgba(248, 250, 252, 0.78);
            }
            .eyebrow, .section-kicker {
                color: #7770d7; font-size: 0.68rem; font-weight: 680; letter-spacing: 0.13em;
                text-transform: uppercase; margin-bottom: 0.28rem;
            }
            .page-title {
                color: var(--primary); font-size: clamp(2.15rem, 3.2vw, 3.65rem); line-height: 1.02;
                font-weight: 680; margin: 0; letter-spacing: -0.02em;
            }
            .page-subtitle, .section-subtitle {
                color: var(--muted); max-width: 860px; font-size: 0.98rem; margin-top: 0.68rem; line-height: 1.65;
            }
            .scope-pill, .status-badge {
                display: inline-flex; align-items: center; gap: 0.38rem; padding: 0.32rem 0.62rem;
                border-radius: 999px; font-size: 0.73rem; font-weight: 620; border: 1px solid var(--line);
                background: rgba(255,255,255,0.86); color: var(--primary);
                box-shadow: 0 6px 16px rgba(17, 24, 39, 0.035);
            }
            .scope-pill { float: right; margin-top: 0.48rem; }
            .scope-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--green); box-shadow: 0 0 12px rgba(79,157,122,.22); }
            .status-success, .status-good { color: #2f765a; background: rgba(79,157,122,0.08); border-color: rgba(79,157,122,0.18); }
            .status-warning, .status-warn { color: #8a642c; background: rgba(184,135,61,0.08); border-color: rgba(184,135,61,0.18); }
            .status-danger, .status-bad { color: #9b4e4e; background: rgba(198,93,93,0.08); border-color: rgba(198,93,93,0.18); }
            .hero-card {
                position: relative; overflow: hidden; padding: clamp(2.2rem, 5vw, 5.2rem);
                border-radius: 28px; border: 1px solid rgba(99, 102, 241, 0.11);
                background:
                    radial-gradient(ellipse at 72% 4%, rgba(141, 127, 255, 0.25), transparent 36%),
                    radial-gradient(ellipse at 18% 18%, rgba(141, 208, 237, 0.20), transparent 36%),
                    linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
                box-shadow: var(--shadow); color: var(--primary);
            }
            .hero-card:after {
                content: ''; position: absolute; inset: 18% 5% auto auto; height: 220px; width: 460px;
                background: radial-gradient(ellipse, rgba(123, 109, 240, 0.18), transparent 68%);
                filter: blur(18px);
                pointer-events: none;
            }
            .hero-title { font-size: clamp(3.1rem, 7vw, 6.6rem); line-height: .95; font-weight: 560; letter-spacing: -0.055em; margin: 0; max-width: 920px; }
            .hero-subtitle { max-width: 760px; color: #5d6677; font-size: 1.08rem; line-height: 1.7; margin-top: 1.05rem; }
            .hero-badges { display: flex; flex-wrap: wrap; gap: 0.58rem; margin-top: 1.45rem; }
            .hero-badge {
                display: inline-flex; align-items: center; gap: 0.38rem; padding: 0.48rem 0.74rem; border-radius: 999px;
                color: #434a5e; background: rgba(255,255,255,0.78); border: 1px solid var(--line);
                font-size: 0.82rem; font-weight: 540; backdrop-filter: blur(12px);
            }
            .glass-card, .method-card, .model-card, .score-card, .scenario-card, .chat-shell {
                border-radius: 22px; border: 1px solid var(--line);
                background: var(--surface); box-shadow: var(--soft-shadow); backdrop-filter: blur(16px);
            }
            .glass-card { padding: 1.25rem; }
            .section-title { color: var(--primary); font-size: 1.18rem; font-weight: 650; margin: 0 0 .25rem; letter-spacing: -0.01em; }
            .metric-card {
                position: relative; overflow: hidden; min-height: 118px; padding: 1.05rem;
                border-radius: 20px; border: 1px solid var(--line);
                background: rgba(255,255,255,0.88); box-shadow: 0 10px 26px rgba(17, 24, 39, 0.045);
            }
            .metric-card:before { content: ''; position: absolute; inset: 0 0 auto 0; height: 2px; background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent, var(--blue)) 35%, white), transparent); }
            .metric-card-blue { --accent: var(--blue); }
            .metric-card-green { --accent: var(--green); }
            .metric-card-purple { --accent: var(--purple); }
            .metric-card-amber { --accent: var(--amber); }
            .metric-card-red { --accent: var(--red); }
            .metric-card-cyan { --accent: var(--cyan); }
            .metric-icon {
                width: 30px; height: 30px; display: inline-grid; place-items: center; border-radius: 10px;
                background: rgba(248,250,252,0.92); border: 1px solid var(--line); margin-bottom: 0.5rem;
                color: var(--primary); font-size: .88rem;
            }
            .metric-label { color: var(--muted); font-size: 0.7rem; font-weight: 620; text-transform: uppercase; letter-spacing: 0.08em; }
            .metric-value { color: var(--primary); font-size: 1.48rem; font-weight: 680; margin-top: 0.22rem; line-height: 1.12; overflow-wrap: anywhere; }
            .metric-note { color: var(--muted); font-size: 0.82rem; margin-top: 0.38rem; line-height: 1.38; }
            .warning-callout, .info-callout {
                border-radius: 22px; padding: 1rem 1.1rem; line-height: 1.55; box-shadow: 0 12px 28px rgba(17, 24, 39, 0.035);
            }
            .warning-callout { border: 1px solid rgba(184,135,61,0.18); background: rgba(255, 252, 244, 0.88); color: #705224; }
            .info-callout { border: 1px solid var(--line-soft); background: rgba(248, 249, 255, 0.92); color: #38405c; }
            .model-grid, .method-grid, .score-mini-grid {
                display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem;
            }
            .method-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .method-card, .model-card { padding: 1.18rem; min-height: 154px; position: relative; overflow: hidden; }
            .model-card:before, .method-card:before, .scenario-card:before {
                content: ''; position: absolute; inset: 0 0 auto 0; height: 2px; background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent, var(--blue)) 30%, white), transparent);
            }
            .model-card-blue, .method-card-blue { --accent: var(--cyan); }
            .model-card-purple, .method-card-purple { --accent: var(--purple); }
            .model-card-amber, .method-card-amber { --accent: var(--amber); }
            .model-card-green, .method-card-green { --accent: var(--green); }
            .model-card-cyan, .method-card-cyan { --accent: var(--cyan); }
            .model-icon, .method-number {
                width: 32px; height: 32px; display: inline-grid; place-items: center; border-radius: 11px;
                background: rgba(248,250,252,0.92); border: 1px solid var(--line); font-weight: 620; margin-bottom: .75rem;
            }
            .model-title, .method-title { color: var(--primary); font-weight: 650; font-size: 1rem; margin-bottom: .35rem; }
            .model-body, .method-body { color: var(--muted); font-size: .86rem; line-height: 1.48; }
            .score-card { padding: 1.15rem; background: #ffffff; color: var(--primary); }
            .score-value { font-size: 2.6rem; font-weight: 680; line-height: 1; color: inherit; }
            .score-label { font-weight: 620; margin-top: .35rem; }
            .formula-panel {
                border-radius: 24px; padding: 1.25rem; color: var(--primary);
                background: linear-gradient(135deg, #ffffff, #f5f6ff);
                border: 1px solid var(--line-soft);
                box-shadow: var(--soft-shadow);
            }
            .formula-title { font-size: 1.05rem; font-weight: 680; margin-bottom: .7rem; }
            .formula-line { display: flex; justify-content: space-between; gap: 1rem; padding: .48rem 0; border-top: 1px solid var(--line); font-weight: 560; color: #394150; }
            .scenario-card { padding: 1.15rem; min-height: 292px; position: relative; overflow: hidden; }
            .scenario-card-blue { --accent: var(--blue); }
            .scenario-card-green { --accent: var(--green); }
            .scenario-card-purple { --accent: var(--purple); }
            .scenario-card-amber { --accent: var(--amber); }
            .scenario-title { font-size: 1.02rem; font-weight: 650; color: var(--primary); margin-bottom: 0.25rem; }
            .scenario-price { font-size: 1.58rem; font-weight: 680; color: var(--primary); line-height: 1.1; }
            .scenario-row { display: flex; justify-content: space-between; gap: .8rem; border-top: 1px solid var(--line); padding-top: .46rem; margin-top: .46rem; color: var(--muted); font-size: .82rem; }
            .scenario-row b { color: var(--primary); }
            .chat-shell { overflow: hidden; background: rgba(255,255,255,0.92); }
            .chat-header {
                padding: 1rem; color: var(--primary); background: linear-gradient(135deg, #ffffff, #f3f5ff);
                border-bottom: 1px solid var(--line);
                display: flex; align-items: center; gap: .75rem;
            }
            .chat-avatar {
                width: 38px; height: 38px; display: grid; place-items: center; border-radius: 13px;
                background: #ffffff; border: 1px solid var(--line-soft); font-weight: 660;
            }
            .chat-header-title { font-weight: 680; font-size: 1.05rem; }
            .chat-header-subtitle { color: var(--muted); font-size: .82rem; margin-top: .15rem; line-height: 1.35; }
            .chat-body {
                min-height: 470px; max-height: 620px; overflow-y: auto; padding: 1rem;
                background: linear-gradient(180deg, rgba(251,252,255,0.92), rgba(255,255,255,0.92));
            }
            .chat-input-area { padding: .8rem 1rem 1rem; border-top: 1px solid var(--line); background: rgba(255,255,255,.90); }
            .quick-question button { border-radius: 999px !important; border-color: var(--line) !important; background: rgba(255,255,255,.94) !important; color: #394150 !important; box-shadow: none !important; }
            .warning-box { border: 1px solid rgba(184,135,61,0.18); background: rgba(255,252,244,0.88); color: #705224; border-radius: 22px; padding: .9rem 1rem; }
            .info-box { border: 1px solid var(--line-soft); background: rgba(248,249,255,0.92); color: #38405c; border-radius: 22px; padding: .9rem 1rem; }
            .nav-card { min-height: 168px; display: flex; flex-direction: column; justify-content: space-between; }
            .soft-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,.18), transparent); margin: 1.6rem 0; }
            .divider-space { margin-top: 1.35rem; }
            @media (max-width: 1100px) {
                .model-grid, .method-grid, .score-mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 760px) {
                .scope-pill { float: none; margin-bottom: .8rem; }
                .model-grid, .method-grid, .score-mini-grid { grid-template-columns: 1fr; }
                .hero-card { padding: 1.35rem; }
                .metric-value { font-size: 1.45rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()


def init_session_state_defaults() -> None:
    st.session_state.setdefault("llm_primary_label", DEFAULT_OPENROUTER_MODEL)
    st.session_state.setdefault("llm_fallback_labels", [])


init_session_state_defaults()


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return normalize_schema(pd.read_csv(SAMPLE_DATA))


@st.cache_data(show_spinner=False)
def cached_backtest(data: pd.DataFrame) -> pd.DataFrame:
    split = temporal_split(data)
    return run_backtest(pd.concat([split["train"], split["validation"]]), split["test"])


def load_active_data() -> pd.DataFrame:
    return st.session_state.get("active_data", load_default_data()).copy()


def get_split() -> dict[str, pd.DataFrame]:
    return temporal_split(load_active_data())


def get_history_frame() -> pd.DataFrame:
    split = get_split()
    return pd.concat([split["train"], split["validation"]], ignore_index=True)


def format_try(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"%{float(value):.1f}".replace(".", ",")


def format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}".replace(",", ".")


def format_score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "Hesaplanamadı"
    return f"{float(value):.1f}/100"


def fit_level(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "Hesaplanamadı"
    value = float(score)
    if value >= 70:
        return "Yüksek uyum"
    if value >= 45:
        return "Orta uyum"
    return "Düşük uyum"


def scenario_rank_comment(rank_pct: float | int | None) -> str:
    if rank_pct is None or pd.isna(rank_pct):
        return "Senaryo sıralaması hesaplanamadı."
    value = float(rank_pct)
    if value >= 90:
        return "Top %10: güçlü uyum."
    if value >= 80:
        return "Top %20: kabul edilebilir uyum."
    if value >= 50:
        return "Top %50: orta düzey uyum."
    return "Alt %50: senaryo skor mantığı bu ihale için zayıf kalmış olabilir."


def render_small_card(title: str, body: str, badge_html: str = "") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)
        if badge_html:
            st.markdown(badge_html, unsafe_allow_html=True)


def render_kv_card(title: str, rows: list[tuple[str, str]], note: str = "") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for label, value in rows:
            st.markdown(f"**{label}:** {value}")
        if note:
            st.caption(note)


def retrieval_quality_from_result(result: dict[str, Any], tender: dict[str, Any]) -> dict[str, float]:
    quality = result.get("retrieval_quality")
    if isinstance(quality, dict):
        return quality
    similar = result.get("similar", pd.DataFrame())
    if isinstance(similar, pd.DataFrame):
        return retrieval_quality(similar, tender)
    return {
        "topk_avg_similarity": 0.0,
        "product_group_match_rate": 0.0,
        "region_match_rate": 0.0,
        "quantity_band_match_rate": 0.0,
    }


def page_header(title: str, subtitle: str, eyebrow: str = "Tender IQ") -> None:
    st.markdown(
        f"""
        <span class="scope-pill"><span class="scope-dot"></span>Sadece kazanılmış ihale verisi</span>
        <div class="eyebrow">{escape(eyebrow)}</div>
        <h1 class="page-title">{escape(title)}</h1>
        <div class="page-subtitle">{escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)


def warning_box() -> None:
    st.markdown(
        f"""
        <div class='warning-callout'>
            <b>Önemli not:</b> {escape(TURKISH_WARNING)}
            Gerçek kazanma olasılığı için kazanılmış ve kaybedilmiş ihale verisi gerekir.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str = "", color: str = "blue", icon: str = "📊") -> None:
    st.markdown(
        f"""
        <div class='metric-card metric-card-{escape(color)}'>
            <div class='metric-icon'>{escape(icon)}</div>
            <div class="metric-label">{escape(title)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", color: str = "blue", icon: str = "📊") -> None:
    render_metric_card(label, value, note, color, icon)


def badge(text: str, status: str = "good") -> str:
    css = {
        "good": "status-success",
        "success": "status-success",
        "warn": "status-warning",
        "warning": "status-warning",
        "bad": "status-danger",
        "danger": "status-danger",
    }.get(status, "")
    return f'<span class="status-badge {css}">{escape(text)}</span>'


def info_callout(text: str, title: str | None = None) -> None:
    heading = f"<b>{escape(title)}</b> " if title else ""
    st.markdown(f"<div class='info-callout'>{heading}{escape(text)}</div>", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    kicker_html = f"<div class='section-kicker'>{escape(kicker)}</div>" if kicker else ""
    subtitle_html = f"<div class='section-subtitle'>{escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"{kicker_html}<div class='section-title'>{escape(title)}</div>{subtitle_html}",
        unsafe_allow_html=True,
    )


def glass_card(title: str, body: str, kicker: str = "", status_html: str = "") -> None:
    with st.container(border=True):
        if kicker:
            st.caption(kicker)
        st.markdown(f"**{title}**")
        st.caption(body)
        if status_html:
            st.markdown(status_html, unsafe_allow_html=True)


def build_gauge(score: float, title: str = "Skor") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "/100", "font": {"size": 34}},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "rgba(15, 23, 42, 0.18)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(220, 38, 38, 0.14)"},
                    {"range": [45, 70], "color": "rgba(217, 119, 6, 0.14)"},
                    {"range": [70, 100], "color": "rgba(22, 163, 74, 0.14)"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=15), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_method_grid(items: list[tuple[str, str]], colors: list[str] | None = None) -> None:
    colors = colors or ["blue", "purple", "green", "amber"]
    for start in range(0, len(items), 3):
        columns = st.columns(3, gap="medium")
        for offset, (title, body) in enumerate(items[start : start + 3]):
            idx = start + offset
            color = colors[idx % len(colors)]
            with columns[offset]:
                st.markdown(
                    f"""
                    <div class='method-card method-card-{escape(color)}'>
                        <div class='method-number'>{idx + 1}</div>
                        <div class='method-title'>{escape(title)}</div>
                        <div class='method-body'>{escape(body)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_model_grid(items: list[tuple[str, str, str, str]]) -> None:
    for start in range(0, len(items), 4):
        columns = st.columns(4, gap="medium")
        for offset, (icon, title, body, color) in enumerate(items[start : start + 4]):
            with columns[offset]:
                st.markdown(
                    f"""
                    <div class='model-card model-card-{escape(color)}'>
                        <div class='model-icon'>{escape(icon)}</div>
                        <div class='model-title'>{escape(title)}</div>
                        <div class='model-body'>{escape(body)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_formula_card() -> None:
    st.markdown(
        """
        <div class='formula-panel'>
            <div class='formula-title'>Senaryo Skoru</div>
            <div class='formula-line'><span>%30 Profil Uyumu</span><span>geçmiş kazanım profili</span></div>
            <div class='formula-line'><span>+ %25 Fiyat Bandı Uyumu</span><span>koridor hizası</span></div>
            <div class='formula-line'><span>+ %20 Marj Skoru</span><span>karlılık sağlığı</span></div>
            <div class='formula-line'><span>+ %15 Model Güveni</span><span>veri ve benzerlik gücü</span></div>
            <div class='formula-line'><span>- %10 Risk Cezası</span><span>manuel inceleme sinyalleri</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_openrouter_api_key() -> str:
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        secret_key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        if secret_key:
            return secret_key
    except Exception:
        return ""
    return ""


def advisor_context(result: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    similar = result["similar"].head(5)
    tender = current_tender() or {}
    context = {
        **best,
        "tender_id": tender.get("tender_id"),
        "product_name": tender.get("product_name"),
        "product_group": tender.get("product_group"),
        "region": tender.get("region"),
        "quantity": tender.get("quantity"),
        "delivery_months": tender.get("delivery_months"),
        "estimated_unit_cost": tender.get("estimated_unit_cost"),
        "corridor": result["corridor"],
        "similar_tender_count": len(result["similar"]),
        "similar_tenders": similar[
            ["tender_id", "product_group", "product_name", "region", "overall_similarity_score"]
        ].to_dict(orient="records"),
        "cluster_name": "Kazanılmış profil kümesi",
        "revealed": bool(st.session_state.get("revealed", False)),
        "method_limit": TURKISH_WARNING,
    }
    if st.session_state.get("revealed", False):
        row = selected_test_tender()
        if row is not None:
            context["revealed_actual"] = {
                "actual_won_unit_price": float(row[CANONICAL_PRICE_COLUMN]),
                "actual_margin_pct": float(row[CANONICAL_MARGIN_COLUMN]),
            }
    return context


def fallback_chat_answer(question: str, context: dict[str, Any], advisor: dict[str, Any]) -> str:
    q = question.casefold()
    corridor = context.get("corridor", {})
    risk_flags = context.get("risk_flags", [])
    risk_text = ", ".join(risk_flags) if risk_flags else "kritik risk bayrağı yok"
    base_warning = (
        "Not: Bu yorum gerçek kazanma olasılığı değildir; yalnızca geçmiş kazanılmış ihale profiline uyumu açıklar."
    )
    if "fiyat" in q or "koridor" in q:
        answer = (
            f"Fiyat koridoru benzer kazanılmış ihalelerden gelir. Bu analizde düşük fiyat "
            f"{format_try(corridor.get('predicted_low_price'))}, orta fiyat {format_try(corridor.get('predicted_mid_price'))}, "
            f"yüksek fiyat {format_try(corridor.get('predicted_high_price'))}. Koridor, agresif/dengeli/muhafazakar "
            "senaryoları karşılaştırmak için kullanılır."
        )
    elif "risk" in q or "manuel" in q:
        answer = (
            f"Manuel inceleme kararı güven skoru, profil uyumu ve risk bayraklarına göre verilir. "
            f"Bu senaryoda risk notları: {risk_text}. "
            f"Manuel inceleme: {'gerekli' if advisor['manual_review_required'] else 'gerekli görünmüyor'}."
        )
    elif "benzer" in q or "profile" in q or "profil" in q or "küme" in q:
        answer = (
            f"Bu ihale {context.get('similar_tender_count', 0)} benzer kazanılmış ihale üzerinden yorumlandı. "
            f"Kazanılmış Profil Uyum Skoru {context.get('won_profile_fit_score', 0):.1f}/100. "
            "Skor ürün, kurum/bölge, miktar, fiyat bandı, marj ve güven sinyallerinin birlikte okunmasıyla oluşur."
        )
    elif "neden" in q or "öner" in q or "senaryo" in q:
        answer = (
            f"Seçilen senaryo {format_try(context.get('proposed_unit_price'))} birim fiyatla "
            f"{format_pct(context.get('computed_margin_pct'))} beklenen marj üretir. "
            f"Senaryo skoru {context.get('scenario_score', 0):.1f}/100; profil uyumu, fiyat bandı uyumu, marj, "
            "model güveni ve risk cezası birlikte hesaplanır."
        )
    else:
        answer = (
            f"{advisor['summary']} {advisor['profile_fit_explanation']} "
            f"{advisor['price_corridor_explanation']} {advisor['risk_explanation']}"
        )
    return f"{answer}\n\n{base_warning}"


def normalize_llm_payload(content: str) -> dict[str, Any] | None:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def call_guarded_llm(context: dict[str, Any], question: str) -> dict[str, Any] | None:
    api_key = get_openrouter_api_key()
    if not api_key:
        return None
    prompt_context = {**context, "user_question": question}
    prompt = build_advisor_prompt(prompt_context)
    body = {
        "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Türkçe yanıt veren, yalnızca verilen yapılandırılmış ihale karar destek çıktısını "
                    "yorumlayan güvenli bir analistsin. Veri uydurma, kesin sonuç iddiası verme, gizli "
                    "gerçek sonucu kullanıcı açmadıysa kullanma."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return None
    parsed = normalize_llm_payload(content)
    if not parsed:
        return None
    validation = validate_advisor_output(parsed)
    forbidden = detect_forbidden_claims(" ".join(str(value) for value in parsed.values()))
    if not validation["valid"] or forbidden["forbidden_claims_detected"]:
        return None
    return parsed


def advisor_payload_to_chat_text(payload: dict[str, Any]) -> str:
    parts = [
        ("Özet", payload.get("summary")),
        ("Profil uyumu", payload.get("profile_fit_explanation")),
        ("Fiyat koridoru", payload.get("price_corridor_explanation")),
        ("Marj", payload.get("margin_explanation")),
        ("Risk", payload.get("risk_explanation")),
        ("Güven", payload.get("confidence_explanation")),
        ("Benzer ihaleler", payload.get("similar_tenders_summary")),
    ]
    return "\n\n".join(f"**{title}:** {text}" for title, text in parts if text)


def selected_test_tender() -> pd.Series | None:
    tender_id = st.session_state.get("selected_tender_id")
    if not tender_id:
        return None
    test = get_split()["test"]
    matches = test[test["tender_id"].astype(str) == str(tender_id)]
    return matches.iloc[0] if not matches.empty else None


def current_tender() -> dict[str, Any] | None:
    return st.session_state.get("adjusted_tender") or st.session_state.get("masked_tender")


def ensure_scenario_result() -> dict[str, Any] | None:
    tender = current_tender()
    if not tender:
        return None
    if st.session_state.get("scenario_result") and st.session_state.get("scenario_tender_id") == tender.get("tender_id"):
        return st.session_state.scenario_result
    with st.spinner("Senaryolar hazırlanıyor..."):
        result = rank_scenarios(get_history_frame(), tender)
    st.session_state.scenario_result = result
    st.session_state.scenario_tender_id = tender.get("tender_id")
    st.session_state.best_scenario = result["scenarios"].iloc[0].to_dict()
    return result


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-mark">TI</div>
            <div class="sidebar-title">Tender IQ</div>
            <div class="sidebar-subtitle">Agentic Bid Advisor</div>
            <div class="sidebar-status-stack">
                <span class="status-badge status-success">Demo Modu</span>
                <span class="status-badge status-warning">Sadece kazanılmış veri</span>
                <span class="status-badge status-danger">Gerçek P(win) değildir</span>
            </div>
            <div class="sidebar-note">
                Fiyat koridoru, profil uyumu, senaryo skoru ve güvenli AI yorumu için karar destek kokpiti.
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Sayfa", PAGE_NAMES, label_visibility="collapsed")
    return page


def render_home() -> None:
    data = load_active_data()
    start = pd.to_datetime(data["tender_date"]).min().date()
    end = pd.to_datetime(data["tender_date"]).max().date()
    st.markdown(
        """
        <div class='hero-card'>
            <div class='eyebrow'>Karar destek platformu</div>
            <h1 class='hero-title'>Tender IQ</h1>
            <div class='hero-subtitle'>
                Kazanılmış ihale verilerinden fiyat koridoru, profil uyumu ve teklif senaryo içgörüleri üreten karar destek platformu.
                Yeni ihaleyi geçmiş kazanılmış ihalelerle karşılaştırır, benzer emsalleri bulur, fiyat bandı üretir ve ihale ekibine teklif senaryolarını yorumlaması için yapılandırılmış destek sağlar.
            </div>
            <div class='hero-badges'>
                <span class='hero-badge'>Profil uyumu</span>
                <span class='hero-badge'>Fiyat koridoru</span>
                <span class='hero-badge'>Senaryo skoru</span>
                <span class='hero-badge'>Güvenli AI yorumu</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    warning_box()
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    section_header(
        "Bu ürün hangi sorulara cevap verir?",
        "Tender IQ, teklif kararını tek bir sayıya indirgemez; fiyat, benzerlik, marj ve risk sinyallerini birlikte okunabilir hale getirir.",
    )
    questions = [
        ("Bu ihaleye girmeli miyim?", "Geçmiş kazanılmış profile ne kadar benzediğini ve manuel inceleme ihtiyacını gösterir."),
        ("Nasıl fiyat vermeliyim?", "Benzer kazanılmış ihalelerden düşük, orta ve yüksek fiyat koridoru üretir."),
        ("Hangi geçmiş ihaleler buna benziyor?", "TF-IDF ve cosine similarity ile emsal kazanılmış ihaleleri listeler."),
        ("Beklenen katkı kârı nasıl görünüyor?", "Aday fiyatların marj ve katkı etkisini senaryo bazında karşılaştırır."),
        ("Bu teklif senaryosu mantıklı mı?", "Profil uyumu, fiyat bandı, marj, güven ve risk cezasını birlikte skorlar."),
        ("Riskli sinyaller var mı?", "Sıra dışı profil, düşük güven veya kısıt ihlali gibi uyarıları görünür kılar."),
    ]
    for start_idx in range(0, len(questions), 3):
        cols = st.columns(3, gap="medium")
        for offset, (title, body) in enumerate(questions[start_idx : start_idx + 3]):
            with cols[offset]:
                glass_card(title, body)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Nasıl çalışır?", "Üç adımlı akış, demo sırasında kullanıcının nereden başlayacağını netleştirir.")
    render_method_grid(
        [
            ("Tarihsel kazanılmış veriyi kullanır", "Sistem geçmişte kazanılmış ihaleleri temel veri seti olarak alır."),
            ("Benzer ihaleleri ve fiyat bandını üretir", "Yeni ihaleye benzeyen emsallerden fiyat koridoru ve profil sinyali çıkarır."),
            ("Senaryoları skorlayıp açıklar", "Aday teklifleri skorlar, riskleri gösterir ve AI Danışman ile yorumlar."),
        ],
        ["blue", "purple", "green"],
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Buradan nereye gitmeliyim?", "Ana akış sayfaları ve ne için kullanılacakları.")
    nav_items = [
        ("Veri Seti", "Sistemin kullandığı kazanılmış ihale verisini ve kalite kontrollerini inceleyin."),
        ("Metodoloji", "Skorun, benzerlik hesabının ve backtest yaklaşımının nasıl kurulduğunu görün."),
        ("Test İhalesi", "Gerçek sonucu gizli bir test ihalesini canlı ihale gibi simüle edin."),
        ("Senaryo Analizi", "Teklif fiyatı, marj, katkı ve risk senaryolarını karşılaştırın."),
        ("AI Danışman", "Model çıktıları hakkında güvenli Türkçe açıklama alın."),
        ("Backtest Sonuçları", "Fiyat koridoru ve senaryo yaklaşımının geçmiş test performansını görün."),
    ]
    for start_idx in range(0, len(nav_items), 3):
        cols = st.columns(3, gap="medium")
        for offset, (title, body) in enumerate(nav_items[start_idx : start_idx + 3]):
            with cols[offset]:
                glass_card(title, body, "Yönlendirme")

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Demo Veri Özeti", "Ana sayfada yalnızca en gerekli veri sinyalleri gösterilir.")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Toplam kayıt", format_int(len(data)), "Kazanılmış ihale")
    with c2:
        metric_card("Ürün grubu", format_int(data["product_group"].nunique()), "Kategori sayısı")
    with c3:
        metric_card("Kurum sayısı", format_int(data["buyer_institution"].nunique()), "Alıcı kurum")
    with c4:
        metric_card("Veri tarih aralığı", f"{start} - {end}", "İhale tarihi")


def render_data_quality() -> None:
    page_header(
        "Veri Seti ve Kalite Kontrol",
        "Bu sayfa, sistemin kullandığı tarihsel kazanılmış ihale verisini ve analiz güvenliğini gösterir.",
        "Veri",
    )
    data = load_active_data()
    schema_result = validate_schema(data)
    quality = validate_data_quality(data)
    summary = schema_quality_summary(data)
    start = pd.to_datetime(data["tender_date"]).min().date()
    end = pd.to_datetime(data["tender_date"]).max().date()

    info_callout(
        "Bu adım, sistemin fiyat koridoru, benzer ihale eşleştirmesi, profil uyumu ve senaryo skorlaması için kullandığı tarihsel kazanılmış ihale veri setini yükler ve doğrular. Kalite kontrolleri, verinin analiz için uygun ve güvenli olup olmadığını gösterir.",
        "Veri neden önemli?",
    )
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    section_header("Veri ne işe yarıyor?", "Veri seti, Tender IQ'nun tüm karar destek çıktılarının temel girdisidir.")
    data_use_cards = [
        ("Benzer ihale bulma", "Yeni ihale, geçmiş kazanılmış ihalelerle karşılaştırılır ve en yakın emsaller bulunur."),
        ("Fiyat koridoru üretme", "Benzer kazanılmış ihalelerden düşük, orta ve yüksek fiyat bandı çıkarılır."),
        ("Profil uyumu hesaplama", "Yeni ihalenin geçmiş kazanılmış işlere ne kadar tanıdık göründüğü ölçülür."),
        ("Senaryo ve marj analizi", "Aday teklif fiyatlarının marj, katkı ve risk etkisi karşılaştırılır."),
    ]
    cols = st.columns(4, gap="medium")
    for idx, (title, body) in enumerate(data_use_cards):
        with cols[idx]:
            glass_card(title, body)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Veri özeti", "Demo veri setinin iş seviyesindeki kısa görünümü.")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Kayıt sayısı", format_int(summary["row_count"]), "Normalize edilmiş ihale")
    with c2:
        metric_card("Ürün grubu sayısı", format_int(data["product_group"].nunique()), "Kategori")
    with c3:
        metric_card("Kurum sayısı", format_int(data["buyer_institution"].nunique()), "Alıcı kurum")
    with c4:
        metric_card("Tarih aralığı", f"{start} - {end}", "İhale tarihi")

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Kalite kontrol sonuçları", "Analize başlamadan önce veri setinin kullanılabilirliği doğrulanır.")
    status = "good" if schema_result.valid and quality["passed"] else "warn"
    quality_cards = [
        ("Şema kontrolü", "Geçti" if schema_result.valid else "Eksik", "Zorunlu kolonların bulunup bulunmadığını kontrol eder.", "good" if schema_result.valid else "bad"),
        ("Zorunlu kolonlar", "Tamam" if not schema_result.missing_columns else "Eksik", "Ürün, kurum, miktar, tarih, fiyat ve marj alanlarının durumunu gösterir.", "good" if not schema_result.missing_columns else "bad"),
        ("Eksik veri durumu", "Uygun" if quality["passed"] else "Uyarı", "Boş veya sorunlu değerlerin analizi bozup bozmadığını inceler.", "good" if quality["passed"] else "warn"),
        ("Tekrarlı kayıt kontrolü", format_int(summary["duplicate_tender_ids"]), "Aynı tender_id ile gelen tekrarları görünür kılar.", "good" if summary["duplicate_tender_ids"] == 0 else "warn"),
        ("Veri kullanıma hazır mı?", "Hazır" if status == "good" else "Kontrol gerekli", "Kalite ve şema kontrollerinin ortak sonucudur.", status),
    ]
    for start_idx in range(0, len(quality_cards), 3):
        cols = st.columns(3, gap="medium")
        for offset, (title, value, body, card_status) in enumerate(quality_cards[start_idx : start_idx + 3]):
            with cols[offset]:
                glass_card(title, body, "", badge(value, card_status))

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Zorunlu kolonlar", "Teknik kolon adları iş dilindeki anlamlarıyla birlikte gösterilir.")
    required_columns = pd.DataFrame(
        [
            ["product_name", "Ürün adı", "İhale konusu ürün veya hizmet adı."],
            ["product_group", "Ürün kategorisi", "Benzerlik ve segment kırılımı için ana kategori."],
            ["buyer_institution", "Alıcı kurum", "Kurum tipi ve tekrar eden alıcı davranışı için kullanılır."],
            ["region", "Bölge", "Bölgesel benzerlik ve fiyat davranışı için kullanılır."],
            ["quantity", "İhale miktarı", "Ölçek ve miktar bandı eşleşmesi için kullanılır."],
            ["tender_date", "İhale tarihi", "Temporal split ve tarihsel test disiplini için gerekir."],
            ["estimated_unit_cost", "Tahmini maliyet", "Marj ve katkı senaryolarını hesaplamak için kullanılır."],
            [CANONICAL_PRICE_COLUMN, "Kazanılmış birim fiyat", "Fiyat koridorunu eğitmek ve backtestte kıyaslamak için kullanılır."],
            [CANONICAL_MARGIN_COLUMN, "Kazanılmış marj", "Geçmiş marj davranışını ve senaryo sağlığını değerlendirmek için kullanılır."],
        ],
        columns=["Kolon", "İş anlamı", "Sistemdeki rolü"],
    )
    st.dataframe(required_columns, hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Eksik veri kontrolü", "Kolon eşleştirme", "Veri önizleme", "İsteğe bağlı yeni veri yükleme"])
    with tabs[0]:
        null_df = pd.DataFrame(
            [{"Kolon": key, "Boş oran": value} for key, value in summary["null_rates"].items()]
        ).sort_values("Boş oran", ascending=False)
        st.dataframe(null_df, hide_index=True, width="stretch", column_config={"Boş oran": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=1)})
        if quality["issues"]:
            st.warning("\n".join(f"- {issue}" for issue in quality["issues"]))
        if schema_result.warnings:
            st.warning("\n".join(schema_result.warnings))
    with tabs[1]:
        mapping = pd.DataFrame(
            [
                ["Kazanılmış birim fiyat", CANONICAL_PRICE_COLUMN, "Bulundu" if CANONICAL_PRICE_COLUMN in data else "Eksik"],
                ["Kazanılmış marj", CANONICAL_MARGIN_COLUMN, "Bulundu" if CANONICAL_MARGIN_COLUMN in data else "Eksik"],
                ["Tahmini maliyet", "estimated_unit_cost", "Bulundu" if "estimated_unit_cost" in data else "Eksik"],
                ["İhale tarihi", "tender_date", "Bulundu" if "tender_date" in data else "Eksik"],
            ],
            columns=["İş anlamı", "Kullanılan kolon", "Durum"],
        )
        st.dataframe(mapping, hide_index=True, width="stretch")
    with tabs[2]:
        st.dataframe(data.head(25), hide_index=True, width="stretch")
    with tabs[3]:
        info_callout(
            "Yeni CSV yüklemek, demo veri seti yerine kurumunuza ait tarihsel kazanılmış ihale verisini kullanmak içindir. Dosyada ürün, kurum, bölge, miktar, tarih, tahmini maliyet, kazanılmış fiyat ve marj alanları bulunmalıdır.",
            "Ne zaman yüklenir?",
        )
        uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"])
        if uploaded is not None:
            st.session_state.active_data = normalize_schema(pd.read_csv(uploaded))
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("scenario_result", None)
            st.success("Veri yüklendi ve uygulama şemasına uyarlandı.")


def render_methodology() -> None:
    page_header(
        "Metodoloji",
        "Sistem neyi, neden ve nasıl hesaplıyor? Aşağıdaki bölüm metodolojiyi temelden ama iş dilinde açıklar.",
        "Metodoloji",
    )
    warning_box()
    info_callout(PWIN_PROXY_EXPLANATION, "P-Win yerine ne var?")
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    render_method_grid(
        [
            (
                "Veri kısıtı",
                "Elimizde yalnızca kazanılmış ihaleler var. Kaybedilmiş veya teklif verilip kazanılamamış kayıt yok.",
            ),
            (
                "Doğru hedef",
                "Sistem sonuç tahmini yapmaz; yeni ihalenin geçmiş kazanılmış profillere ne kadar uyduğunu ölçer.",
            ),
            (
                "Test disiplini",
                "Test ihalesi canlı ihale gibi ele alınır. Gerçek fiyat ve marj sonuç açma adımına kadar gizlenir.",
            ),
        ],
        ["amber", "green", "blue"],
    )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    tabs = st.tabs(
        [
            "Temel Mantık",
            "Benzerlik Hesabı",
            "Model Bileşenleri",
            "Skor ve Senaryo",
            "Backtest ve Metrikler",
            "AI Danışman",
        ]
    )

    with tabs[0]:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            glass_card(
                "Veri Kısıtı",
                "Elimizde sadece geçmişte kazanılmış ihale kayıtları var. Kaybedilmiş veya teklif verilip kazanılamamış kayıt olmadığı için klasik kazan/kaybet sınıflandırması kurulmaz.",
                "01",
            )
        with c2:
            glass_card(
                "Doğru Okuma",
                "Sistem gerçek olasılık üretmez. Yeni ihalenin geçmişte kazanılmış işlere ne kadar benzediğini ve fiyat/marj/risk dengesinin geçmiş profile uyup uymadığını gösterir.",
                "02",
            )
        with c3:
            glass_card(
                "Kazanılmış Profil Uyum Skoru",
                "Ürün, kurum, bölge, miktar, teslim süresi, fiyat bandı, beklenen marj, risk ve model güveni birlikte okunur. Düşük skor çoğu zaman manuel inceleme sinyalidir.",
                "03",
            )
        with st.expander("Neden accuracy, precision, recall veya ROC-AUC ana başarı metriği değil?", expanded=False):
            st.markdown(
                "Negatif sınıf güvenilir olmadığı için kazan/kaybet sınıflandırması ölçülmez. Backtest, fiyat koridoru hizası, profil uyumu, segment performansı, sızıntı kontrolü ve danışman güvenliği üzerinden yapılır."
            )

    with tabs[1]:
        section_header("Benzerlik nasıl hesaplanıyor?", "TF-IDF ve cosine similarity yeni ihaleyi geçmiş kazanılmış ihalelerle karşılaştırır.", "Retrieval")
        render_method_grid(
            [
                ("İhale metni hazırlanır", "Ürün adı, ürün grubu, kurum, bölge, ihale tipi ve miktar bilgileri tek bir ihale profiline dönüştürülür."),
                ("TF-IDF ile vektöre çevrilir", "Ayırt edici kelimeler daha güçlü temsil edilir; genel kelimelerin etkisi azaltılır."),
                ("Cosine similarity hesaplanır", "Yeni ihale vektörü geçmiş ihale vektörleriyle karşılaştırılır."),
                ("Top-K benzer ihaleler seçilir", "En yüksek skorlu kazanılmış ihaleler emsal listeye alınır."),
                ("Koridor ve skorlar beslenir", "Fiyat koridoru, profil uyumu ve danışman açıklamaları bu emsal setten destek alır."),
            ],
            ["blue", "purple", "cyan", "green", "amber"],
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            glass_card(
                "TF-IDF",
                "Metin içindeki kelimeleri sayısal hale getirir. Sık geçen ama ayırt edici olmayan kelimelerin etkisini azaltır; ürün, kurum veya ihale tipi gibi ayırt edici kelimelerin etkisini artırır.",
                "Metin temsili",
            )
        with c2:
            glass_card(
                "Cosine similarity",
                "İki ihalenin sayısal vektörlerinin birbirine ne kadar yakın olduğunu ölçer. Skor 1'e yaklaştıkça benzerlik artar; 0'a yaklaştıkça azalır.",
                "Yakınlık hesabı",
            )
        with st.expander("Top-K retrieval ve eşleşme metrikleri", expanded=False):
            st.markdown(
                """
                Örnek profil: “serum ürün grubu + kamu hastanesi + Marmara bölgesi + açık ihale + 50.000 adet”.
                Aynı ürün grubu, benzer kurum tipi, aynı bölge ve yakın miktar varsa benzerlik güçlenir.
                """
            )
            metrics = pd.DataFrame(
                [
                    ["Ürün Grubu Eşleşme Oranı", "İlk K benzer ihale içinde aynı ürün grubuna düşen kayıt oranı."],
                    ["Bölge Eşleşme Oranı", "Benzer ihalelerin seçili ihale ile aynı bölgede olma oranı."],
                    ["Miktar Bandı Eşleşme Oranı", "Benzer ihalelerin yakın miktar ölçeğinde olma oranı."],
                    ["İlk K Ortalama Benzerlik", "Getirilen benzer ihalelerin ortalama cosine/özellik benzerliği."],
                ],
                columns=["Metrik", "Ne anlatır?"],
            )
            st.dataframe(metrics, hide_index=True, width="stretch")

    with tabs[2]:
        section_header("Model Bileşenleri", "Her bileşen karar destek çıktısının farklı bir parçasını açıklar.", "Model")
        render_model_grid(
            [
                ("01", "TF-IDF + Cosine Similarity", "Ne yapar: Yeni ihaleye benzeyen kazanılmış ihaleleri bulur. Neden var: Emsal seti olmadan fiyat ve profil yorumu zayıf kalır. Katkı: Benzer ihaleler listesini ve koridor girdisini üretir.", "blue"),
                ("02", "K-Means", "Ne yapar: Kazanılmış ihaleleri benzer başarı profillerine ayırır. Neden var: Tek tek ihale yerine profil segmenti görmeyi sağlar. Katkı: Cluster ve profil yorumunu destekler.", "purple"),
                ("03", "Isolation Forest", "Ne yapar: Yeni ihalenin geçmiş profile normal mi sıra dışı mı uyduğunu kontrol eder. Neden var: Aykırı durumları saklamaz. Katkı: Risk ve manuel inceleme sinyali üretir.", "amber"),
                ("04", "Price Corridor Engine", "Ne yapar: Emsal kazanılmış ihalelerden düşük, orta ve yüksek fiyat bandı çıkarır. Neden var: Tek nokta fiyat yerine karar aralığı verir. Katkı: Senaryo fiyatlarını besler.", "green"),
                ("05", "Scenario Scoring", "Ne yapar: Fiyat, marj, profil uyumu, güven ve risk cezasını tek karar destek skorunda birleştirir. Neden var: Alternatif teklifleri kıyaslanabilir hale getirir. Katkı: Sıralı senaryo önerisi üretir.", "cyan"),
                ("06", "Model Confidence / Risk", "Ne yapar: Benzer ihale sayısı, veri kalitesi, band genişliği ve aykırılık sinyallerini birlikte okur. Neden var: Skorun ne kadar güvenle okunacağını gösterir. Katkı: AI Danışman ve manuel inceleme kararını destekler.", "blue"),
                ("07", "Linear Regression Baz Modeli", "Ne yapar: Ürün grubu, bölge, ihale tipi, miktar, teslim süresi ve tahmini rakip sayısı gibi alanlardan beklenen fiyat için doğrusal referans üretir. Neden var: Emsal tabanlı fiyat koridorunu basit ve açıklanabilir bir fiyat tahminiyle kıyaslamak için kullanılır. Katkı: Backtestte koridor yaklaşımının basit doğrusal modele göre ne kadar tutarlı olduğunu gösterir.", "green"),
                ("08", "XGBoost / Ağaç Tabanlı Kontrol", "Ne yapar: Doğrusal olmayan fiyat ilişkilerini yakalayabilen ağaç tabanlı referans model ailesini temsil eder. Neden var: Miktar, bölge ve ürün grubunun fiyat üzerindeki doğrusal olmayan etkilerini kontrol etmek için kullanılır. Katkı: Nihai P-Win üretmez; fiyat koridorunun regresyon bazlı tahminlerle tutarlılığını değerlendirmek için metodolojik karşılaştırma sağlar.", "amber"),
            ]
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        with st.expander("Sıra dışı durum örnekleri", expanded=False):
            st.markdown("- çok yüksek veya çok düşük miktar\n- alışılmadık ürün-kurum kombinasyonu\n- çok kısa teslim süresi\n- çok yüksek veya çok düşük fiyat\n- düşük benzer ihale sayısı")
        with st.expander("Fiyat koridoru nasıl oluşuyor?", expanded=True):
            info_callout(
                "Sistem seçili ihaleye en çok benzeyen kazanılmış ihaleleri bulur. Ana fiyat koridoru bu emsal ihalelerdeki normalize fiyatların alt, orta ve üst yüzdeliklerinden üretilir. Lineer regresyon ve XGBoost/ağaç tabanlı modeller ise bu koridorun basit ve daha esnek fiyat tahminleriyle kıyaslanması için metodolojik kontrol katmanı olarak anlatılır; gerçek kazanma olasılığı üretmez. Koridor çok genişse tek başına güçlü kanıt sayılmaz, bu yüzden backtestte band genişliği de ölçülür.",
                "Fiyat Koridoru",
            )
            render_method_grid(
                [
                    ("Alt Bant", "Daha agresif fiyat seviyesi."),
                    ("Orta Bant", "Dengeli fiyat seviyesi."),
                    ("Üst Bant", "Daha muhafazakar fiyat seviyesi."),
                ],
                ["blue", "green", "amber"],
            )

    with tabs[3]:
        section_header("Senaryo Skoru", "Tek bir model çıktısı değildir; beş bileşen birlikte değerlendirilir.", "Skor")
        render_formula_card()
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        render_model_grid(
            [
                ("🧭", "Profil Uyumu", "Yeni ihalenin geçmiş kazanılmış ihalelere benzerliği.", "blue"),
                ("🎯", "Fiyat Bandı Uyumu", "Önerilen fiyatın geçmiş fiyat koridoruna yakınlığı.", "green"),
                ("💹", "Marj Skoru", "Beklenen karlılık ve katkı karı sağlığı.", "purple"),
                ("🛡️", "Model Güveni", "Yeterli benzer ihale ve veri kalitesi olup olmadığı.", "cyan"),
                ("⚠️", "Risk Cezası", "Sıra dışı durumlar, düşük benzerlik, düşük güven ve kısıt ihlalleri.", "amber"),
            ]
        )
        warning_box()

    with tabs[4]:
        section_header(
            "Backtest ve Metrikler",
            "Test yılındaki geçmiş kazanılmış ihaleler, sonuç fiyatı ve gerçek marjı gizlenerek sisteme o gün yeni gelmiş canlı ihale gibi verilir; sistem önce emsal, profil, koridor ve senaryo çıktısı üretir, sonra gerçek sonuç açılarak bu çıktılarla karşılaştırılır.",
            "Ölçüm",
        )
        info_callout(
            "Backtest’in amacı kazanma/kaybetme tahmini doğruluğunu ölçmek değildir. Amaç, sistemin geçmişte kazanılmış ama sonucu gizlenmiş ihaleleri geçmiş başarı profillerine doğru yerleştirip yerleştiremediğini ve fiyat/senaryo önerilerinin tarihsel sonuçlarla ne kadar tutarlı olduğunu ölçmektir.",
            "Backtest amacı:",
        )
        render_method_grid(
            [
                ("Zaman bazlı ayrım", "Model geçmişi bilir, geleceği bilmez. Random split canlı karar anını yeterince temsil etmez."),
                ("Sonuç gizleme", "Kazanılmış fiyat ve marj reveal adımına kadar retrieval, scorer, optimizer ve advisor katmanlarından maskelenir."),
                ("Karşılaştırma", "Fiyat koridoru, senaryo sıralaması, segment metrikleri ve sızıntı kontrolü birlikte raporlanır."),
            ],
            ["blue", "amber", "green"],
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        with st.expander("Başarıyı hangi metriklerle ölçüyoruz?", expanded=True):
            metrics_table = pd.DataFrame(
                [
                    ["Fiyat Koridoru Kapsama Oranı", "Gerçek kazanılmış fiyat sistemin önerdiği fiyat bandının içinde mi?"],
                    ["MAE", "Tahmin edilen orta fiyat ile gerçek fiyat arasındaki ortalama mutlak fark."],
                    ["MAPE / SMAPE / WAPE", "Hatanın yüzde olarak farklı okuma biçimleri."],
                    ["Ortalama Koridor Genişliği", "Bandın ne kadar geniş olduğunu gösterir."],
                    ["Coverage Adjusted Band Score", "Kapsama oranını band genişliğiyle birlikte değerlendirir."],
                    ["Gerçek Kazanılmış Senaryo Sıralaması", "Tarihsel gerçek konfigürasyon aday senaryolar arasında ne kadar üstte kaldı?"],
                    ["Kazanılmış test ihalelerini geçmiş profile uygun tanıma oranı", "Profil modelinin kazanılmış test kayıtlarını inlier görme oranı."],
                    ["Synthetic outlier detection rate", "Sentetik sıra dışı örneklerin model tarafından riskli veya aykırı görülme oranı."],
                    ["Yasak iddia üretme oranı", "AI Danışman garanti, kesin sonuç veya gerçek P(win) iddiası üretiyor mu? Hedef sıfırdır."],
                ],
                columns=["Metrik", "Ne anlatır?"],
            )
            st.dataframe(metrics_table, hide_index=True, width="stretch")
        with st.expander("Sızıntı kontrolü nedir?", expanded=False):
            st.markdown(
                """
                Sızıntı, test sırasında modelin normalde bilmemesi gereken gerçek sonucu önceden görmesidir.
                Test ihalesinin kazanılmış fiyatı, final tutarı veya gerçek marjı modele verilirse test güvenilir olmaz.
                """
            )
            blocked = pd.DataFrame(
                [
                    ["won_unit_price"],
                    ["won_total_amount"],
                    ["actual_margin_pct"],
                    ["actual_unit_margin"],
                    ["final_contract_amount"],
                    ["actual_award_result"],
                    ["revealed_actual_result"],
                ],
                columns=["Sonuç açılmadan önce engellenen alan örnekleri"],
            )
            st.dataframe(blocked, hide_index=True, width="stretch")

    with tabs[5]:
        section_header("AI Danışman", "Skor hesaplamaz; mevcut model çıktılarını güvenli Türkçe açıklamaya dönüştürür.", "LLM Guardrails")
        render_model_grid(
            [
                ("💬", "Açıklama Üretir", "Profil uyumu, fiyat koridoru, marj, risk ve benzer ihaleleri anlaşılır hale getirir.", "blue"),
                ("🧱", "Guardrail Uygular", "Garanti, kesin sonuç veya gerçek P(win) iddiası üretirse çıktı reddedilir.", "amber"),
                ("🔒", "Reveal Kuralına Uyar", "Gerçek sonuç açılmadıysa kazanılmış fiyat veya gerçek marjı kullanamaz.", "purple"),
                ("🧰", "Fallback Çalışır", "LLM sağlayıcısı yoksa deterministik danışman aynı sohbet akışında yanıt verir.", "green"),
            ]
        )
def render_test_simulator() -> None:
    page_header(
        "Test İhalesi Simülatörü",
        "Geçmişte kazanılmış bir ihale, gerçek sonucu gizlenmiş şekilde sisteme yeni gelen canlı ihale gibi verilir. Sistem; emsal benzerlik, başarı profili, sıra dışılık, fiyat bandı, risk/güven ve senaryo uygunluğunu birlikte analiz eder. Gerçek sonuç daha sonra açılarak bu çıktılarla karşılaştırılır.",
        "Simülasyon",
    )
    warning_box()
    info_callout(PWIN_PROXY_EXPLANATION, "P-Win yerine ne var?")
    info_callout(
        "Bu ekran sadece fiyat koridorunu test etmez. Seçilen ihale; emsal benzerlik, K-Means profil ataması, Isolation Forest uygunluğu, profil uyum skoru, fiyat bandı, risk/güven ve senaryo skoru açısından birlikte analiz edilir.",
        "Bu ekranın kapsamı:",
    )
    split = get_split()
    test = split["test"]
    selected = st.selectbox("Test ihalesi seç", test["tender_id"].astype(str).tolist())
    st.session_state.selected_tender_id = selected
    st.session_state.revealed = False if st.session_state.get("last_selected_tender") != selected else st.session_state.get("revealed", False)
    st.session_state.last_selected_tender = selected

    row = test[test["tender_id"].astype(str) == selected].iloc[0]
    masked = mask_actual_result_fields(row.to_dict())
    audit = audit_pre_reveal_input(selected, masked)
    st.session_state.masked_tender = masked
    st.session_state.leakage_audit = audit

    section_header("Bu ekran neyi test eder?", "Simülasyon fiyat bandından ibaret değildir; geçmiş başarı profiline yerleştirme kalitesini de gösterir.")
    test_cards = [
        ("Emsal Bulma", "Sistem, yeni ihale için geçmişte kazanılmış en benzer ihaleleri bulur."),
        ("Profil Atama", "K-Means ile ihale geçmiş kazanılmış başarı kümelerinden birine yerleştirilir."),
        ("Sıra Dışılık Kontrolü", "Isolation Forest, ihalenin geçmiş kazanım profiline normal mi yoksa sıra dışı mı uyduğunu kontrol eder."),
        ("Profil Uyum Skoru", "Sistem, ihalenin geçmiş kazanılmış ihale profiline ne kadar uyduğunu 0-100 arası skorlar."),
        ("Fiyat ve Senaryo Analizi", "Benzer ihalelerden fiyat koridoru ve teklif senaryo skorları üretilir."),
    ]
    for start_idx in range(0, len(test_cards), 3):
        cols = st.columns(3, gap="medium")
        for offset, (title, body) in enumerate(test_cards[start_idx : start_idx + 3]):
            with cols[offset]:
                render_small_card(title, body)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    selected_body = (
        f"İhale ID: {selected} | Ürün grubu: {masked.get('product_group', '-')} | "
        f"Bölge: {masked.get('region', '-')} | Kurum: {masked.get('buyer_institution', '-')}"
    )
    glass_card("Seçili İhale", selected_body, "Canlı simülasyon", badge("Gerçek sonuç gizli", "warning"))
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    section_header("Canlı İhale Girdileri", "Bu alanlar simülasyon için düzenlenebilir; gerçek kazanılmış fiyat ve marj maskelidir.", "Kontrol paneli")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        masked["quantity"] = int(c1.number_input("Miktar", min_value=1, value=int(masked.get("quantity", 1))))
        masked["delivery_months"] = int(c2.number_input("Teslim Süresi (Ay)", min_value=1, value=int(masked.get("delivery_months", 6))))
        masked["competitor_count_estimate"] = int(c3.number_input("Tahmini Rakip Sayısı", min_value=0, value=int(masked.get("competitor_count_estimate", 3))))
        masked["estimated_unit_cost"] = float(c4.number_input("Tahmini Birim Maliyet", min_value=0.01, value=float(masked.get("estimated_unit_cost", 1.0))))
    st.session_state.adjusted_tender = masked

    if st.button("Simülasyonu çalıştır", type="primary"):
        st.session_state.pop("scenario_result", None)
        result = ensure_scenario_result()
        write_audit_event({"event_type": "test_tender_simulation", "tender_id": selected, "leakage_audit": audit})
        if result:
            st.success("Simülasyon tamamlandı. Senaryo Analizi sayfasında detayları görebilirsiniz.")

    result = ensure_scenario_result()
    best = result["scenarios"].iloc[0].to_dict() if result else {}
    similar = result.get("similar", pd.DataFrame()) if result else pd.DataFrame()
    quality = retrieval_quality_from_result(result, masked) if result else {}
    top10_avg_similarity = float(result.get("top10_avg_similarity", 0.0)) if result else 0.0
    profile_label = "Geçmiş profile uygun" if bool(best.get("is_inlier", best.get("won_profile_fit_score", 0) >= 55)) else "Sıra dışı / manuel inceleme önerilir"
    profile_status = "good" if profile_label == "Geçmiş profile uygun" else "warn"
    risk_fit_score = max(0.0, 100 - float(best.get("risk_penalty_score", 0)))
    cluster_display = best.get("cluster_name") or "Hesaplanamadı"

    section_header("Ana Analiz Skorları", "Kazanılmış Profil Uyum Skoru ana karar destek göstergesidir; gerçek kazanma olasılığı değildir.")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Kazanılmış Profil Uyum Skoru", format_score(best.get("won_profile_fit_score")), "Yeni ihalenin geçmişte kazanılmış ihale profillerine yakınlığı")
    with c2:
        metric_card("Emsal Benzerlik Gücü", f"{top10_avg_similarity:.2f}", "Bulunan benzer ihalelerin yeni ihaleyle ortalama benzerliği")
    with c3:
        metric_card("K-Means Profil Kümesi", str(best.get("cluster_id", "Hesaplanamadı")), cluster_display)
    with c4:
        metric_card("Isolation Forest Durumu", profile_label, "İhale geçmiş kazanılmış profile normal mi, sıra dışı mı?")

    c5, c6, c7 = st.columns(3, gap="medium")
    with c5:
        metric_card("Fiyat Bandı Uyumu", format_score(best.get("price_band_fit_score")), "Önerilen fiyatın geçmiş kazanılmış fiyat koridoruna uyumu")
    with c6:
        metric_card("Model Güveni", format_score(best.get("model_confidence_score")), "Veri kalitesi, benzer ihale sayısı ve skor tutarlılığına göre güven seviyesi")
    with c7:
        metric_card("Risk Uygunluk Skoru", format_score(risk_fit_score), "Yüksek risk uygunluğu, belirgin risk sinyalinin düşük olduğunu gösterir.")

    left, right = st.columns([1, 1], gap="medium")
    with left:
        section_header("Maskelenmiş Girdi", "Sonuç açılmadan önce görünen alanlar.", "Girdi")
        safe_preview = pd.DataFrame([masked]).T.reset_index()
        safe_preview.columns = ["Alan", "Değer"]
        safe_preview["Değer"] = safe_preview["Değer"].astype(str)
        st.dataframe(safe_preview, hide_index=True, width="stretch")
    with right:
        cluster_text = f"Cluster: {best.get('cluster_id', 'Kazanılmış profil kümesi')}"
        isolation_text = "Isolation Forest: " + profile_label
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='section-title'>Sızıntı ve Profil Durumu</div>
                <div style='display:flex; flex-wrap:wrap; gap:.5rem; margin:.7rem 0'>
                    {badge("Sızıntı yok" if audit["audit_status"] == "pass" else "Sızıntı var", "good" if audit["audit_status"] == "pass" else "bad")}
                    {badge(profile_label, profile_status)}
                    {badge(cluster_text, "success")}
                    {badge(isolation_text, profile_status)}
                </div>
                <div class='metric-note'>Gerçek kazanılmış fiyat ve marj bu ekranda gösterilmez.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Profil ve Sıra Dışılık Analizi", "K-Means ve Isolation Forest sonuçları simülasyonun ana parçasıdır.")
    left, right = st.columns(2, gap="medium")
    with left:
        render_kv_card(
            "K-Means Profil Ataması",
            [
                ("Cluster adı", str(cluster_display)),
                ("Cluster ID", str(best.get("cluster_id", "Hesaplanamadı"))),
                ("Geçmiş ihale sayısı", format_int(best.get("cluster_count"))),
                ("Baskın ürün grubu", str(best.get("cluster_dominant_product_group", "Hesaplanamadı"))),
                ("Ortalama/medyan fiyat", format_try(best.get("cluster_median_price"))),
                ("Medyan marj", format_pct(best.get("cluster_median_margin"))),
            ],
            "Bu ihale, geçmişte kazanılmış benzer başarı profillerinden biriyle ilişkilendiriliyorsa skor daha okunabilir hale gelir.",
        )
    with right:
        render_kv_card(
            "Isolation Forest Kontrolü",
            [
                ("Durum", profile_label),
                ("Inlier skoru", format_score(best.get("inlier_score"))),
                ("Manuel inceleme", "Önerilir" if profile_status == "warn" else "Şu an güçlü sinyal yok"),
            ],
            "Bu sonuç kazanma olasılığı değildir; yalnızca ihalenin geçmiş kazanılmış veri dağılımına aykırı görünüp görünmediğini anlatır.",
        )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Bu ihaleye benzeyen geçmiş kazanılmış ihaleler", "Bu liste, sistemin yeni ihaleyi hangi geçmiş kazanılmış ihalelerle karşılaştırdığını gösterir. Profil uyumu, fiyat koridoru ve senaryo analizi bu emsal havuzundan beslenir.")
    similar_display = similar.head(10)[
        [
            "tender_id",
            "product_group",
            "product_name",
            "buyer_institution",
            "region",
            "quantity",
            "overall_similarity_score",
            CANONICAL_PRICE_COLUMN,
        ]
    ].copy()
    similar_display.columns = [
        "İhale ID",
        "Ürün grubu",
        "Ürün adı",
        "Kurum",
        "Bölge",
        "Miktar",
        "Benzerlik skoru",
        "Geçmiş kazanılmış fiyat",
    ]
    st.dataframe(
        similar_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Benzerlik skoru": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Geçmiş kazanılmış fiyat": st.column_config.NumberColumn(format="%.2f TL"),
        },
    )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Fiyat Koridoru — Emsal İhalelerden Üretilen Fiyat Bandı", "Fiyat koridoru, benzer geçmiş kazanılmış ihalelerde oluşan fiyatlardan üretilir. Bu, sistemin çıktılarından sadece biridir. Ana analiz; emsal benzerlik, profil uyumu, K-Means segmenti, Isolation Forest uygunluğu, risk/güven ve senaryo skoru ile birlikte değerlendirilmelidir.")
    corridor = result.get("corridor", {}) if result else {}
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Alt bant", format_try(corridor.get("predicted_low_price")), "Daha agresif fiyat seviyesi")
    with c2:
        metric_card("Orta bant", format_try(corridor.get("predicted_mid_price")), "Dengeli fiyat seviyesi")
    with c3:
        metric_card("Üst bant", format_try(corridor.get("predicted_high_price")), "Daha muhafazakar fiyat seviyesi")
    with c4:
        metric_card("Senaryo skoru", format_score(best.get("scenario_score")), "En yüksek sıralı teklif senaryosu")


def scenario_name(index: int) -> str:
    names = ["Muhafazakâr Senaryo", "Dengeli Senaryo", "Agresif Senaryo"]
    return names[index] if index < len(names) else f"Alternatif Senaryo {index + 1}"


def render_scenario_analysis() -> None:
    page_header(
        "Senaryo Analizi",
        "Aday teklif fiyatlarını, beklenen marjı, katkı kârını, riskleri ve senaryo skorunu birlikte görün.",
        "Senaryo",
    )
    result = ensure_scenario_result()
    if not result:
        st.info("Önce Test İhalesi Simülatörü sayfasında bir ihale seçin.")
        return
    scenarios = result["scenarios"].copy()
    st.session_state.best_scenario = scenarios.iloc[0].to_dict()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Model Güven Skoru", f"{result['model_confidence_score']:.1f}/100", "Benzer ihale sayısı ve uyumu", "blue", "🛡️")
    with c2:
        metric_card("Orta Fiyat", format_try(result["corridor"]["predicted_mid_price"]), "Fiyat koridoru merkezi", "green", "🎯")
    with c3:
        metric_card("Geçerli Senaryo", format_int(scenarios["hard_constraints_valid"].sum()), "Sert kuralları geçen", "amber", "✅")

    section_header("Öne Çıkan Senaryolar", "Her kart teklif fiyatı, marj, katkı, risk, kural durumu ve senaryo skorunu birlikte gösterir.", "Senaryo kartları")
    top_cards = scenarios.head(3).reset_index(drop=True)
    cols = st.columns(3, gap="medium")
    card_colors = ["blue", "green", "amber"]
    for idx, (_, scenario) in enumerate(top_cards.iterrows()):
        with cols[idx]:
            status = "good" if bool(scenario["hard_constraints_valid"]) else "bad"
            contribution = float(scenario["proposed_unit_price"]) * float(current_tender().get("quantity", 0)) * float(scenario["computed_margin_pct"]) / 100
            risk_value = max(0.0, 100 - float(scenario["risk_penalty_score"]))
            risk_label = "Düşük" if risk_value >= 75 else "Orta" if risk_value >= 55 else "Yüksek"
            st.markdown(
                f"""
                <div class="scenario-card scenario-card-{card_colors[idx % len(card_colors)]}">
                    <div class="scenario-title">{escape(scenario_name(idx))}</div>
                    <div class="scenario-price">{escape(format_try(scenario["proposed_unit_price"]))}</div>
                    <div class="metric-note">Önerilen Birim Fiyat</div>
                    <div style="margin-top:.65rem">
                    {badge("Kurallara uygun" if scenario["hard_constraints_valid"] else "Kural ihlali", status)}
                    </div>
                    <div class="scenario-row"><span>Beklenen Marj</span><b>{escape(format_pct(scenario["computed_margin_pct"]))}</b></div>
                    <div class="scenario-row"><span>Beklenen Katkı</span><b>{escape(format_try(contribution))}</b></div>
                    <div class="scenario-row"><span>Risk Seviyesi</span><b>{escape(risk_label)}</b></div>
                    <div class="scenario-row"><span>Profil Uyumu</span><b>{scenario["won_profile_fit_score"]:.1f}/100</b></div>
                    <div class="scenario-row"><span>Senaryo Skoru</span><b>{scenario["scenario_score"]:.0f}/100</b></div>
                    <div class="metric-note">Kısa yorum: fiyat bandı, marj ve risk cezası birlikte okunmalıdır.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    table = scenarios[
        [
            "scenario_id",
            "proposed_unit_price",
            "computed_margin_pct",
            "won_profile_fit_score",
            "price_band_fit_score",
            "margin_score",
            "risk_penalty_score",
            "model_confidence_score",
            "scenario_score",
            "hard_constraints_valid",
            "risk_flags",
        ]
    ].copy()
    table.columns = [
        "Senaryo ID",
        "Önerilen Birim Fiyat",
        "Beklenen Marj",
        "Profil Uyumu",
        "Fiyat Bandı Uyumu",
        "Marj Skoru",
        "Risk Cezası",
        "Güven Skoru",
        "Senaryo Skoru",
        "Sert Kural Durumu",
        "Yumuşak Ceza / Risk Notları",
    ]
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Önerilen Birim Fiyat": st.column_config.NumberColumn(format="%.2f TL"),
            "Beklenen Marj": st.column_config.NumberColumn(format="%.2f"),
            "Senaryo Skoru": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
        },
    )


def render_reveal_compare() -> None:
    page_header(
        "Gerçek Sonuçla Karşılaştır",
        "Simülasyon sırasında gizlenen gerçek kazanılmış sonuç bu adımda açılır. Amaç yalnızca fiyat koridorunu kontrol etmek değildir; profil uyumu, emsal kalitesi, cluster ataması, sıra dışılık kontrolü ve senaryo sıralaması birlikte değerlendirilir.",
        "Sonuç Açma",
    )
    row = selected_test_tender()
    result = ensure_scenario_result()
    if row is None or not result:
        st.info("Önce test ihalesi seçip senaryo analizini çalıştırın.")
        return

    if not st.session_state.get("revealed", False):
        warning_box()
        info_callout(
            "Gerçek sonuç açıldığında yalnızca fiyat bandı değil; profil uyumu, emsal kalitesi, cluster ataması, sıra dışılık kontrolü ve senaryo sıralaması da değerlendirilir.",
            "Reveal sonrası ne değişir?",
        )
        st.info("Gerçek kazanılmış fiyat ve marj henüz gizli. Bu bilgi model, senaryo skoru ve AI danışmana verilmedi.")
        if st.button("Gerçek sonucu aç", type="primary"):
            st.session_state.revealed = True
            write_audit_event({"event_type": "actual_result_revealed", "tender_id": row["tender_id"]})
        return

    actual_price = float(row[CANONICAL_PRICE_COLUMN])
    actual_margin = float(row[CANONICAL_MARGIN_COLUMN])
    corridor = result["corridor"]
    scenarios_with_actual = generate_candidate_scenarios(
        current_tender(),
        corridor,
        include_actual={"actual_won_unit_price": actual_price},
    )
    scored = []
    for scenario in scenarios_with_actual:
        validation = validate_scenario(scenario, current_tender(), corridor)
        profile_output = {
            "won_profile_fit_score": result["scenarios"].iloc[0]["won_profile_fit_score"],
        }
        scored.append(score_scenario(scenario, current_tender(), corridor, profile_output, result["model_confidence_score"], validation))
    scored_df = pd.DataFrame(scored)
    rank_pct = actual_rank_percentile(scored_df)
    best = result["scenarios"].iloc[0]
    inside = corridor["predicted_low_price"] <= actual_price <= corridor["predicted_high_price"]
    abs_error = abs(actual_price - corridor["predicted_mid_price"])
    pct_error = abs_error / max(abs(actual_price), 1) * 100
    tender = current_tender() or {}
    similar = result.get("similar", pd.DataFrame())
    quality = retrieval_quality_from_result(result, tender)
    top10_avg_similarity = float(result.get("top10_avg_similarity", 0.0))
    top50_avg_similarity = float(result.get("top50_avg_similarity", quality.get("topk_avg_similarity", 0.0)))
    isolation_status = "Geçmiş profile uygun" if bool(best.get("is_inlier", best.get("won_profile_fit_score", 0) >= 55)) else "Sıra dışı / manuel inceleme önerilir"

    section_header("Profil Uyum Karşılaştırması", "Gerçekten kazanılmış bir test ihalesi için skorun orta veya yüksek çıkması beklenir.")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Kazanılmış Profil Uyum Skoru", format_score(best.get("won_profile_fit_score")), "Gerçek P-Win değildir")
    with c2:
        metric_card("Uyum seviyesi", fit_level(best.get("won_profile_fit_score")), "Geçmiş başarı profiline yakınlık")
    with c3:
        metric_card("Model Güveni", format_score(best.get("model_confidence_score")), "Benzer ihale ve veri sinyali")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        render_kv_card(
            "K-Means Profil Karşılaştırması",
            [
                ("Atandığı cluster", str(best.get("cluster_name", "Hesaplanamadı"))),
                ("Cluster ID", str(best.get("cluster_id", "Hesaplanamadı"))),
                ("Geçmiş ihale sayısı", format_int(best.get("cluster_count"))),
                ("Baskın ürün grubu", str(best.get("cluster_dominant_product_group", "Hesaplanamadı"))),
                ("Medyan fiyat", format_try(best.get("cluster_median_price"))),
                ("Medyan marj", format_pct(best.get("cluster_median_margin"))),
            ],
            "Bu cluster, test ihalesinin geçmişte kazanılmış hangi başarı profiline yakın konumlandığını gösterir.",
        )
    with right:
        render_kv_card(
            "Isolation Forest Karşılaştırması",
            [
                ("Durum", isolation_status),
                ("Inlier skoru", format_score(best.get("inlier_score"))),
                ("Yorum", "Model bu kazanılmış test ihalesini geçmiş profile uygun tanımış." if isolation_status == "Geçmiş profile uygun" else "Bu ihale kazanılmış olsa bile geçmiş profilden sıra dışı olabilir; manuel inceleme gerekir."),
            ],
            "Bu sonuç kazanma olasılığı değildir; geçmiş kazanılmış veri dağılımına uygunluk kontrolüdür.",
        )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Emsal İhale Kalitesi", "Sistemin gerçek sonucu açmadan önce seçtiği emsal havuzunun ne kadar tutarlı olduğunu gösterir.")
    e1, e2, e3, e4 = st.columns(4, gap="medium")
    with e1:
        metric_card("Top-10 Ortalama Benzerlik", f"{top10_avg_similarity:.2f}", "En yakın emsal seti")
    with e2:
        metric_card("Top-50 Ortalama Benzerlik", f"{top50_avg_similarity:.2f}", "Geniş emsal havuzu")
    with e3:
        metric_card("Ürün Grubu Eşleşmesi", format_pct(quality.get("product_group_match_rate", 0) * 100), "Emsal havuzunda aynı ürün grubu")
    with e4:
        metric_card("Bölge Eşleşmesi", format_pct(quality.get("region_match_rate", 0) * 100), "Emsal havuzunda aynı bölge")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Fiyat Koridoru Karşılaştırması", "Fiyat bandı çıktılardan yalnızca biridir; profil, emsal ve senaryo sinyalleriyle birlikte okunmalıdır.")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Tahmin düşük", format_try(corridor["predicted_low_price"]), "Koridor alt bandı")
    with c2:
        metric_card("Tahmin orta", format_try(corridor["predicted_mid_price"]), "Koridor merkezi")
    with c3:
        metric_card("Tahmin yüksek", format_try(corridor["predicted_high_price"]), "Koridor üst bandı")
    with c4:
        metric_card("Gerçek kazanılmış fiyat", format_try(actual_price), "Reveal sonrası")

    c5, c6, c7 = st.columns(3, gap="medium")
    with c5:
        metric_card("Band içinde mi?", "Evet" if inside else "Hayır", "Gerçek fiyat band kontrolü")
    with c6:
        metric_card("Mutlak hata", format_try(abs_error), "Orta fiyata göre")
    with c7:
        metric_card("Yüzde hata", format_pct(pct_error), "Orta fiyata göre")

    chart_df = pd.DataFrame(
        {
            "Gösterge": ["Düşük", "Orta", "Yüksek", "Gerçek", "Seçilen"],
            "Birim Fiyat": [
                corridor["predicted_low_price"],
                corridor["predicted_mid_price"],
                corridor["predicted_high_price"],
                actual_price,
                float(best["proposed_unit_price"]),
            ],
        }
    )
    st.bar_chart(chart_df, x="Gösterge", y="Birim Fiyat", color="#2563eb")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Senaryo Sıralaması", "Gerçek kazanılmış senaryonun aday senaryo skorları içindeki konumu.")
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        metric_card("Gerçek kazanılmış senaryo sıralaması", format_pct(rank_pct), "Percentile değeri yükseldikçe daha iyi")
    with r2:
        render_small_card("Yorum", scenario_rank_comment(rank_pct))

    comparison = pd.DataFrame(
        [
            {
                "İhale ID": row["tender_id"],
                "Gerçek Kazanılmış Birim Fiyat": actual_price,
                "Tahmin Düşük": corridor["predicted_low_price"],
                "Tahmin Orta": corridor["predicted_mid_price"],
                "Tahmin Yüksek": corridor["predicted_high_price"],
                "Band İçinde mi": "Evet" if inside else "Hayır",
                "Mutlak Hata": abs_error,
                "Yüzde Hata": pct_error,
                "Gerçek Marj": actual_margin,
                "Senaryo Skoru": float(best["scenario_score"]),
                "Profil Uyum Skoru": float(best.get("won_profile_fit_score", 0)),
                "Cluster": best.get("cluster_name", "Hesaplanamadı"),
                "Isolation Forest": isolation_status,
                "Top-10 Ortalama Benzerlik": top10_avg_similarity,
                "Gerçek Senaryo Sıralaması": rank_pct,
            }
        ]
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    message = (
        "Gerçek kazanılmış fiyat, sistemin önerdiği fiyat koridorunun içinde kaldı. "
        "Bu, fiyat koridorunun bu ihale tipi için geçmiş kazanım davranışını makul şekilde yakaladığını gösterir."
        if inside
        else "Gerçek kazanılmış fiyat koridorun dışında kaldı. Bu ihale tipi için manuel fiyat incelemesi önerilir."
    )
    st.markdown(f"<div class='info-box'>{escape(message)}</div>", unsafe_allow_html=True)
    st.download_button("Karşılaştırma CSV indir", dataframe_to_csv_bytes(comparison), "senaryo_karsilastirma.csv")


def render_backtest() -> None:
    page_header(
        "Backtest Sonuçları",
        "Test yılındaki ihaleler, gerçek sonucu gizlenerek simüle edilir. Sonra sonuç açılır ve sistem performansı ölçülür.",
        "Backtest",
    )
    data = load_active_data()
    with st.spinner("Backtest çalışıyor..."):
        split = temporal_split(data)
        results = cached_backtest(data)
    st.session_state.backtest_results = results
    metrics = price_corridor_metrics(results)
    opt = optimizer_metrics(results)
    forbidden_rate = 1 - float((results["advisor_validation_status"] == "pass").mean()) if not results.empty else 0
    inlier_recall = float((results["won_profile_fit_score"] >= 45).mean()) if not results.empty else 0

    info_callout(
        "Bu ekran pseudo-live test mantığıyla çalışır: test ihalesinin gerçek kazanılmış fiyatı ve marjı model girdisinden gizlenir, sistem çıktı ürettikten sonra gerçek sonuçla karşılaştırılır.",
        "Backtest okuma notu:",
    )
    info_callout(
        "Backtest’in amacı kazanma/kaybetme tahmini doğruluğunu ölçmek değildir. Amaç, sistemin geçmişte kazanılmış ama sonucu gizlenmiş ihaleleri geçmiş başarı profillerine doğru yerleştirip yerleştiremediğini ve fiyat/senaryo önerilerinin tarihsel sonuçlarla ne kadar tutarlı olduğunu ölçmektir.",
        "Backtest amacı:",
    )
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Fiyat Koridoru Kapsama Oranı", format_pct(metrics["band_coverage"] * 100), "Gerçek kazanılmış fiyatların ne kadarı önerilen fiyat bandının içinde kaldı?", "green", "🎯")
    with c2:
        metric_card("MAPE", format_pct(metrics["mape"]), "Orta fiyatın gerçek fiyattan ortalama yüzde sapması.", "blue", "📉")
    with c3:
        metric_card("Ortalama Koridor Genişliği", format_try(metrics["average_band_width"]), "Bandın ne kadar geniş olduğunu gösterir.", "amber", "↔️")

    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        metric_card("Coverage Adjusted Band Score", f"{metrics['coverage_adjusted_band_score']:.2f}", "Kapsama oranını band genişliğiyle birlikte değerlendirir.", "purple", "⚖️")
    with c5:
        metric_card("Kazanılmış Test İhalelerini Geçmiş Profile Uygun Tanıma Oranı", format_pct(inlier_recall * 100), "Profil modelinin inlier yakalama gücü.", "green", "✅")
    with c6:
        metric_card("Yasak İddia Üretme Oranı", format_pct(forbidden_rate * 100), "AI Danışman güvenlik kontrolü. Hedef sıfırdır.", "red", "🛡️")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Baz Model Karşılaştırması", "Mevcut yöntem basit ortalama/medyan ve benzeri baz yaklaşımlarla kıyaslanır.", "Kıyas")
    baseline = baseline_predictions(pd.concat([split["train"], split["validation"]]), split["test"])
    baseline = baseline.rename(
        columns={
            "Model": "Yöntem",
            "MAE": "Ortalama Mutlak Hata",
            "MAPE": "Ortalama Yüzde Hata",
            "Coverage": "Kapsama",
            "Avg Band Width": "Ortalama Band Genişliği",
        }
    )
    current_row = pd.DataFrame(
        [
            {
                "Yöntem": "Tender IQ mevcut yöntem",
                "Ortalama Mutlak Hata": metrics["mae"],
                "Ortalama Yüzde Hata": metrics["mape"],
                "Kapsama": metrics["band_coverage"],
                "Ortalama Band Genişliği": metrics["average_band_width"],
            }
        ]
    )
    st.dataframe(pd.concat([baseline, current_row], ignore_index=True), hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Segment Bazlı Performans", "Ürün, bölge ve profil kırılımlarında hata ve kapsama değerleri.", "Segment")
    segment_display = segment_level_metrics(results)
    if "segment_value" in segment_display.columns:
        segment_display["segment_value"] = segment_display["segment_value"].astype(str)
    st.dataframe(segment_display, hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Sızıntı Kontrolü", "Sonuç açılmadan önce gerçek sonuç alanlarının modele girmediği doğrulanır.", "Audit")
    leak_status = results["leakage_audit_status"].value_counts().reset_index()
    leak_status.columns = ["Audit durumu", "İhale sayısı"]
    status = "success" if (results["leakage_audit_status"] == "pass").all() else "danger"
    st.markdown(
        f"<div class='glass-card'>{badge('Sızıntı yok' if status == 'success' else 'Sızıntı uyarısı', status)}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(leak_status, hide_index=True, width="stretch")

    with st.expander("İhale bazlı sonuç detayı", expanded=False):
        st.dataframe(results, hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Export", "Backtest çıktıları dışa aktarılabilir.", "Rapor")
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        st.download_button("Backtest Raporu", dataframe_to_csv_bytes(pd.DataFrame([metrics])), "backtest_raporu.csv")
    with e2:
        st.download_button("Tender-Level Sonuçlar", dataframe_to_csv_bytes(results), "tender_level_sonuclar.csv")
    with e3:
        st.download_button("Leakage Audit", dataframe_to_csv_bytes(leak_status), "leakage_audit.csv")


def render_similar_tenders() -> None:
    page_header(
        "Benzer İhaleler",
        "Seçili test ihalesine benzeyen geçmiş kazanılmış ihaleleri ve eşleşme oranlarını gösterir.",
        "Emsal",
    )
    tender = current_tender()
    if not tender:
        st.info("Önce Test İhalesi Simülatörü sayfasında bir ihale seçin.")
        return
    info_callout(
        "Benzerlik motoru ürün adı, ürün grubu, kurum, bölge, ihale tipi ve miktar bilgisinden ihale profili oluşturur. TF-IDF bu profili sayısal vektöre çevirir; cosine similarity en yakın geçmiş kazanılmış ihaleleri sıralar.",
        "TF-IDF + cosine similarity:",
    )
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    retriever = RetrievalEngine.fit(get_history_frame())
    similar = retriever.retrieve(tender, top_k=50)
    quality = retrieval_quality(similar, tender)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Ortalama Benzerlik", f"{quality['topk_avg_similarity']:.2f}", "İlk 50 benzer ihale", "blue", "🔎")
    with c2:
        metric_card("Ürün Grubu Eşleşme Oranı", format_pct(quality["product_group_match_rate"] * 100), "Top-K içinde aynı ürün grubu", "green", "🧭")
    with c3:
        metric_card("Bölge Eşleşme Oranı", format_pct(quality["region_match_rate"] * 100), "Top-K içinde aynı bölge", "purple", "📍")
    with c4:
        metric_card("Miktar Bandı Eşleşme Oranı", format_pct(quality["quantity_band_match_rate"] * 100), "Yakın ölçek oranı", "amber", "📦")

    display = similar[
        [
            "tender_id",
            "product_group",
            "product_name",
            "buyer_institution",
            "region",
            "quantity",
            "overall_similarity_score",
            CANONICAL_PRICE_COLUMN,
            CANONICAL_MARGIN_COLUMN,
        ]
    ].copy()
    display.columns = [
        "İhale ID",
        "Ürün Grubu",
        "Ürün",
        "Kurum",
        "Bölge",
        "Miktar",
        "Benzerlik Skoru",
        "Tarihsel Kazanılmış Fiyat",
        "Marj",
    ]
    section_header("Top-K Benzer İhaleler", "Benzerlik skoru yükseldikçe yeni ihale geçmiş kazanılmış profile daha yakın görünür.", "Emsal liste")
    st.dataframe(
        display.head(25),
        hide_index=True,
        width="stretch",
        column_config={
            "Benzerlik Skoru": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Tarihsel Kazanılmış Fiyat": st.column_config.NumberColumn(format="%.2f TL"),
            "Marj": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_advisor() -> None:
    page_header(
        "AI Danışman",
        "Sistem çıktısını Türkçe ve güvenli şekilde yorumlar. Soru sorabilir, gerekirse deterministik sistem yorumu alabilirsiniz.",
        "AI Danışman",
    )
    warning_box()
    info_callout(PWIN_PROXY_EXPLANATION, "P-Win yerine ne var?")
    result = ensure_scenario_result()
    if not result:
        st.info("Önce Test İhalesi Simülatörü sayfasında bir ihale seçin.")
        return
    best = result["scenarios"].iloc[0].to_dict()
    context = advisor_context(result, best)
    advisor = build_fallback_advisor(context)
    validation = validate_advisor_output(advisor)
    st.session_state.advisor_output = advisor
    st.session_state.advisor_validation = validation

    context_signature = json.dumps(
        {
            "tender_id": context.get("tender_id"),
            "scenario_score": round(float(context.get("scenario_score", 0)), 3),
            "revealed": context.get("revealed", False),
        },
        sort_keys=True,
    )
    if st.session_state.get("advisor_chat_context_signature") != context_signature:
        st.session_state.advisor_chat_context_signature = context_signature
        st.session_state.advisor_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Analiz bağlamı hazır. Bu ihalenin profil uyumunu, fiyat koridorunu, marjını, risklerini "
                    "ve benzer ihalelerini açıklayabilirim."
                ),
            }
        ]

    quick_questions = [
        "Bu ihale neden bu skoru aldı?",
        "Benzer ihaleler ne söylüyor?",
        "Fiyat koridoru nasıl yorumlanmalı?",
        "Riskli görünen noktalar neler?",
        "Manuel inceleme gerekir mi?",
        "Bu ihale hangi cluster’a benziyor?",
        "Isolation Forest sonucu ne demek?",
    ]

    left, right = st.columns([0.9, 1.55], gap="large")
    with left:
        section_header("Bağlam Paneli", "Danışman yalnızca bu yapılandırılmış çıktıları yorumlar.", "Güvenli bağlam")
        corridor = context.get("corridor", {})
        risk_flags = context.get("risk_flags", [])
        risk_status = "Uyarı var" if risk_flags else "Düşük"
        leak = st.session_state.get("leakage_audit", {"audit_status": "pass"})
        context_rows = [
            ("Seçili ihale", str(context.get("tender_id", "-"))),
            ("Ürün grubu", str(context.get("product_group", "-"))),
            ("Profil uyumu", f"{context.get('won_profile_fit_score', 0):.1f}/100"),
            ("Fiyat koridoru", f"{format_try(corridor.get('predicted_low_price'))} - {format_try(corridor.get('predicted_high_price'))}"),
            ("Risk seviyesi", risk_status),
            ("Model güveni", f"{context.get('model_confidence_score', 0):.1f}/100"),
            ("Cluster", str(context.get("cluster_name", "Kazanılmış profil kümesi"))),
        ]
        rows_html = "".join(
            f"<div class='scenario-row'><span>{escape(label)}</span><b>{escape(value)}</b></div>"
            for label, value in context_rows
        )
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='section-title'>Seçili ihale bağlamı</div>
                <div class='metric-note'>Bu panel dışındaki bilgi danışman yanıtına dayanak yapılmaz.</div>
                {rows_html}
                <div style='margin-top:.85rem'>{badge("Sızıntı yok" if leak.get("audit_status") == "pass" else "Sızıntı uyarısı", "success" if leak.get("audit_status") == "pass" else "danger")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <div class='chat-shell'>
                    <div class='chat-header'>
                        <div class='chat-avatar'>AI</div>
                        <div>
                            <div class='chat-header-title'>AI Danışman</div>
                            <div class='chat-header-subtitle'>Model çıktıları üzerinden güvenli açıklama üretir. Gerçek kazanma olasılığı vermez.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='chat-input-area quick-question'>", unsafe_allow_html=True)
            qcols = st.columns(3, gap="small")
            selected_question = None
            for idx, question in enumerate(quick_questions):
                with qcols[idx % 3]:
                    if st.button(question, key=f"quick_advisor_{idx}", width="stretch"):
                        selected_question = question
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='chat-body'>", unsafe_allow_html=True)
            for message in st.session_state.get("advisor_chat_messages", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            st.markdown("</div>", unsafe_allow_html=True)

        typed_question = st.chat_input("Bu ihale hakkında sorunuzu yazın...")
        user_question = selected_question or typed_question
        if user_question:
            st.session_state.advisor_chat_messages.append({"role": "user", "content": user_question})
            with st.spinner("Yanıt hazırlanıyor..."):
                llm_payload = call_guarded_llm(context, user_question)
                if llm_payload:
                    assistant_text = advisor_payload_to_chat_text(llm_payload)
                    st.session_state.advisor_output = llm_payload
                    st.session_state.advisor_validation = validate_advisor_output(llm_payload)
                else:
                    assistant_text = fallback_chat_answer(user_question, context, advisor)
                    st.session_state.advisor_validation = validation
            st.session_state.advisor_chat_messages.append({"role": "assistant", "content": assistant_text})
            st.rerun()

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    with st.expander("Sistem yorumu, doğrulama ve bağlam", expanded=False):
        st.markdown(f"**Özet:** {advisor['summary']}")
        st.markdown(f"**Profil uyumu:** {advisor['profile_fit_explanation']}")
        st.markdown(f"**Fiyat koridoru:** {advisor['price_corridor_explanation']}")
        st.markdown(f"**Risk:** {advisor['risk_explanation']}")
        st.json(st.session_state.get("advisor_validation", validation))
        safe_context = dict(context)
        if not safe_context.get("revealed"):
            safe_context.pop("revealed_actual", None)
        st.json(safe_context, expanded=False)


def render_reports() -> None:
    page_header(
        "Raporlar ve Kontroller",
        "Backtest, senaryo, sızıntı kontrolü, segment metrikleri, expert review ve model card çıktıları.",
        "Rapor",
    )
    results = st.session_state.get("backtest_results")
    if results is None:
        with st.spinner("Raporlar için backtest hazırlanıyor..."):
            results = cached_backtest(load_active_data())
            st.session_state.backtest_results = results
    segment = segment_level_metrics(results)
    if "segment_value" in segment.columns:
        segment["segment_value"] = segment["segment_value"].astype(str)
    model_card = generate_model_card(price_corridor_metrics(results))
    advisor_validation = st.session_state.get("advisor_validation", {"advisor_validation_status": "henüz çalışmadı"})
    leakage_ok = bool((results["leakage_audit_status"] == "pass").all())
    advisor_ok = advisor_validation.get("advisor_validation_status") in {"pass", "geçti", "henüz çalışmadı"}
    hard_violation_rate = float((~results["hard_constraints_valid"].astype(bool)).mean() * 100)
    audit_cards = [
        (
            "Leakage Audit",
            "Sızıntı kontrolü",
            "Sonuç açılmadan önce gerçek kazanılmış fiyat, final tutar ve gerçek marj alanlarının modele girmediğini kontrol eder.",
            "Sızıntı yok" if leakage_ok else "Uyarı",
            "success" if leakage_ok else "danger",
        ),
        (
            "Forbidden Claim Detector",
            "Yasak iddia kontrolü",
            "AI Danışman çıktısında garanti, kesin sonuç veya gerçek P(win) iddiası olup olmadığını denetler.",
            "Geçti" if advisor_ok else "Uyarı",
            "success" if advisor_ok else "danger",
        ),
        (
            "Advisor Validation",
            "Danışman doğrulama",
            "Yanıtların yapılandırılmış bağlama ve guardrail kurallarına uyduğunu izler.",
            str(advisor_validation.get("advisor_validation_status", "henüz çalışmadı")),
            "success" if advisor_ok else "warning",
        ),
        (
            "Hard Constraint Check",
            "Sert kural kontrolü",
            "Minimum marj ve kural ihlali gibi geçersiz senaryo durumlarını raporlar.",
            format_pct(hard_violation_rate),
            "success" if hard_violation_rate == 0 else "warning",
        ),
        (
            "Export Status",
            "Dışa aktarım",
            "Backtest, tender-level sonuç, senaryo, audit, model card ve expert review çıktıları hazırlanır.",
            "Hazır",
            "success",
        ),
    ]
    for start in range(0, len(audit_cards), 3):
        columns = st.columns(3, gap="medium")
        for offset, (title, tr_title, body, status_text, status) in enumerate(audit_cards[start : start + 3]):
            idx = start + offset
            color = ["blue", "green", "purple", "amber", "cyan"][idx % 5]
            with columns[offset]:
                st.markdown(
                    f"""
                    <div class='model-card model-card-{color}'>
                        <div class='model-title'>{escape(tr_title)}</div>
                        <div class='method-body'>{escape(body)}</div>
                        <div style='margin-top:.75rem'>{badge(status_text, status)}</div>
                        <div class='metric-note'>Son çalışma: oturum içi son hesaplama</div>
                        <div class='metric-note'>{escape(title)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Export Grupları", "Rapor ve denetim çıktıları.", "Dışa aktar")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.download_button("Backtest Raporu", dataframe_to_csv_bytes(pd.DataFrame([price_corridor_metrics(results)])), "backtest_ozeti.csv")
        st.download_button("Tender-Level Sonuçlar", dataframe_to_csv_bytes(results), "ihale_bazli_sonuclar.csv")
    with c2:
        scenario_result = st.session_state.get("scenario_result", {}).get("scenarios", pd.DataFrame())
        if isinstance(scenario_result, pd.DataFrame) and not scenario_result.empty:
            st.download_button("Senaryo Karşılaştırması", dataframe_to_csv_bytes(scenario_result), "senaryo_karsilastirma.csv")
        st.download_button("Leakage Audit", dataframe_to_csv_bytes(results[["tender_id", "leakage_audit_status"]]), "leakage_audit.csv")
    with c3:
        st.download_button("Model Card", model_card, "model_karti.md")
        st.download_button("Expert Review Template", dataframe_to_csv_bytes(expert_review_template(results)), "uzman_inceleme_sablonu.csv")
    with st.expander("Segment metrikleri ve audit tablo detayı", expanded=False):
        st.dataframe(segment, hide_index=True, width="stretch")


page = render_sidebar()

if page == "Ana Sayfa":
    render_home()
elif page == "Veri Seti ve Kalite Kontrol":
    render_data_quality()
elif page == "Metodoloji":
    render_methodology()
elif page == "Test İhalesi Simülatörü":
    render_test_simulator()
elif page == "Senaryo Analizi":
    render_scenario_analysis()
elif page == "Gerçek Sonuçla Karşılaştır":
    render_reveal_compare()
elif page == "Backtest Sonuçları":
    render_backtest()
elif page == "Benzer İhaleler":
    render_similar_tenders()
elif page == "AI Danışman":
    render_advisor()
elif page == "Raporlar ve Kontroller":
    render_reports()
