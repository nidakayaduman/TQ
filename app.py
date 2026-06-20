"""Tender IQ Agentic Bid Advisor - Turkish Streamlit dashboard."""

from __future__ import annotations

import json
import os
import re
import warnings
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.advisor.fallback_advisor import build_fallback_advisor
from src.advisor.forbidden_claim_detector import detect_forbidden_claims
from src.advisor.grounding_validator import validate_grounding
from src.advisor.output_validator import validate_advisor_output
from src.advisor.prompt_builder import build_advisor_prompt
from src.config_loader import active_config_summary, load_app_config, load_scenario_weights
from src.constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from src.evaluation.backtest_runner import actual_rank_percentile, run_backtest
from src.evaluation.baseline_models import baseline_predictions, predict_baseline_prices
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

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")

TURKISH_WARNING = (
    "Skor gerçek kazanma olasılığı değil, geçmiş kazanılmış ihale profiline uyum göstergesidir."
)

PWIN_PROXY_EXPLANATION = (
    "Bu MVP’de gerçek kazanma olasılığı hesaplanmaz. Bunun yerine karar destek göstergesi "
    "olarak Kazanılmış Profil Uyum Skoru kullanılır. Bu skor; emsal benzerlik, K-Means başarı profili, "
    "Isolation Forest uygunluğu, fiyat bandı uyumu, karlılık/risk dengesi ve model güveninden beslenir."
)

PAGE_NAMES = [
    "Ana Sayfa",
    "Veri Seti ve Kalite Kontrol",
    "Metodoloji",
    "Test için İhale Seç",
    "Emsal İhale Analizi",
    "Profil Uyum Analizi",
    "Fiyat Koridoru ve Model Karşılaştırması",
    "Teklif Senaryoları",
    "Gerçek Sonuçla Karşılaştır",
    "Backtest Sonuçları",
    "AI Danışman",
    "Raporlar ve Kontroller",
]

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
            .card-grid {
                display: grid;
                gap: 1.35rem;
                width: 100%;
                align-items: stretch;
                padding: 1.35rem;
                border-radius: 30px;
                border: 1px solid rgba(255,255,255,0.58);
                background:
                    radial-gradient(circle at 18% 12%, rgba(191, 219, 254, 0.24), transparent 34%),
                    radial-gradient(circle at 82% 18%, rgba(196, 181, 253, 0.18), transparent 36%),
                    rgba(255,255,255,0.34);
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.28), 0 20px 70px rgba(15,23,42,0.05);
                backdrop-filter: blur(20px);
            }
            .card-grid.two-col { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .card-grid.three-col { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .card-grid.auto-col { grid-template-columns: repeat(auto-fit, minmax(285px, 1fr)); }
            .premium-card {
                position: relative;
                min-height: 214px;
                height: 100%;
                padding: 1.65rem;
                border-radius: 26px;
                border: 1px solid rgba(255,255,255,0.72);
                background: rgba(255,255,255,0.74);
                box-shadow: 0 18px 45px rgba(15, 23, 42, 0.075);
                backdrop-filter: blur(18px);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow: hidden;
            }
            .premium-card:before {
                content: '';
                position: absolute;
                inset: 0;
                pointer-events: none;
                opacity: .95;
            }
            .premium-card.card-blue:before { background: radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 42%); }
            .premium-card.card-purple:before { background: radial-gradient(circle at top left, rgba(167,139,250,0.16), transparent 42%); }
            .premium-card.card-mint:before { background: radial-gradient(circle at top left, rgba(45,212,191,0.14), transparent 42%); }
            .premium-card.card-green:before { background: radial-gradient(circle at top left, rgba(74,222,128,0.13), transparent 42%); }
            .premium-card.card-amber:before { background: radial-gradient(circle at top left, rgba(251,191,36,0.13), transparent 42%); }
            .premium-card.card-cyan:before { background: radial-gradient(circle at top left, rgba(34,211,238,0.13), transparent 42%); }
            .premium-card.card-red:before { background: radial-gradient(circle at top left, rgba(248,113,113,0.12), transparent 42%); }
            .premium-card > * { position: relative; z-index: 1; }
            .premium-card.metric-size { min-height: 148px; }
            .premium-card.large-size { min-height: 268px; }
            .premium-card.scenario-size { min-height: 338px; }
            .card-icon-row { display: flex; align-items: center; gap: .75rem; margin-bottom: 1rem; }
            .card-icon {
                width: 44px; height: 44px; border-radius: 15px;
                display: inline-flex; align-items: center; justify-content: center;
                background: rgba(255,255,255,0.9);
                border: 1px solid rgba(15,23,42,0.06);
                box-shadow: inset 0 0 0 1px rgba(15,23,42,0.025);
                font-weight: 680;
            }
            .card-title {
                color: #0f172a;
                font-size: 1.16rem;
                line-height: 1.18;
                font-weight: 700;
                letter-spacing: 0;
                margin: 0 0 .55rem;
            }
            .card-value {
                color: #0f172a;
                font-size: 1.55rem;
                line-height: 1.1;
                font-weight: 720;
                margin: .1rem 0 .55rem;
                overflow-wrap: anywhere;
            }
            .card-body {
                color: #64748b;
                font-size: .88rem;
                line-height: 1.5;
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            .card-list {
                color: #475569;
                font-size: .86rem;
                line-height: 1.55;
                margin-top: .85rem;
            }
            .card-line { display: flex; justify-content: space-between; gap: .8rem; padding: .32rem 0; border-top: 1px solid rgba(15,23,42,.07); }
            .card-line b { color: #0f172a; }
            .card-footer { margin-top: 1.15rem; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .45rem; }
            .card-pill {
                font-size: .75rem;
                color: #475569;
                background: rgba(15,23,42,0.055);
                border-radius: 999px;
                padding: .38rem .62rem;
                border: 1px solid rgba(15,23,42,0.04);
            }
            .advisor-panel { padding: .25rem .1rem .1rem; }
            .soft-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,.18), transparent); margin: 1.6rem 0; }
            .divider-space { margin-top: 1.35rem; }
            @media (max-width: 1100px) {
                .model-grid, .method-grid, .score-mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .card-grid.three-col { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 760px) {
                .scope-pill { float: none; margin-bottom: .8rem; }
                .model-grid, .method-grid, .score-mini-grid { grid-template-columns: 1fr; }
                .card-grid.three-col, .card-grid.two-col, .card-grid.auto-col { grid-template-columns: 1fr; }
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


def format_optional_try(value: float | int | None, empty: str = "-") -> str:
    if value is None or pd.isna(value):
        return empty
    return format_try(value)


def fit_level(score: float | int | None) -> str:
    if score is None or pd.isna(score):
        return "Hesaplanamadı"
    value = float(score)
    if value >= 70:
        return "Yüksek uyum"
    if value >= 45:
        return "Orta uyum"
    return "Düşük uyum"


def profile_status_from_best(best: dict[str, Any] | pd.Series) -> tuple[str, str]:
    is_inlier = bool(best.get("is_inlier", best.get("won_profile_fit_score", 0) >= 55))
    if is_inlier:
        return "Geçmiş profile uygun", "good"
    return "Sıra dışı / manuel inceleme önerilir", "warn"


def profile_business_comment(best: dict[str, Any] | pd.Series) -> str:
    product_group = best.get("cluster_dominant_product_group", "benzer ürün grupları")
    buyer_or_region = best.get("cluster_dominant_institution_type") or best.get("cluster_dominant_region", "benzer alıcı profili")
    quantity = best.get("cluster_average_quantity")
    quantity_text = "orta/yüksek hacimli" if pd.notna(quantity) and float(quantity) >= 50000 else "standart hacimli"
    return (
        f"Bu ihale, geçmişte kazanılmış {product_group} ağırlıklı, {buyer_or_region} karakterli "
        f"ve {quantity_text} ihalelere benziyor."
    )


def isolation_business_comment(best: dict[str, Any] | pd.Series) -> str:
    label, _ = profile_status_from_best(best)
    if label.startswith("Sıra dışı"):
        return (
            "Bu ihale kazanılmış veri içinde daha az görülen bir profile benziyor olabilir. "
            "Miktar, kurum, ürün grubu, fiyat veya teslim süresi gibi alanlarda geçmiş örneklerden ayrıştığı için manuel inceleme önerilir."
        )
    return (
        "Bu ihale geçmişte kazanılmış ihale dağılımının dışında görünmüyor. "
        "Sistem bu ihaleyi mevcut başarı profilleriyle uyumlu buluyor."
    )


def require_test_tender_message() -> None:
    st.info("Önce Test için İhale Seç sayfasında bir ihale seçip simülasyonu çalıştırın.")


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
    st.markdown(
        premium_card_html(title, body, icon="•", footer_html=badge_html, size="large-size", color="blue"),
        unsafe_allow_html=True,
    )


def render_kv_card(title: str, rows: list[tuple[str, str]], note: str = "") -> None:
    st.markdown(
        premium_card_html(title, note, icon="•", lines=rows, size="large-size", color="purple"),
        unsafe_allow_html=True,
    )


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
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str = "", color: str = "blue", icon: str = "📊") -> None:
    st.markdown(
        premium_card_html(title, subtitle, icon=icon, value=value, color=color, size="metric-size"),
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", color: str = "blue", icon: str = "📊") -> None:
    render_metric_card(label, value, note, color, icon)


def premium_card_html(
    title: str,
    body: str,
    icon: str = "•",
    value: str = "",
    pill: str = "",
    color: str = "blue",
    size: str = "",
    lines: list[tuple[str, str]] | None = None,
    footer_html: str = "",
) -> str:
    lines_html = ""
    if lines:
        lines_html = "<div class='card-list'>" + "".join(
            f"<div class='card-line'><span>{escape(line_label)}</span><b>{escape(line_value)}</b></div>"
            for line_label, line_value in lines
        ) + "</div>"
    value_html = f"<div class='card-value'>{escape(value)}</div>" if value else ""
    footer = (
        f"<div class='card-footer'>{footer_html}</div>"
        if footer_html
        else f"<div class='card-footer'><span class='card-pill'>{escape(pill)}</span></div>"
        if pill
        else ""
    )
    return (
        f"<div class='premium-card card-{escape(color)} {escape(size)}'>"
        "<div>"
        f"<div class='card-icon-row'><span class='card-icon'>{escape(icon)}</span></div>"
        f"<div class='card-title'>{escape(title)}</div>"
        f"{value_html}"
        f"<div class='card-body'>{escape(body)}</div>"
        f"{lines_html}"
        "</div>"
        f"{footer}"
        "</div>"
    )


def render_premium_grid(items: list[dict[str, Any]], columns: int = 3, size: str = "") -> None:
    grid_class = "three-col" if columns == 3 else "two-col" if columns == 2 else "auto-col"
    html = "".join(
        premium_card_html(
            title=str(item.get("title", "")),
            body=str(item.get("body", "")),
            icon=str(item.get("icon", "•")),
            value=str(item.get("value", "")),
            pill=str(item.get("pill", "")),
            color=str(item.get("color", "blue")),
            size=size or str(item.get("size", "")),
            lines=item.get("lines"),
        )
        for item in items
    )
    st.markdown(f"<div class='card-grid {grid_class}'>{html}</div>", unsafe_allow_html=True)


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
    st.markdown(
        premium_card_html(title, body, icon="•", pill=kicker, footer_html=status_html, color="blue", size="large-size"),
        unsafe_allow_html=True,
    )


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
    render_premium_grid(
        [
            {
                "icon": f"{idx + 1:02d}",
                "title": title,
                "body": body,
                "color": colors[idx % len(colors)],
                "pill": "Adım",
            }
            for idx, (title, body) in enumerate(items)
        ],
        columns=3,
        size="large-size",
    )


def render_model_grid(items: list[tuple[str, str, str, str]]) -> None:
    render_premium_grid(
        [
            {
                "icon": icon,
                "title": title,
                "body": body,
                "color": color,
                "pill": "Model",
            }
            for icon, title, body, color in items
        ],
        columns=3,
        size="large-size",
    )


def render_formula_card() -> None:
    weights = load_scenario_weights()
    risk_weight = abs(float(weights.get("risk_penalty_score", -0.10))) * 100
    st.markdown(
        f"""
        <div class='formula-panel'>
            <div class='formula-title'>Senaryo Skoru</div>
            <div class='formula-line'><span>%{weights.get("won_profile_fit_score", 0) * 100:.0f} Profil Uyumu</span><span>geçmiş kazanım profili</span></div>
            <div class='formula-line'><span>+ %{weights.get("price_band_fit_score", 0) * 100:.0f} Fiyat Bandı Uyumu</span><span>koridor hizası</span></div>
            <div class='formula-line'><span>+ %{weights.get("margin_score", 0) * 100:.0f} Karlılık Skoru</span><span>beklenen karlılık sağlığı</span></div>
            <div class='formula-line'><span>+ %{weights.get("model_confidence_score", 0) * 100:.0f} Model Güveni</span><span>veri ve benzerlik gücü</span></div>
            <div class='formula-line'><span>- %{risk_weight:.0f} Risk Cezası</span><span>manuel inceleme sinyalleri</span></div>
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
    baseline = predict_baseline_prices(get_history_frame(), tender) if tender else pd.DataFrame()
    quality = retrieval_quality_from_result(result, tender)
    scenario_weights = load_scenario_weights()
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
        "price_methods": {
            "similarity_corridor": result["corridor"],
            "baseline_predictions": baseline.to_dict(orient="records") if not baseline.empty else [],
        },
        "retrieval_quality": quality,
        "score_weights": {
            "scenario_score": {
                "won_profile_fit_score": scenario_weights.get("won_profile_fit_score", 0.30),
                "price_band_fit_score": scenario_weights.get("price_band_fit_score", 0.25),
                "margin_score": scenario_weights.get("margin_score", 0.20),
                "model_confidence_score": scenario_weights.get("model_confidence_score", 0.15),
                "risk_penalty_score": scenario_weights.get("risk_penalty_score", -0.10),
            },
            "won_profile_fit_score": {
                "historical_distribution_fit": 0.65,
                "success_group_closeness": 0.35,
            },
        },
        "similar_tender_count": len(result["similar"]),
        "similar_tenders": similar[
            ["tender_id", "product_group", "product_name", "buyer_institution", "region", "quantity", "overall_similarity_score"]
        ].to_dict(orient="records"),
        "cluster_name": best.get("cluster_name", "Kazanılmış profil grubu"),
        "isolation_forest": {
            "status": "Geçmiş profile uygun" if bool(best.get("is_inlier", False)) else "Sıra dışı / manuel inceleme önerilir",
            "profile_fit_score": best.get("inlier_score"),
            "training_normal_rate": best.get("training_inlier_rate"),
            "training_manual_review_rate": best.get("training_anomaly_rate"),
            "sensitivity_setting": best.get("isolation_contamination"),
        },
        "kmeans": {
            "cluster_id": best.get("cluster_id"),
            "cluster_name": best.get("cluster_name"),
            "cluster_count": best.get("cluster_count"),
            "dominant_product_group": best.get("cluster_dominant_product_group"),
            "dominant_region": best.get("cluster_dominant_region"),
            "average_price": best.get("cluster_average_price"),
            "average_margin": best.get("cluster_average_margin"),
        },
        "baseline_model_predictions": baseline.to_dict(orient="records") if not baseline.empty else [],
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
    base_warning = "Bu yorum gerçek kazanma olasılığı değildir; geçmiş kazanılmış ihale profiline uyumu açıklar."
    if "fiyat" in q or "koridor" in q:
        answer = (
            f"Fiyat koridoru benzer kazanılmış ihalelerden gelir. Bu analizde düşük fiyat "
            f"{format_try(corridor.get('predicted_low_price'))}, orta fiyat {format_try(corridor.get('predicted_mid_price'))}, "
            f"yüksek fiyat {format_try(corridor.get('predicted_high_price'))}. Koridor, agresif/dengeli/muhafazakar "
            f"senaryoları karşılaştırmak için kullanılır. Baz model sinyalleri: {advisor.get('learner_signals', {}).get('regression_models', '-')}"
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
            "Profil uyum skoru yaklaşık %65 geçmiş dağılımda normal görünme ve %35 başarı grubuna yakınlık sinyalinden oluşur."
        )
    elif "neden" in q or "öner" in q or "senaryo" in q:
        answer = (
            f"Seçilen senaryo {format_try(context.get('proposed_unit_price'))} birim fiyatla "
            f"{format_pct(context.get('computed_margin_pct'))} beklenen karlılık oranı üretir. "
            f"Senaryo skoru {context.get('scenario_score', 0):.1f}/100; profil uyumu, fiyat bandı uyumu, karlılık, "
            "model güveni ve risk cezası birlikte hesaplanır."
        )
    else:
        answer = (
            f"{advisor.get('decision_summary', '')} {advisor.get('recommended_action', '')} "
            f"{advisor.get('pricing_interpretation', '')} {advisor.get('margin_risk', '')}"
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
                    "Sen Türkçe yanıt veren, yalnızca verilen MODEL_CONTEXT_JSON içeriğini yorumlayan "
                    "ihale karar destek analistisin. Hesap yapma, sayı uydurma, eksik bilgiyi tamamlama. "
                    "Kullanıcının sorusunu özellikle cevapla ama yanıtı mutlaka verilen emsal ihale, fiyat "
                    "koridoru, Linear Regression Baseline, Random Forest Baseline, medyan baz, Cost Plus Margin, "
                    "K-Means başarı grubu, Isolation Forest sıra dışılık kontrolü, senaryo skorları, risk "
                    "bayrakları ve sızıntı kontrolü bağlamıyla sınırla. Veri setinde sadece kazanılmış "
                    "ihaleler olduğunu açıkla; kaybedilmiş ihale olmadığı için gerçek kazanma olasılığı, "
                    "rakip bazlı kazanma tahmini veya kesin teklif kararı iddia etme. Kullanıcı gerçek "
                    "sonucu açmadıysa gizli gerçek fiyat veya karlılık sonucunu kullanma. Yanıt geçerli "
                    "JSON olmalı; markdown, tablo veya serbest metin verme."
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
    grounding = validate_grounding(parsed, context)
    forbidden = detect_forbidden_claims(" ".join(str(value) for value in parsed.values()))
    if not validation["valid"] or not grounding["grounded"] or forbidden["forbidden_claims_detected"]:
        return None
    return parsed


def advisor_payload_to_chat_text(payload: dict[str, Any]) -> str:
    learner = payload.get("learner_signals", {}) if isinstance(payload.get("learner_signals"), dict) else {}
    parts = [
        ("Yönetici özeti", payload.get("decision_summary")),
        ("Veri durumu", payload.get("data_situation")),
        ("Önerilen aksiyon", payload.get("recommended_action")),
        ("Profil uyum yorumu", payload.get("pwin_interpretation")),
        ("Fiyat yorumu", payload.get("pricing_interpretation")),
        ("Karlılık ve risk", payload.get("margin_risk")),
        ("Isolation Forest", learner.get("isolation_forest")),
        ("K-Means", learner.get("kmeans")),
        ("Regresyon modelleri", learner.get("regression_models")),
    ]
    text = "\n\n".join(f"**{title}:** {value}" for title, value in parts if value)
    for title, key in [("Kanıtlar", "supporting_evidence"), ("Riskler", "risks"), ("Sonraki adımlar", "next_actions")]:
        values = payload.get(key)
        if isinstance(values, list) and values:
            text += "\n\n" + f"**{title}:**\n" + "\n".join(f"- {item}" for item in values[:4])
    return text


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
                <span class="status-badge status-danger">Gerçek kazanma olasılığı değildir</span>
            </div>
            <div class="sidebar-note">
                Fiyat koridoru, profil uyumu, senaryo skoru ve güvenli AI yorumu için karar destek kokpiti.
            </div>
            """,
            unsafe_allow_html=True,
        )
        sidebar_data = load_active_data()
        st.download_button(
            "Veriyi indir",
            dataframe_to_csv_bytes(sidebar_data),
            "tender_iq_veri_seti.csv",
            "text/csv",
            help="Uygulamada kullanılan aktif kazanılmış ihale veri setini CSV olarak indirir.",
            width="stretch",
        )
        page = st.radio("Sayfa", PAGE_NAMES, label_visibility="collapsed")
    return page


def render_home() -> None:
    weights = load_scenario_weights()
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

    section_header(
        "Bu ürün ne sağlar?",
        "Tender IQ, teklif kararını tek bir sayıya indirgemez; fiyat, benzerlik, karlılık ve risk sinyallerini birlikte okunabilir hale getirir.",
    )
    render_premium_grid(
        [
            {"icon": "01", "title": "Emsal İhale Bulur", "body": "Seçili ihaleye benzeyen geçmiş kazanılmış işleri listeler.", "pill": "Retrieval", "color": "blue"},
            {"icon": "02", "title": "Profil Uyumunu Gösterir", "body": "İhalenin geçmiş başarı profiline ne kadar yakın olduğunu açıklar.", "pill": "Profil", "color": "purple"},
            {"icon": "03", "title": "Fiyat Koridoru Üretir", "body": "Düşük, dengeli ve yüksek fiyat seviyelerini emsallerden çıkarır.", "pill": "Fiyat", "color": "mint"},
            {"icon": "04", "title": "Teklif Senaryolarını Karşılaştırır", "body": "Agresif, dengeli ve muhafazakar teklif seçeneklerini yan yana verir.", "pill": "Senaryo", "color": "amber"},
            {"icon": "05", "title": "Risk Sinyallerini Gösterir", "body": "Düşük güven, sıra dışı profil veya kural sorunlarını görünür kılar.", "pill": "Kontrol", "color": "cyan"},
            {"icon": "06", "title": "AI Danışman ile Açıklar", "body": "Model çıktısını yönetici dilinde ve güvenli bağlamla yorumlar.", "pill": "Danışman", "color": "blue"},
        ],
        columns=3,
        size="large-size",
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    section_header("Skorlar nasıl okunur?", "Ana skorlar gerçek kazanma olasılığı değildir; geçmiş kazanılmış veriye göre uyum ve tutarlılık göstergesidir.")
    render_premium_grid(
        [
            {"icon": f"%{weights.get('won_profile_fit_score', 0) * 100:.0f}", "title": "Profil Uyumu", "body": "İhalenin geçmişte kazanılmış işlere benzerliğini temsil eder.", "pill": "Senaryo skor etkisi", "color": "purple"},
            {"icon": f"%{weights.get('price_band_fit_score', 0) * 100:.0f}", "title": "Fiyat Bandı Uyumu", "body": "Önerilen fiyatın emsal fiyat koridoruna ne kadar oturduğunu gösterir.", "pill": "Senaryo skor etkisi", "color": "mint"},
            {"icon": f"%{weights.get('margin_score', 0) * 100:.0f}", "title": "Karlılık Skoru", "body": "Önerilen fiyat ile tahmini maliyet arasındaki beklenen karlılığı ölçer.", "pill": "Senaryo skor etkisi", "color": "amber"},
        ],
        columns=3,
    )


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
        ("Senaryo ve karlılık analizi", "Aday teklif fiyatlarının karlılık, katkı ve risk etkisi karşılaştırılır."),
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
        ("Zorunlu kolonlar", "Tamam" if not schema_result.missing_columns else "Eksik", "Ürün, kurum, miktar, tarih, fiyat ve karlılık alanlarının durumunu gösterir.", "good" if not schema_result.missing_columns else "bad"),
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
            ["estimated_unit_cost", "Tahmini maliyet", "Karlılık oranı ve katkı senaryolarını hesaplamak için kullanılır."],
            [CANONICAL_PRICE_COLUMN, "Kazanılmış birim fiyat", "Fiyat koridorunu eğitmek ve backtestte kıyaslamak için kullanılır."],
            [CANONICAL_MARGIN_COLUMN, "Kazanılmış karlılık oranı", "Geçmiş karlılık davranışını ve senaryo sağlığını değerlendirmek için kullanılır."],
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
                ["Kazanılmış karlılık oranı", CANONICAL_MARGIN_COLUMN, "Bulundu" if CANONICAL_MARGIN_COLUMN in data else "Eksik"],
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
            "Yeni CSV yüklemek, demo veri seti yerine kurumunuza ait tarihsel kazanılmış ihale verisini kullanmak içindir. Dosyada ürün, kurum, bölge, miktar, tarih, tahmini maliyet, kazanılmış fiyat ve karlılık oranı alanları bulunmalıdır.",
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
    info_callout(PWIN_PROXY_EXPLANATION, "Profil uyum göstergesi ne anlatır?")
    with st.expander("Aktif config", expanded=False):
        summary = active_config_summary()
        config_rows = [
            ("App config", summary["app_config"]),
            ("Hard constraints", summary["hard_constraints"]),
            ("Soft penalties", summary["soft_penalties"]),
            ("Top-K emsal sayısı", str(summary["default_top_k"])),
            ("Isolation Forest hassasiyet ayarı", format_pct(summary["isolation_contamination"] * 100)),
            ("Agresif anomaly uyarı eşiği", format_pct(summary["aggressive_anomaly_rate_threshold"] * 100)),
            ("Senaryo skor ağırlıkları", json.dumps(summary["scenario_weights"], ensure_ascii=False)),
            ("Sert kural config anahtarları", ", ".join(summary["hard_constraint_keys"])),
            ("Soft penalty config anahtarları", ", ".join(summary["soft_penalty_keys"])),
        ]
        st.dataframe(pd.DataFrame(config_rows, columns=["Config", "Aktif değer"]), hide_index=True, width="stretch")
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    section_header("Nasıl çalışır?", "Teknik bileşenler iş akışına göre beş adımda okunur.")
    render_premium_grid(
        [
            {"icon": "01", "title": "İhale Profili Oluşturulur", "body": "Ürün, kurum, bölge, miktar ve ihale tipi tek profilde toplanır.", "pill": "Girdi", "color": "blue"},
            {"icon": "02", "title": "Emsal İhaleler Bulunur", "body": "Geçmiş kazanılmış ihaleler arasından en yakın emsaller seçilir.", "pill": "Emsal", "color": "purple"},
            {"icon": "03", "title": "Profil Uyumu Kontrol Edilir", "body": "Başarı grubu ve sıra dışılık sinyali birlikte değerlendirilir.", "pill": "Profil", "color": "mint"},
            {"icon": "04", "title": "Fiyat Koridoru Üretilir", "body": "Emsal fiyatlardan düşük, dengeli ve yüksek seviye çıkarılır.", "pill": "Fiyat", "color": "amber"},
            {"icon": "05", "title": "Senaryolar Skorlanır", "body": "Fiyat, profil, karlılık, güven ve risk birlikte puanlanır.", "pill": "Skor", "color": "cyan"},
        ],
        columns=3,
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
                "Sistem gerçek olasılık üretmez. Yeni ihalenin geçmişte kazanılmış işlere ne kadar benzediğini ve fiyat/karlılık/risk dengesinin geçmiş profile uyup uymadığını gösterir.",
                "02",
            )
        with c3:
            glass_card(
                "Kazanılmış Profil Uyum Skoru",
                "Ürün, kurum, bölge, miktar, teslim süresi, fiyat bandı, beklenen karlılık, risk ve model güveni birlikte okunur. Düşük skor çoğu zaman manuel inceleme sinyalidir.",
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
                ("02", "K-Means", "Ne yapar: Kazanılmış ihaleleri ihale anında bilinen profil özelliklerine göre gruplar. Neden var: Tek tek ihale yerine profil segmenti görmeyi sağlar. Katkı: Başarı grubu ve profil yorumunu destekler; yeni/test ihalesi atamasında gerçek fiyat veya senaryo fiyatı kullanılmaz.", "purple"),
                ("03", "Isolation Forest", "Ne yapar: Yeni ihalenin geçmiş profile normal mi sıra dışı mı uyduğunu kontrol eder. Neden var: Aykırı durumları saklamaz. Katkı: Risk ve manuel inceleme sinyali üretir.", "amber"),
                ("04", "Price Corridor Engine", "Ne yapar: Emsal kazanılmış ihalelerden düşük, orta ve yüksek fiyat bandı çıkarır. Neden var: Tek nokta fiyat yerine karar aralığı verir. Katkı: Senaryo fiyatlarını besler.", "green"),
                ("05", "Scenario Scoring", "Ne yapar: Fiyat, karlılık, profil uyumu, güven ve risk cezasını tek karar destek skorunda birleştirir. Neden var: Alternatif teklifleri kıyaslanabilir hale getirir. Katkı: Sıralı senaryo önerisi üretir.", "cyan"),
                ("06", "Model Confidence / Risk", "Ne yapar: Benzer ihale sayısı, veri kalitesi, band genişliği ve aykırılık sinyallerini birlikte okur. Neden var: Skorun ne kadar güvenle okunacağını gösterir. Katkı: AI Danışman ve manuel inceleme kararını destekler.", "blue"),
                ("07", "Linear Regression Baseline", "Ne yapar: Ürün grubu, bölge, ihale tipi, miktar, teslim süresi ve tahmini rakip sayısı gibi alanlardan beklenen fiyat için doğrusal referans üretir. Neden var: Emsal tabanlı fiyat koridorunu basit ve açıklanabilir bir fiyat tahminiyle kıyaslamak için kullanılır. Katkı: Backtestte koridor yaklaşımının basit doğrusal modele göre ne kadar tutarlı olduğunu gösterir.", "green"),
                ("08", "Random Forest / Ağaç Tabanlı Baseline", "Ne yapar: Doğrusal olmayan fiyat ilişkilerini yakalayabilen ağaç tabanlı referans modelini temsil eder. Neden var: Miktar, bölge ve ürün grubunun fiyat üzerindeki doğrusal olmayan etkilerini kontrol etmek için kullanılır. Katkı: Gerçek kazanma olasılığı üretmez; fiyat koridorunun regresyon bazlı tahminlerle tutarlılığını değerlendirmek için metodolojik karşılaştırma sağlar.", "amber"),
            ]
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        with st.expander("Sıra dışı durum örnekleri", expanded=False):
            st.markdown("- çok yüksek veya çok düşük miktar\n- alışılmadık ürün-kurum kombinasyonu\n- çok kısa teslim süresi\n- çok yüksek veya çok düşük fiyat\n- düşük benzer ihale sayısı")
        with st.expander("Fiyat koridoru nasıl oluşuyor?", expanded=True):
            info_callout(
                "Sistem seçili ihaleye en çok benzeyen kazanılmış ihaleleri bulur. Ana fiyat koridoru bu emsal ihalelerdeki normalize fiyatların alt, orta ve üst yüzdeliklerinden üretilir. Lineer regresyon ve Random Forest/ağaç tabanlı baseline ise bu koridorun basit ve daha esnek fiyat tahminleriyle kıyaslanması için metodolojik kontrol katmanı olarak anlatılır; gerçek kazanma olasılığı üretmez. Koridor çok genişse tek başına güçlü kanıt sayılmaz, bu yüzden backtestte band genişliği de ölçülür.",
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
                ("💹", "Karlılık Skoru", "Beklenen karlılık ve katkı karı sağlığı.", "purple"),
                ("🛡️", "Model Güveni", "Yeterli benzer ihale ve veri kalitesi olup olmadığı.", "cyan"),
                ("⚠️", "Risk Cezası", "Sıra dışı durumlar, düşük benzerlik, düşük güven ve kısıt ihlalleri.", "amber"),
            ]
        )
        info_callout("Not: Skor gerçek kazanma olasılığı değil, geçmiş kazanılmış ihale profiline uyum göstergesidir.")

    with tabs[4]:
        section_header(
            "Backtest ve Metrikler",
            "Test yılındaki geçmiş kazanılmış ihaleler, sonuç fiyatı ve gerçek karlılık oranı gizlenerek sisteme o gün yeni gelmiş canlı ihale gibi verilir; sistem önce emsal, profil, koridor ve senaryo çıktısı üretir, sonra gerçek sonuç açılarak bu çıktılarla karşılaştırılır.",
            "Ölçüm",
        )
        info_callout(
            "Backtest’in amacı kazanma/kaybetme tahmini doğruluğunu ölçmek değildir. Amaç, sistemin geçmişte kazanılmış ama sonucu gizlenmiş ihaleleri geçmiş başarı profillerine doğru yerleştirip yerleştiremediğini ve fiyat/senaryo önerilerinin tarihsel sonuçlarla ne kadar tutarlı olduğunu ölçmektir.",
            "Backtest amacı:",
        )
        render_method_grid(
            [
                ("Zaman bazlı ayrım", "Model geçmişi bilir, geleceği bilmez. Random split canlı karar anını yeterince temsil etmez."),
                ("Sonuç gizleme", "Kazanılmış fiyat ve karlılık oranı reveal adımına kadar retrieval, scorer, optimizer ve advisor katmanlarından maskelenir."),
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
                    ["Ortalama yüzde fiyat hatası", "Önerilen orta fiyatın gerçek kazanılmış fiyattan ortalama yüzde sapması."],
                    ["Ortalama fiyat aralığı genişliği", "Düşük ve yüksek fiyat önerisi arasındaki ortalama fark."],
                    ["Band kalite skoru", "Fiyat bandının hem gerçek fiyatı kapsamasını hem de çok geniş olmamasını birlikte değerlendirir."],
                    ["Gerçek Kazanılmış Senaryo Sıralaması", "Tarihsel gerçek konfigürasyon aday senaryolar arasında ne kadar üstte kaldı?"],
                    ["Geçmiş profile uygun görülen test ihalesi oranı", "Kazanılmış test kayıtlarının ne kadarının geçmiş başarı profiline normal uyduğu."],
                    ["Synthetic outlier detection rate", "Sentetik sıra dışı örneklerin model tarafından riskli veya aykırı görülme oranı."],
                    ["Yasak iddia üretme oranı", "AI Danışman garanti, kesin sonuç veya gerçek kazanma olasılığı iddiası üretiyor mu? Hedef sıfırdır."],
                ],
                columns=["Metrik", "Ne anlatır?"],
            )
            st.dataframe(metrics_table, hide_index=True, width="stretch")
        with st.expander("Sızıntı kontrolü nedir?", expanded=False):
            st.markdown(
                """
                Sızıntı, test sırasında modelin normalde bilmemesi gereken gerçek sonucu önceden görmesidir.
                Test ihalesinin kazanılmış fiyatı, final tutarı veya gerçek karlılık oranı modele verilirse test güvenilir olmaz.
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
                ("💬", "Açıklama Üretir", "Profil uyumu, fiyat koridoru, karlılık, risk ve benzer ihaleleri anlaşılır hale getirir.", "blue"),
                ("🧱", "Guardrail Uygular", "Garanti, kesin sonuç veya gerçek kazanma olasılığı iddiası üretirse çıktı reddedilir.", "amber"),
                ("🔒", "Reveal Kuralına Uyar", "Gerçek sonuç açılmadıysa kazanılmış fiyat veya gerçek karlılık oranını kullanamaz.", "purple"),
                ("🧰", "Fallback Çalışır", "LLM sağlayıcısı yoksa deterministik danışman aynı sohbet akışında yanıt verir.", "green"),
            ]
        )
def render_test_simulator() -> None:
    page_header(
        "Test için İhale Seç",
        "Geçmişte kazanılmış bir ihale, gerçek sonucu gizlenmiş şekilde yeni gelen ihale gibi seçilir. Sistem sonuç açılmadan önce emsal, profil, fiyat ve teklif senaryosu üretir.",
        "Test seçimi",
    )
    info_callout(
        "Skor gerçek kazanma olasılığı değil, geçmiş kazanılmış ihale profiline uyum göstergesidir. Gerçek kazanılmış fiyat ve karlılık oranı, karşılaştırma adımına kadar gizli kalır.",
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

    section_header("Seçili ihale", "Bu bilgiler canlı ihale girdisi gibi kullanılır; gerçek sonuç alanları maskelidir.")
    selected_body = (
        f"İhale ID: {selected} | Ürün grubu: {masked.get('product_group', '-')} | "
        f"Bölge: {masked.get('region', '-')} | Kurum: {masked.get('buyer_institution', '-')}"
    )
    glass_card("Test girdisi", selected_body, "Gerçek sonuç gizli", badge("Sızıntı yok" if audit["audit_status"] == "pass" else "Sızıntı uyarısı", "good" if audit["audit_status"] == "pass" else "bad"))
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    section_header("Bu test akışı ne üretir?", "Sonuçlar ayrı sayfalarda, aynı simülasyon çıktısı üzerinden gösterilir.")
    test_cards = [
        ("Emsal ihale analizi", "Geçmişte kazanılmış en benzer ihaleleri ve eşleşme gücünü gösterir."),
        ("Profil uyum analizi", "Başarı grubu ve sıra dışılık kontrolünü tek sayfada açıklar."),
        ("Fiyat koridoru", "Low / mid / high fiyatları ve baz model tahminlerini karşılaştırır."),
        ("Teklif senaryoları", "Agresif, dengeli ve muhafazakar teklif seçeneklerini skorlar."),
        ("Gerçek sonuçla karşılaştırma", "Sonuç açıldıktan sonra gerçek fiyatı, profili ve senaryo sırasını kıyaslar."),
    ]
    for start_idx in range(0, len(test_cards), 3):
        cols = st.columns(3, gap="medium")
        for offset, (title, body) in enumerate(test_cards[start_idx : start_idx + 3]):
            with cols[offset]:
                render_small_card(title, body)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Canlı ihale girdileri", "Bu alanlar simülasyon için düzenlenebilir; gerçek kazanılmış fiyat ve karlılık oranı görünmez.", "Kontrol paneli")
    with st.container():
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
            st.success("Simülasyon tamamlandı. Sonuçları Emsal İhale Analizi sayfasından başlayarak inceleyebilirsiniz.")

    with st.expander("Maskelenmiş girdi alanları", expanded=False):
        safe_preview = pd.DataFrame([masked]).T.reset_index()
        safe_preview.columns = ["Alan", "Değer"]
        safe_preview["Değer"] = safe_preview["Değer"].astype(str)
        st.dataframe(safe_preview, hide_index=True, width="stretch")


def scenario_name(index: int) -> str:
    names = ["Agresif Senaryo", "Dengeli Senaryo", "Muhafazakâr Senaryo"]
    return names[index] if index < len(names) else f"Alternatif Senaryo {index + 1}"


def render_profile_fit_analysis() -> None:
    page_header(
        "Profil Uyum Analizi",
        "Bu sayfa, seçili ihalenin geçmişte kazanılmış ihale profillerine ne kadar benzediğini gösterir. Sistem; benzer ihaleleri, geçmiş başarı gruplarını ve sıra dışılık kontrolünü birlikte değerlendirir.",
        "Profil uyumu",
    )
    result = ensure_scenario_result()
    if not result:
        require_test_tender_message()
        return
    weights = load_scenario_weights()
    best = result["scenarios"].iloc[0].to_dict()
    quality = retrieval_quality_from_result(result, current_tender() or {})
    profile_label, profile_status = profile_status_from_best(best)
    anomaly_rate = float(best.get("training_anomaly_rate", 0.0) or 0.0)
    inlier_rate = float(best.get("training_inlier_rate", 0.0) or 0.0)
    segment_rate = best.get("segment_anomaly_rate")

    section_header("Genel Uyum Özeti", "Üç temel soruyu yanıtlar: hangi geçmiş profile benziyor, normal mi görünüyor, genel skor ne söylüyor?")
    render_premium_grid(
        [
            {
                "icon": "Skor",
                "title": "Kazanılmış Profil Uyum Skoru",
                "value": format_score(best.get("won_profile_fit_score")),
                "body": "Isolation Forest geçmişte kazanılmış dağılıma alışıldık uyumu, K-Means başarı grubuna yakınlığı ölçer.",
                "pill": fit_level(best.get("won_profile_fit_score")),
                "color": "blue",
            },
            {
                "icon": "Grup",
                "title": "Geçmiş Başarı Grubu",
                "value": str(best.get("cluster_id", "Hesaplanamadı")),
                "body": str(best.get("cluster_name", "Geçmiş başarı grubu")),
                "pill": "K-Means",
                "color": "purple",
            },
            {
                "icon": "Kontrol",
                "title": "Sıra Dışılık Kontrolü (Isolation Forest)",
                "value": profile_label,
                "body": "Geçmiş kazanılmış kayıtlar içinde normal mi, yoksa manuel inceleme gerektirecek kadar farklı mı?",
                "pill": "Isolation Forest",
                "color": "amber",
            },
            {
                "icon": "Emsal",
                "title": "Emsal Benzerlik Gücü",
                "value": f"{result.get('top10_avg_similarity', 0):.2f}",
                "body": "En yakın emsallerin seçili ihaleye ortalama yakınlığını gösterir.",
                "pill": "0-1 yakınlık",
                "color": "mint",
            },
        ],
        columns=2,
        size="metric-size",
    )
    info_callout(
        "Bu skor iki modelin birleşimidir. Isolation Forest, seçili ihalenin geçmişte kazanılmış işler arasında ne kadar alışıldık göründüğünü ölçer ve yaklaşık %65 ağırlık taşır. K-Means, ihalenin hangi geçmiş başarı grubuna ne kadar yakın olduğunu ölçer ve yaklaşık %35 ağırlık taşır. Skor fiyat kararı veya gerçek kazanma olasılığı değildir; profilin geçmiş başarı örneklerine ne kadar tanıdık göründüğünü anlatır.",
        "Profil uyum skoru nasıl hesaplanır?",
    )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header(
        "Geçmiş Başarı Grubu",
        "K-Means, geçmişte kazanılmış ihaleleri benzer özelliklerine göre gruplar. Bu sonuç, seçili ihalenin geçmişte hangi tip kazanılmış ihalelere benzediğini gösterir.",
    )
    info_callout(
        "Cluster açıklamasında tarihsel gerçekleşmiş fiyat ve karlılık özetleri gösterilir. Ancak seçili/test ihalesini cluster’a yerleştirirken gerçek kazanılmış fiyat, gerçek karlılık veya önerilen senaryo fiyatı kullanılmaz; atama yalnızca ihale anında bilinen profil alanlarıyla yapılır.",
        "K-Means hangi bilgileri kullanır?",
    )
    left, right = st.columns([1.15, 0.85], gap="medium")
    with left:
        render_kv_card(
            "Başarı grubu detayı",
            [
                ("Başarı grubu adı", str(best.get("cluster_name", "Hesaplanamadı"))),
                ("Bu gruptaki ihale sayısı", format_int(best.get("cluster_count"))),
                ("Baskın ürün grubu", str(best.get("cluster_dominant_product_group", "Hesaplanamadı"))),
                ("Baskın kurum tipi", str(best.get("cluster_dominant_institution_type", "Hesaplanamadı"))),
                ("Baskın bölge", str(best.get("cluster_dominant_region", "Hesaplanamadı"))),
                ("Ortalama miktar", format_int(best.get("cluster_average_quantity"))),
                ("Ortalama fiyat", format_try(best.get("cluster_average_price"))),
                ("Ortalama karlılık oranı", format_pct(best.get("cluster_average_margin"))),
            ],
            profile_business_comment(best),
        )
    with right:
        st.plotly_chart(build_gauge(float(best.get("won_profile_fit_score", 0)), "Profil uyum skoru"), use_container_width=True)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header(
        "Sıra Dışılık Kontrolü (Isolation Forest)",
        "Bu kontrol, seçili ihalenin geçmişte kazanılmış işlere ne kadar alışıldık göründüğünü anlatır. Sıra dışı sonuç kötü ihale anlamına gelmez; yalnızca geçmiş örneklerden farklılaştığı için manuel inceleme gerekebileceğini söyler.",
    )
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Durum", profile_label, isolation_business_comment(best), "amber" if profile_status == "warn" else "green")
    with c2:
        metric_card("Normal görülen kayıt oranı", format_pct(inlier_rate * 100), "Modelin eğitimde geçmiş başarı profiline uygun bulduğu kazanılmış kayıt oranı")
    with c3:
        metric_card("Manuel inceleme oranı", format_pct(anomaly_rate * 100), "Modelin eğitimde daha az tipik bulduğu kazanılmış kayıt oranı")
    with c4:
        metric_card("Hassasiyet ayarı", format_pct(float(best.get("isolation_contamination", 0)) * 100), "Yaklaşık her 100 kazanılmış kayıttan kaçının daha az tipik ayrılacağını belirleyen ayar")

    info_callout(
        "Bu veri setindeki tüm kayıtlar kazanılmış ihalelerden oluşur. Bu nedenle Isolation Forest’ın sıra dışı dediği bir kayıt, kaybedilecek ihale anlamına gelmez. Sadece geçmiş kazanılmış ihaleler arasında daha az tipik bir örnek olduğunu gösterir.",
        "Sıra dışılık nasıl okunmalı?",
    )
    if segment_rate is not None and not pd.isna(segment_rate):
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        metric_card("Ürün grubunda manuel inceleme oranı", format_pct(float(segment_rate) * 100), "Seçili ürün grubunda geçmiş profile göre daha az tipik görülen kayıt oranı", "purple")
    if anomaly_rate >= 0.25:
        st.warning("Eğer kazanılmış test ihalelerinin büyük kısmı manuel inceleme gerektiriyor görünüyorsa, model fazla hassas olabilir ve hassasiyet ayarı gözden geçirilmelidir.")
    elif float(best.get("isolation_contamination", 0)) >= 0.10:
        st.info("Hassasiyet ayarı orta seviyede. Çok sayıda kazanılmış ihale sıra dışı görünürse ayar düşürülebilir.")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Emsal sinyali", "Profil uyumu emsal ihale kalitesiyle birlikte okunur.")
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        metric_card("Ürün grubu eşleşmesi", format_pct(quality.get("product_group_match_rate", 0) * 100), "Benzer ihalelerin ne kadarı aynı ürün grubunda?")
    with e2:
        metric_card("Bölge eşleşmesi", format_pct(quality.get("region_match_rate", 0) * 100), "Benzer ihalelerin ne kadarı aynı bölgede?")
    with e3:
        metric_card("Miktar bandı eşleşmesi", format_pct(quality.get("quantity_band_match_rate", 0) * 100), "Benzer ihalelerin ne kadarı yakın miktar ölçeğinde?")


def render_price_corridor_models() -> None:
    page_header(
        "Fiyat Koridoru ve Model Karşılaştırması",
        "Bu sayfa, seçili ihale için farklı yöntemlerin ürettiği fiyat tahminlerini ve benzer ihalelerden oluşan fiyat koridorunu gösterir.",
        "Fiyat",
    )
    result = ensure_scenario_result()
    tender = current_tender()
    if not result or not tender:
        require_test_tender_message()
        return
    corridor = result["corridor"]
    mid_ref = max(float(corridor["predicted_mid_price"]), 0.01)
    low_ratio = float(corridor["predicted_low_price"]) / mid_ref
    high_ratio = float(corridor["predicted_high_price"]) / mid_ref
    baseline = predict_baseline_prices(get_history_frame(), tender)
    baseline_map = {row["method"]: row for _, row in baseline.iterrows()}
    rows = [
        {
            "Yöntem": "Benzerlik Tabanlı Fiyat Koridoru",
            "Düşük fiyat / low": corridor["predicted_low_price"],
            "Orta fiyat / mid": corridor["predicted_mid_price"],
            "Yüksek fiyat / high": corridor["predicted_high_price"],
            "Tahmin fiyatı": corridor["predicted_mid_price"],
            "Açıklama": "Top-K benzer kazanılmış ihalelerden fiyat bandı üretir.",
            "Güven seviyesi": "Yüksek" if result["model_confidence_score"] >= 70 else "Orta" if result["model_confidence_score"] >= 45 else "Düşük",
        }
    ]
    for method in ["Linear Regression Baseline", "Random Forest Baseline", "Median Baseline", "Cost Plus Margin"]:
        item = baseline_map.get(method)
        if item is None:
            rows.append(
                {
                    "Yöntem": method,
                    "Düşük fiyat / low": None,
                    "Orta fiyat / mid": None,
                    "Yüksek fiyat / high": None,
                    "Tahmin fiyatı": None,
                    "Açıklama": "Henüz aktif değil.",
                    "Güven seviyesi": "Aktif değil",
                }
            )
            continue
        prediction = max(float(item["prediction"]), 0.01)
        rows.append(
            {
                "Yöntem": method,
                "Düşük fiyat / low": max(0.01, prediction * low_ratio),
                "Orta fiyat / mid": prediction,
                "Yüksek fiyat / high": max(0.01, prediction * high_ratio),
                "Tahmin fiyatı": prediction,
                "Açıklama": str(item["description"]),
                "Güven seviyesi": str(item["confidence"]),
            }
        )
    price_table = pd.DataFrame(rows)
    numeric_rows = price_table.dropna(subset=["Düşük fiyat / low", "Orta fiyat / mid", "Yüksek fiyat / high"])
    avg_low = float(numeric_rows["Düşük fiyat / low"].mean())
    avg_mid = float(numeric_rows["Orta fiyat / mid"].mean())
    avg_high = float(numeric_rows["Yüksek fiyat / high"].mean())
    price_display = price_table.copy()
    for column in ["Düşük fiyat / low", "Orta fiyat / mid", "Yüksek fiyat / high"]:
        price_display[column] = price_display[column].apply(lambda value: format_optional_try(value, "Henüz aktif değil"))
    price_display["Tahmin fiyatı"] = price_display["Tahmin fiyatı"].apply(lambda value: format_optional_try(value, "Henüz aktif değil"))

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Ortalama alt fiyat", format_try(avg_low), "Tüm aktif yöntemlerin düşük fiyat ortalaması", "blue")
    with c2:
        metric_card("Ortalama orta fiyat", format_try(avg_mid), "Tüm aktif yöntemlerin dengeli fiyat ortalaması", "green")
    with c3:
        metric_card("Ortalama üst fiyat", format_try(avg_high), "Tüm aktif yöntemlerin yüksek fiyat ortalaması", "amber")
    with c4:
        metric_card("Model güveni", format_score(result["model_confidence_score"]), "Benzer ihale sayısı ve benzerlik gücü", "purple")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header(
        "Fiyat modeli karşılaştırması",
        "Her yöntem için düşük, orta ve yüksek fiyat gösterilir. Linear Regression Baseline, Random Forest Baseline, Median Baseline ve Cost Plus Margin modelleri tek fiyat üretir; düşük/yüksek aralık bu tahminin etrafında emsal koridor genişliğiyle türetilir.",
    )
    model_cards = []
    model_labels = {
        "Benzerlik Tabanlı Fiyat Koridoru": "Benzerlik Tabanlı Koridor",
        "Random Forest Baseline": "Random Forest / Ağaç Tabanlı Baseline",
        "Cost Plus Margin": "Cost Plus Margin",
    }
    colors = ["blue", "purple", "mint", "amber", "cyan"]
    for idx, row in price_table.iterrows():
        model_cards.append(
            {
                "icon": f"0{idx + 1}",
                "title": model_labels.get(str(row["Yöntem"]), str(row["Yöntem"])),
                "body": str(row["Açıklama"]),
                "pill": str(row["Güven seviyesi"]),
                "color": colors[idx % len(colors)],
                "lines": [
                    ("Düşük", format_optional_try(row["Düşük fiyat / low"], "Aktif değil")),
                    ("Orta", format_optional_try(row["Orta fiyat / mid"], "Aktif değil")),
                    ("Yüksek", format_optional_try(row["Yüksek fiyat / high"], "Aktif değil")),
                ],
            }
        )
    render_premium_grid(model_cards, columns=3, size="large-size")
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.dataframe(
        price_display,
        hide_index=True,
        width="stretch",
    )
    info_callout(
        "Fiyat koridoru tek başına karar değildir. Bu fiyatlar; emsal ihale analizi, profil uyumu, karlılık beklentisi ve risk göstergeleriyle birlikte değerlendirilmelidir.",
        "Karar notu:",
    )


def render_scenario_analysis() -> None:
    page_header(
        "Teklif Senaryoları",
        "Bu sayfa fiyat koridorundan türetilen agresif, dengeli ve muhafazakâr teklif stratejilerini karşılaştırır. Amaç tek bir doğru fiyat vermek değil; fiyat, karlılık, katkı kârı ve risk etkisini yan yana göstermektir.",
        "Teklif",
    )
    result = ensure_scenario_result()
    if not result:
        require_test_tender_message()
        return
    scenarios = result["scenarios"].copy()
    st.session_state.best_scenario = scenarios.iloc[0].to_dict()
    tender = current_tender() or {}
    corridor = result["corridor"]

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Model Güven Skoru", f"{result['model_confidence_score']:.1f}/100", "Benzer ihale sayısı ve uyumu", "blue", "🛡️")
    with c2:
        metric_card("Orta Fiyat", format_try(result["corridor"]["predicted_mid_price"]), "Fiyat koridoru merkezi", "green", "🎯")
    with c3:
        metric_card("Temel kurala uygun senaryo", format_int(scenarios["hard_constraints_valid"].sum()), "Minimum karlılık eşiğini ve temel fiyat kurallarını geçen seçenek sayısı", "amber", "✅")

    info_callout(
        (
            "Senaryo skoru gerçek kazanma olasılığı değildir. Aktif ağırlıklar: "
            f"%{weights.get('won_profile_fit_score', 0) * 100:.0f} profil uyumu, "
            f"%{weights.get('price_band_fit_score', 0) * 100:.0f} fiyat bandı uyumu, "
            f"%{weights.get('margin_score', 0) * 100:.0f} beklenen karlılık, "
            f"%{weights.get('model_confidence_score', 0) * 100:.0f} model güveni ve "
            f"-%{abs(weights.get('risk_penalty_score', 0)) * 100:.0f} risk cezası. "
            "Skor, teklif seçeneğinin geçmiş kazanılmış veriyle ne kadar uyumlu olduğunu gösterir."
        ),
        "Senaryo skoru nasıl okunur?",
    )
    render_formula_card()

    section_header("Öne Çıkan Senaryolar", "Her kart teklif fiyatı, karlılık, katkı kârı, risk, kural durumu ve senaryo skorunu birlikte gösterir.", "Senaryo kartları")
    strategy_targets = [
        ("Agresif Senaryo", corridor["predicted_low_price"], "Daha rekabetçi fiyat verir. Kazanım profiline daha yakın olabilir ancak karlılığı düşürebilir."),
        ("Dengeli Senaryo", corridor["predicted_mid_price"], "Fiyat ve karlılık arasında orta yol arar."),
        ("Muhafazakâr Senaryo", corridor["predicted_high_price"], "Karlılığı korumaya daha yakındır ancak fiyat rekabeti açısından daha riskli olabilir."),
    ]
    selected_indices: set[int] = set()
    selected_cards = []
    for label, target, description in strategy_targets:
        candidates = scenarios.loc[~scenarios.index.isin(selected_indices)].copy()
        if candidates.empty:
            break
        idx = (candidates["proposed_unit_price"].astype(float) - float(target)).abs().idxmin()
        selected_indices.add(int(idx))
        selected_cards.append((label, description, scenarios.loc[idx]))
    scenario_cards_html = ""
    card_colors = ["blue", "green", "amber"]
    for idx, (label, description, scenario) in enumerate(selected_cards):
        status = "good" if bool(scenario["hard_constraints_valid"]) else "bad"
        status_label = "Temel kurallar uygun" if bool(scenario["hard_constraints_valid"]) else "Minimum karlılık kuralı sağlanmıyor"
        total_offer = float(scenario["proposed_unit_price"]) * float(tender.get("quantity", 0))
        contribution = total_offer * float(scenario["computed_margin_pct"]) / 100
        risk_value = max(0.0, 100 - float(scenario["risk_penalty_score"]))
        risk_label = "Düşük" if risk_value >= 75 else "Orta" if risk_value >= 55 else "Yüksek"
        violations = scenario["risk_flags"] if isinstance(scenario["risk_flags"], list) else []
        scenario_cards_html += premium_card_html(
            title=label,
            value=format_try(scenario["proposed_unit_price"]),
            body=description,
            icon=f"0{idx + 1}",
            pill=status_label,
            color=card_colors[idx % len(card_colors)],
            size="scenario-size",
            lines=[
                ("Toplam teklif", format_try(total_offer)),
                ("Karlılık oranı", format_pct(scenario["computed_margin_pct"])),
                ("Katkı kârı", format_try(contribution)),
                ("Risk seviyesi", risk_label),
                ("Senaryo skoru", f"{scenario['scenario_score']:.0f}/100"),
                ("Kural kontrolü", "; ".join(violations) if violations else "Uygun"),
            ],
        )
    st.markdown(f"<div class='card-grid three-col'>{scenario_cards_html}</div>", unsafe_allow_html=True)

    table = scenarios[
        [
            "scenario_id",
            "proposed_unit_price",
            "price_anchor",
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
        "Referans Fiyat",
        "Beklenen Karlılık Oranı",
        "Profil Uyumu",
        "Fiyat Bandı Uyumu",
        "Karlılık Skoru",
        "Risk Cezası",
        "Güven Skoru",
        "Senaryo Skoru",
        "Kural Durumu",
        "Kural / Risk Notları",
    ]
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Önerilen Birim Fiyat": st.column_config.NumberColumn(format="%.2f TL"),
            "Beklenen Karlılık Oranı": st.column_config.NumberColumn(format="%.2f"),
            "Senaryo Skoru": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
        },
    )


def render_reveal_compare() -> None:
    page_header(
        "Gerçek Sonuçla Karşılaştır",
        "Bu sayfa tek seçili test ihalesi içindir. Önce sistemin sonuç gizliyken ürettiği fiyat ve profil çıktıları gösterilir; sonra gerçek kazanılmış fiyat açılarak bu çıktılarla yan yana karşılaştırılır.",
        "Sonuç Açma",
    )
    row = selected_test_tender()
    result = ensure_scenario_result()
    if row is None or not result:
        require_test_tender_message()
        return

    if not st.session_state.get("revealed", False):
        info_callout(
            "Gerçek sonuç açıldığında fiyat bandı, profil uyumu, sıra dışılık yorumu ve senaryo sıralaması karşılaştırılır.",
            "Gerçek sonuç açılınca ne değişir?",
        )
        st.info("Gerçek kazanılmış fiyat ve karlılık oranı henüz gizli. Bu bilgi model, senaryo skoru ve AI danışmana verilmedi.")
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

    info_callout(
        "Bu ekran tek bir ihaleyi inceler. Backtest Sonuçları sayfası ise aynı mantığı test yılındaki tüm ihalelere uygular ve toplu performans oranlarını gösterir.",
        "Bu sayfa Backtest’ten nasıl farklı?",
    )

    section_header("Profil Uyumu Değerlendirmesi", "Seçili ihale geçmiş kazanılmış profile ne kadar yakın görünüyor?")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Profil uyum skoru", format_score(best.get("won_profile_fit_score")), "Seçili ihalenin geçmiş kazanılmış profile yakınlığı")
    with c2:
        metric_card("Uyum yorumu", fit_level(best.get("won_profile_fit_score")), "Düşükse manuel inceleme ihtiyacı artar")
    with c3:
        metric_card("Veri güveni", format_score(best.get("model_confidence_score")), "Benzer ihale sayısı ve emsal gücüne göre okuma kalitesi")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        render_kv_card(
            "Profil Grubu",
            [
                ("Atandığı profil grubu", str(best.get("cluster_name", "Hesaplanamadı"))),
                ("Profil grup ID", str(best.get("cluster_id", "Hesaplanamadı"))),
                ("Geçmiş ihale sayısı", format_int(best.get("cluster_count"))),
                ("Baskın ürün grubu", str(best.get("cluster_dominant_product_group", "Hesaplanamadı"))),
                ("Medyan fiyat", format_try(best.get("cluster_median_price"))),
                ("Medyan karlılık oranı", format_pct(best.get("cluster_median_margin"))),
            ],
            "Bu grup, test ihalesinin geçmişte kazanılmış hangi başarı profiline yakın konumlandığını gösterir.",
        )
    with right:
        render_kv_card(
            "Profil ve Sıra Dışılık Yorumu",
            [
                ("Durum", isolation_status),
                ("Profile uygunluk skoru", format_score(best.get("inlier_score"))),
                ("Yorum", "Model bu kazanılmış test ihalesini geçmiş profile uygun tanımış." if isolation_status == "Geçmiş profile uygun" else "Bu ihale kazanılmış olsa bile geçmiş profilden sıra dışı olabilir; bu kayıp tahmini değildir, manuel inceleme sinyalidir."),
            ],
            "Bu sonuç kazanma olasılığı değildir; geçmiş kazanılmış veri dağılımına uygunluk kontrolüdür.",
        )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Emsal İhale Kalitesi", "Sistemin gerçek sonucu açmadan önce seçtiği emsal havuzunun ne kadar tutarlı olduğunu gösterir.")
    e1, e2, e3, e4 = st.columns(4, gap="medium")
    with e1:
        metric_card("En yakın 10 emsalin benzerliği", f"{top10_avg_similarity:.2f}", "1'e yaklaştıkça emsaller daha güçlü")
    with e2:
        metric_card("İlk 50 emsalin benzerliği", f"{top50_avg_similarity:.2f}", "Geniş emsal havuzunun ortalama yakınlığı")
    with e3:
        metric_card("Ürün Grubu Eşleşmesi", format_pct(quality.get("product_group_match_rate", 0) * 100), "Emsal havuzunda aynı ürün grubu")
    with e4:
        metric_card("Bölge Eşleşmesi", format_pct(quality.get("region_match_rate", 0) * 100), "Emsal havuzunda aynı bölge")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Fiyat Karşılaştırması", "Gerçek kazanılmış fiyat düşük, orta ve yüksek koridorla karşılaştırılır.")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Düşük öneri", format_try(corridor["predicted_low_price"]), "Daha rekabetçi fiyat seviyesi")
    with c2:
        metric_card("Dengeli öneri", format_try(corridor["predicted_mid_price"]), "Emsallerin orta fiyat seviyesi")
    with c3:
        metric_card("Yüksek öneri", format_try(corridor["predicted_high_price"]), "Daha karlı ama rekabet riski yüksek seviye")
    with c4:
        metric_card("Gerçek kazanılmış fiyat", format_try(actual_price), "Sonuç açıldıktan sonra")

    c5, c6, c7 = st.columns(3, gap="medium")
    with c5:
        metric_card("Gerçek fiyat aralıkta mı?", "Evet" if inside else "Hayır", "Gerçek kazanılmış fiyat düşük-yüksek aralığında mı?")
    with c6:
        metric_card("Dengeli fiyattan fark", format_try(abs_error), "Gerçek fiyat ile orta öneri arasındaki TL farkı")
    with c7:
        metric_card("Dengeli fiyattan yüzde fark", format_pct(pct_error), "Gerçek fiyat ile orta öneri arasındaki yüzde fark")

    price_compare = pd.DataFrame(
        [
            {"Sıra": 1, "Fiyat noktası": "Düşük öneri", "Birim fiyat": corridor["predicted_low_price"], "Ne anlatır?": "Daha rekabetçi teklif seviyesi"},
            {"Sıra": 2, "Fiyat noktası": "Gerçek kazanılmış fiyat", "Birim fiyat": actual_price, "Ne anlatır?": "Sonuç açıldıktan sonra görülen tarihsel gerçek fiyat"},
            {"Sıra": 3, "Fiyat noktası": "Dengeli öneri", "Birim fiyat": corridor["predicted_mid_price"], "Ne anlatır?": "Emsallerin orta fiyat seviyesi"},
            {"Sıra": 4, "Fiyat noktası": "Seçilen en iyi senaryo", "Birim fiyat": float(best["proposed_unit_price"]), "Ne anlatır?": "Sistemin en yüksek skorlu teklif seçeneği"},
            {"Sıra": 5, "Fiyat noktası": "Yüksek öneri", "Birim fiyat": corridor["predicted_high_price"], "Ne anlatır?": "Daha yüksek karlılık hedefleyen fiyat seviyesi"},
        ]
    )
    st.dataframe(
        price_compare.drop(columns=["Sıra"]),
        hide_index=True,
        width="stretch",
        column_config={"Birim fiyat": st.column_config.NumberColumn(format="%.2f TL")},
    )

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Senaryo Sıralaması", "Gerçek kazanılmış senaryonun sistem önerileri içinde ne kadar üstte kaldığını gösterir.")
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        metric_card("Gerçek senaryo sırası", format_pct(rank_pct), "Gerçek fiyat aday senaryoların ne kadar üstünde kaldı? Yüksek değer daha iyi.")
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
                "Gerçek Karlılık Oranı": actual_margin,
                "Senaryo Skoru": float(best["scenario_score"]),
                "Profil Uyum Skoru": float(best.get("won_profile_fit_score", 0)),
                "Profil Grubu": best.get("cluster_name", "Hesaplanamadı"),
                "Sıra Dışılık Durumu": isolation_status,
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
        "Bu sayfa tek bir ihaleyi değil, test yılındaki tüm ihaleleri topluca ölçer. Her ihale önce gerçek sonucu gizlenmiş gibi analiz edilir; sonra gerçek kazanılmış sonuç açılarak sistemin genel tutarlılığı hesaplanır.",
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
    inlier_recall = float(results["is_inlier"].astype(bool).mean()) if "is_inlier" in results and not results.empty else float((results["won_profile_fit_score"] >= 45).mean()) if not results.empty else 0
    anomaly_rate = 1 - inlier_recall
    app_config = load_app_config()
    anomaly_warning_threshold = float(app_config.get("profile_fit", {}).get("aggressive_anomaly_rate_threshold", 0.25))

    info_callout(
        "Backtest geriye dönük canlı testtir: test yılındaki her ihalenin gerçek kazanılmış fiyatı ve karlılığı model girdisinden gizlenir, sistem önce emsal/profil/fiyat/senaryo çıktısı üretir, sonra gerçek sonuç açılarak karşılaştırılır. Gerçek Sonuçla Karşılaştır sayfası tek seçili ihaleyi gösterir; Backtest bu kontrolü tüm test yılına yayar. Amaç kazanma/kaybetme tahmini değil, fiyat aralığı ve profil uyumu yaklaşımının geçmiş kazanılmış ihalelerde ne kadar tutarlı çalıştığını ölçmektir.",
        "Backtest neyi anlatır?",
    )
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    section_header("Emsal ve Profil Metrikleri", "Benzer ihale kalitesi ve profil uyum dağılımı birlikte okunur.", "Emsal / Profil")
    retrieval_quantity = (
        float(results["retrieval_quantity_band_match_rate"].mean())
        if "retrieval_quantity_band_match_rate" in results and not results.empty
        else 0.0
    )
    p1, p2, p3, p4 = st.columns(4, gap="medium")
    with p1:
        metric_card("İlk 10 emsal benzerliği", f"{float(results['top10_avg_similarity'].mean()):.2f}", "Test yılı ortalaması", "blue")
    with p2:
        metric_card("Ürün grubu eşleşmesi", format_pct(float(results["retrieval_product_group_match_rate"].mean()) * 100), "Emsal havuzunda aynı ürün grubu", "green")
    with p3:
        metric_card("Bölge eşleşmesi", format_pct(float(results["retrieval_region_match_rate"].mean()) * 100), "Emsal havuzunda aynı bölge", "purple")
    with p4:
        metric_card("Miktar bandı eşleşmesi", format_pct(retrieval_quantity * 100), "Emsal havuzunda yakın ölçek", "amber")
    profile_distribution = pd.DataFrame(
        [
            ["Ortalama profil uyumu", format_score(results["won_profile_fit_score"].mean())],
            ["Medyan profil uyumu", format_score(results["won_profile_fit_score"].median())],
            ["Düşük uyum oranı", format_pct(float((results["won_profile_fit_score"] < 45).mean()) * 100)],
            ["Yüksek uyum oranı", format_pct(float((results["won_profile_fit_score"] >= 70).mean()) * 100)],
        ],
        columns=["Metrik", "Değer"],
    )
    st.dataframe(profile_distribution, hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("K-Means Metrikleri", "Test ihalelerinin hangi geçmiş başarı gruplarına dağıldığını gösterir.", "Profil grupları")
    cluster_summary = (
        results.groupby(["cluster_id", "cluster_name"], dropna=False)
        .agg(
            ihale_sayisi=("tender_id", "count"),
            baskin_urun_grubu=("product_group", lambda value: value.mode().iloc[0] if not value.mode().empty else "-"),
            baskin_kurum_tipi=("buyer_institution_type", lambda value: value.mode().iloc[0] if not value.mode().empty else "-"),
            baskin_bolge=("region", lambda value: value.mode().iloc[0] if not value.mode().empty else "-"),
            ortalama_profil_uyumu=("won_profile_fit_score", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "cluster_id": "Cluster ID",
                "cluster_name": "Cluster adı",
                "ihale_sayisi": "İhale sayısı",
                "baskin_urun_grubu": "Baskın ürün grubu",
                "baskin_kurum_tipi": "Baskın kurum tipi",
                "baskin_bolge": "Baskın bölge",
                "ortalama_profil_uyumu": "Ortalama profil uyumu",
            }
        )
    )
    st.dataframe(cluster_summary, hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Sıra Dışılık Kontrolü", "Isolation Forest kazanılmış test ihalelerini geçmiş profile göre normal mi daha az tipik mi görüyor?", "Isolation Forest")
    i1, i2, i3, i4 = st.columns(4, gap="medium")
    with i1:
        metric_card("Geçmiş profile uygun test oranı", format_pct(inlier_recall * 100), "Kazanılmış test ihalelerinin normal görülen oranı", "green")
    with i2:
        metric_card("Manuel inceleme oranı", format_pct(anomaly_rate * 100), "Daha az tipik görülen kazanılmış test oranı", "amber")
    with i3:
        metric_card("Hassasiyet ayarı", format_pct(float(results["isolation_contamination"].mean()) * 100), "Aktif contamination değeri", "purple")
    with i4:
        metric_card("En yüksek segment oranı", format_pct(float(results["segment_anomaly_rate"].max()) * 100), "Ürün grubu bazında maksimum manuel inceleme oranı", "red" if results["segment_anomaly_rate"].max() >= anomaly_warning_threshold else "blue")
    info_callout(
        "Bu veri setindeki tüm kayıtlar kazanılmış ihalelerden oluşur. Bu nedenle sıra dışı sonucu kayıp tahmini değildir; geçmiş kazanılmış profilden farklılık ve manuel inceleme sinyalidir.",
        "Sıra dışılık nasıl okunmalı?",
    )
    if anomaly_rate >= anomaly_warning_threshold:
        st.warning("Kazanılmış test ihalelerinde sıra dışı oranı yüksek. Isolation Forest ayarı fazla agresif olabilir; contamination değeri veya kullanılan özellikler gözden geçirilmeli.")
    segment_anomaly = (
        results.groupby("product_group", dropna=False)
        .agg(test_ihale_sayisi=("tender_id", "count"), anomaly_orani=("is_inlier", lambda value: 1 - value.astype(bool).mean()))
        .reset_index()
        .rename(columns={"product_group": "Ürün grubu", "test_ihale_sayisi": "Test ihalesi", "anomaly_orani": "Manuel inceleme oranı"})
    )
    st.dataframe(segment_anomaly, hide_index=True, width="stretch")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Fiyat Koridoru Metrikleri", "Fiyat aralığı doğruluğu, hata ve band genişliği birlikte değerlendirilir.", "Fiyat")
    f1, f2, f3, f4 = st.columns(4, gap="medium")
    with f1:
        metric_card("Band coverage", format_pct(metrics["band_coverage"] * 100), "Gerçek fiyatların düşük-yüksek aralığında kalma oranı", "green")
    with f2:
        metric_card("MAE", format_try(metrics["mae"]), "Dengeli fiyat ile gerçek fiyat arasındaki ortalama TL farkı", "blue")
    with f3:
        metric_card("MAPE", format_pct(metrics["mape"]), "Ortalama yüzde fiyat hatası", "amber")
    with f4:
        metric_card("Band kalite skoru", f"{metrics['coverage_adjusted_band_score']:.2f}", "Kapsama ve band genişliği birlikte okunur", "purple")
    f5, f6, f7 = st.columns(3, gap="medium")
    with f5:
        metric_card("SMAPE", format_pct(metrics["smape"]), "Simetrik yüzde hata", "cyan")
    with f6:
        metric_card("WAPE", format_pct(metrics["wape"]), "Ağırlıklı yüzde hata", "cyan")
    with f7:
        metric_card("Ortalama band genişliği", format_try(metrics["average_band_width"]), "Düşük ve yüksek öneri arasındaki ortalama fark", "amber")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Senaryo Metrikleri", "Tarihsel gerçek fiyat aday senaryolar içinde ne kadar iyi konumlandı?", "Senaryo")
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    with s1:
        metric_card("Gerçek senaryo sıra ortalaması", format_pct(opt["actual_won_scenario_rank_percentile_mean"]), "Yüksek değer daha iyi", "blue")
    with s2:
        metric_card("Top %30 hit rate", format_pct(opt["top30_hit_rate"] * 100), "Gerçek senaryo üst grupta mı?", "green")
    with s3:
        metric_card("Sert kural ihlal oranı", format_pct(opt["hard_constraint_violation_rate"] * 100), "Kuralı geçemeyen en iyi senaryo oranı", "amber")
    with s4:
        metric_card("Soft penalty ortalaması", f"{float(results['soft_penalty_score'].mean()):.1f}/100", "Risk ve soft penalty cezası", "purple")
    if "soft_penalty_score" in results:
        penalty_distribution = results["soft_penalty_score"].describe().reset_index()
        penalty_distribution.columns = ["Özet", "Soft penalty skoru"]
        st.dataframe(penalty_distribution, hide_index=True, width="stretch")
    metric_card("Yasak İddia Üretme Oranı", format_pct(forbidden_rate * 100), "AI Danışman güvenlik kontrolü. Hedef sıfırdır.", "red", "🛡️")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    section_header("Baz Model Karşılaştırması", "Tender IQ fiyat aralığı yaklaşımı; medyan, maliyet üstü fiyat ve regresyon gibi daha basit referanslarla kıyaslanır.", "Kıyas")
    baseline = baseline_predictions(pd.concat([split["train"], split["validation"]]), split["test"])
    baseline = baseline.rename(
        columns={
            "Model": "Yöntem",
            "MAE": "Ortalama Mutlak Hata",
            "MAPE": "Ortalama Yüzde Hata",
            "Coverage": "Aralıkta Kalma Oranı",
            "Avg Band Width": "Ortalama Aralık Genişliği",
        }
    )
    current_row = pd.DataFrame(
        [
            {
                "Yöntem": "Tender IQ mevcut yöntem",
                "Ortalama Mutlak Hata": metrics["mae"],
                "Ortalama Yüzde Hata": metrics["mape"],
                "Aralıkta Kalma Oranı": metrics["band_coverage"],
                "Ortalama Aralık Genişliği": metrics["average_band_width"],
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
        premium_card_html(
            "Sızıntı kontrolü",
            "Sonuç açılmadan önce gerçek sonuç alanlarının modele girmediği doğrulanır.",
            icon="SK",
            color="mint" if status == "success" else "amber",
            size="metric-size",
            footer_html=badge("Sızıntı yok" if status == "success" else "Sızıntı uyarısı", status),
        ),
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
        "Emsal İhale Analizi",
        "Bu sayfa, seçili ihaleye geçmişte kazanılmış en benzer ihaleleri gösterir. Bu emsaller profil uyumu, fiyat koridoru ve teklif senaryolarını besleyen ana referanslardır.",
        "Emsal",
    )
    tender = current_tender()
    if not tender:
        require_test_tender_message()
        return
    info_callout(
        "Benzerlik hesabında ürün adı, ürün grubu, kurum, bölge, ihale tipi ve miktar birlikte değerlendirilir. Metinsel alanlar TF-IDF ile sayısallaştırılır, ardından cosine similarity ile yeni ihale ve geçmiş ihaleler arasındaki yakınlık hesaplanır. Örneğin yeni ihale IV Solution ürün grubundaysa sistem geçmişteki IV Solution ihalelerini daha yüksek benzerlikte görür; aynı kurum tipi, benzer bölge ve yakın miktar varsa benzerlik daha da güçlenir.",
        "Benzerlik hesabı ve basit örnek:",
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
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    metric_card("Top-10 Emsal İhale Sayısı", format_int(min(10, len(similar))), "Karar ekranlarında ilk 10 güçlü emsal ayrıca izlenir.", "cyan")

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
        "Karlılık Oranı",
    ]
    section_header("Top-K Emsal İhaleler", "Benzerlik skoru yükseldikçe yeni ihale geçmiş kazanılmış profile daha yakın görünür.", "Emsal liste")
    st.dataframe(
        display.head(25),
        hide_index=True,
        width="stretch",
        column_config={
            "Benzerlik Skoru": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Tarihsel Kazanılmış Fiyat": st.column_config.NumberColumn(format="%.2f TL"),
            "Karlılık Oranı": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_advisor() -> None:
    init_session_state_defaults()
    page_header(
        "AI Danışman",
        "Sistem çıktısını Türkçe ve güvenli şekilde yorumlar. Soru sorabilir, gerekirse deterministik sistem yorumu alabilirsiniz.",
        "AI Danışman",
    )
    result = ensure_scenario_result()
    if not result:
        require_test_tender_message()
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
                    "Analiz bağlamı hazır. Bu ihalenin profil uyumunu, fiyat koridorunu, karlılığını, risklerini "
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
        "Bu ihale hangi profile benziyor?",
        "Sıra dışılık sonucu ne demek?",
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
            ("Profil grubu", str(context.get("cluster_name", "Kazanılmış profil grubu"))),
        ]
        st.markdown(
            premium_card_html(
                "Seçili ihale bağlamı",
                "Bu panel dışındaki bilgi danışman yanıtına dayanak yapılmaz.",
                icon="AI",
                color="blue",
                size="large-size",
                lines=context_rows,
                pill="Sızıntı yok" if leak.get("audit_status") == "pass" else "Sızıntı uyarısı",
            ),
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown("<div class='advisor-panel'>", unsafe_allow_html=True)
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
            st.markdown("<div class='divider-space'></div>", unsafe_allow_html=True)
            qcols = st.columns(3, gap="small")
            selected_question = None
            for idx, question in enumerate(quick_questions):
                with qcols[idx % 3]:
                    if st.button(question, key=f"quick_advisor_{idx}", width="stretch"):
                        selected_question = question

            st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
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
        st.markdown(advisor_payload_to_chat_text(advisor))
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
            "Sonuç açılmadan önce gerçek kazanılmış fiyat, final tutar ve gerçek karlılık oranı alanlarının modele girmediğini kontrol eder.",
            "Sızıntı yok" if leakage_ok else "Uyarı",
            "success" if leakage_ok else "danger",
        ),
        (
            "Forbidden Claim Detector",
            "Yasak iddia kontrolü",
            "AI Danışman çıktısında garanti, kesin sonuç veya gerçek kazanma olasılığı iddiası olup olmadığını denetler.",
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
            "Minimum karlılık oranı ve kural ihlali gibi geçersiz senaryo durumlarını raporlar.",
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
    render_premium_grid(
        [
            {
                "icon": f"{idx + 1:02d}",
                "title": tr_title,
                "body": body,
                "pill": f"{status_text} · {title}",
                "color": ["blue", "green", "purple", "amber", "cyan"][idx % 5],
            }
            for idx, (title, tr_title, body, status_text, status) in enumerate(audit_cards)
        ],
        columns=3,
        size="large-size",
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
elif page == "Test için İhale Seç":
    render_test_simulator()
elif page == "Emsal İhale Analizi":
    render_similar_tenders()
elif page == "Profil Uyum Analizi":
    render_profile_fit_analysis()
elif page == "Fiyat Koridoru ve Model Karşılaştırması":
    render_price_corridor_models()
elif page == "Teklif Senaryoları":
    render_scenario_analysis()
elif page == "Gerçek Sonuçla Karşılaştır":
    render_reveal_compare()
elif page == "Backtest Sonuçları":
    render_backtest()
elif page == "AI Danışman":
    render_advisor()
elif page == "Raporlar ve Kontroller":
    render_reports()
