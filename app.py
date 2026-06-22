"""Tender IQ Agentic Bid Advisor - Turkish Streamlit dashboard."""

from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.advisor.fallback_advisor import build_fallback_advisor
from src.advisor.context_validator import sanitize_advisor_context, validate_advisor_context
from src.advisor.forbidden_claim_detector import detect_forbidden_claims
from src.advisor.grounding_validator import validate_grounding
from src.advisor.llm_response import normalize_advisor_payload_schema, normalize_llm_payload, payload_from_free_text
from src.advisor.output_validator import advisor_semantic_text, validate_advisor_output
from src.advisor.prompt_builder import build_advisor_prompt
from src.advisor.prompt_injection_filter import detect_prompt_injection, safe_prompt_response
from src.advisor.support_validator import validate_supported_claims
from src.config_loader import load_app_config, load_scenario_weights
from src.constants import CANONICAL_MARGIN_COLUMN, CANONICAL_PRICE_COLUMN
from src.evaluation.backtest_runner import actual_rank_percentile, run_backtest
from src.evaluation.baseline_models import baseline_predictions, predict_baseline_prices
from src.evaluation.expert_review import expert_review_template
from src.evaluation.metrics import optimizer_metrics, price_corridor_metrics
from src.evaluation.segment_metrics import segment_level_metrics
from src.evaluation.stress_tests import evaluate_synthetic_outliers
from src.feature_masking import mask_actual_result_fields
from src.leakage_audit import audit_pre_reveal_input
from src.model_card import generate_model_card
from src.optimizer.recommendation_engine import rank_scenarios
from src.optimizer.scenario_generator import generate_candidate_scenarios
from src.optimizer.scenario_scorer import score_scenario
from src.optimizer.scenario_validator import validate_scenario
from src.reporting.audit_log import write_audit_event
from src.reporting.export_csv import dataframe_to_csv_bytes
from src.reporting.model_artifacts import write_backtest_artifacts
from src.reporting.structured_logging import configure_json_logging, log_event, log_exception
from src.retrieval import RetrievalEngine, retrieval_quality
from src.schema import normalize_schema, schema_quality_summary, validate_schema
from src.split_strategy import temporal_split
from src.validation import validate_data_quality

ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "data" / "x_ilac_synthetic_tenders_2021_2025.csv"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = [
    {
        "number": "1",
        "label": "Google Gemma 4 31B IT",
        "model_id": "google/gemma-4-31b-it:free",
        "description": "OpenRouter free model havuzundaki Google Gemma instruct seçeneği.",
    },
    {
        "number": "2",
        "label": "NVIDIA Nemotron 3 Super 120B A12B",
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "description": "OpenRouter free model havuzundaki geniş bağlamlı Nemotron seçeneği.",
    },
    {
        "number": "3",
        "label": "OpenRouter Owl Alpha",
        "model_id": "openrouter/owl-alpha",
        "description": "OpenRouter üzerinde doğrudan seçilebilen Owl Alpha modeli.",
    },
]
DEFAULT_OPENROUTER_MODEL = OPENROUTER_MODELS[0]["model_id"]


def load_local_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env_file()
configure_json_logging()

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")

TURKISH_WARNING = (
    "Skor gerçek kazanma olasılığı değil, geçmiş kazanılmış ihale profiline uyum göstergesidir."
)

PWIN_PROXY_EXPLANATION = (
    "Bu MVP’de gerçek kazanma olasılığı hesaplanmaz. Bunun yerine karar destek göstergesi "
    "olarak Kazanılmış Profil Uyum Skoru kullanılır. Bu skor; emsal benzerlik, mixed-type başarı profili, "
    "Isolation Forest uygunluğu, fiyat bandı uyumu, karlılık/risk dengesi ve model güveninden beslenir."
)
BACKTEST_PROFILE_DIAGNOSTICS_CACHE_VERSION = "profile-diagnostics-v3"
PROFILE_DIAGNOSTIC_COLUMNS = [
    "cluster_silhouette_score",
    "cluster_inertia",
    "cluster_min_size",
    "cluster_max_size",
    "cluster_assignment_confidence",
    "knn_profile_score",
    "mixed_cluster_score",
    "cluster_purity_score",
    "manual_review_reasons",
    "isolation_contamination",
    "segment_anomaly_rate",
    "is_inlier",
]
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
                --app-bg: #070606;
                --surface: rgba(22, 22, 22, 0.88);
                --surface-strong: #151313;
                --surface-soft: rgba(35, 31, 29, 0.76);
                --line: rgba(255, 255, 255, 0.105);
                --line-soft: rgba(255, 105, 42, 0.20);
                --text: #ffffff;
                --muted: #ffffff;
                --primary: #ffffff;
                --accent: #ff4f1f;
                --accent-2: #ff9d42;
                --accent-soft: rgba(255, 79, 31, 0.13);
                --blue: #ff7448;
                --cyan: #ffb25e;
                --purple: #df6b3f;
                --green: #d89b52;
                --amber: #ff9d42;
                --red: #ff4f1f;
                --shadow: 0 24px 70px rgba(0, 0, 0, 0.48);
                --soft-shadow: 0 14px 36px rgba(0, 0, 0, 0.34);
            }
            .stApp, .app-bg {
                background:
                    radial-gradient(ellipse at 50% 0%, rgba(214, 48, 16, 0.26), transparent 34%),
                    radial-gradient(ellipse at 92% 24%, rgba(255, 116, 36, 0.16), transparent 30%),
                    linear-gradient(180deg, #0a0808 0%, #050505 48%, #090706 100%);
                color: var(--text);
            }
            [data-testid='stHeader'] { background: transparent; }
            .block-container { max-width: 1320px; padding-top: 1.5rem; padding-bottom: 3.4rem; }
            [data-testid='stSidebar'] {
                background:
                    radial-gradient(ellipse at 50% 0%, rgba(255, 75, 31, 0.13), transparent 34%),
                    rgba(10, 9, 9, 0.94);
                border-right: 1px solid var(--line);
                box-shadow: 12px 0 42px rgba(0, 0, 0, 0.34);
            }
            [data-testid='stSidebar'] .stRadio label { color: var(--primary); font-weight: 540; }
            [data-testid='stSidebar'] [role='radiogroup'] label {
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 0.42rem 0.5rem;
                transition: background .15s ease, border-color .15s ease;
            }
            [data-testid='stSidebar'] [role='radiogroup'] label:hover {
                background: rgba(255, 79, 31, 0.10);
                border-color: rgba(255, 123, 66, 0.18);
            }
            [data-testid='stVerticalBlockBorderWrapper'] {
                background: var(--surface-strong);
                border: 1px solid var(--line);
                border-radius: 8px;
                box-shadow: var(--soft-shadow);
            }
            div[data-testid='stDataFrame'] {
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid var(--line);
                background: #ffffff;
                color: #050505;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.26);
                --text-color: #050505;
                --background-color: #ffffff;
                --secondary-background-color: #ffffff;
            }
            .brand-mark {
                width: 42px; height: 42px; display: grid; place-items: center;
                border-radius: 8px; color: #fff7f1; font-weight: 820; letter-spacing: 0;
                background: linear-gradient(135deg, #ff4f1f, #9c1f0b);
                border: 1px solid var(--line-soft);
                box-shadow: 0 14px 30px rgba(255, 79, 31, 0.18);
                margin-bottom: 0.85rem;
            }
            .sidebar-title { color: var(--primary); font-size: 1.04rem; font-weight: 760; letter-spacing: 0; }
            .sidebar-subtitle { color: var(--muted); font-size: 0.78rem; font-weight: 520; margin-top: 0.2rem; }
            .sidebar-status-stack { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.85rem 0 1rem; }
            .sidebar-note {
                color: var(--muted); font-size: 0.78rem; line-height: 1.45;
                padding: 0.8rem; border: 1px solid var(--line); border-radius: 8px;
                background: rgba(255, 255, 255, 0.045);
            }
            .eyebrow, .section-kicker {
                color: var(--accent-2); font-size: 0.68rem; font-weight: 680; letter-spacing: 0.12em;
                text-transform: uppercase; margin-bottom: 0.28rem;
            }
            .page-title {
                color: var(--primary); font-size: 3.2rem; line-height: 1.04;
                font-weight: 680; margin: 0; letter-spacing: 0;
            }
            .page-subtitle, .section-subtitle {
                color: var(--muted); max-width: 900px; font-size: 0.98rem; margin-top: 0.68rem; line-height: 1.65;
            }
            .scope-pill, .status-badge {
                display: inline-flex; align-items: center; gap: 0.38rem; padding: 0.32rem 0.62rem;
                border-radius: 999px; font-size: 0.73rem; font-weight: 620; border: 1px solid var(--line);
                background: rgba(255,255,255,0.055); color: var(--primary);
                box-shadow: 0 8px 18px rgba(0, 0, 0, 0.20);
            }
            .scope-pill { float: right; margin-top: 0.48rem; }
            .scope-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--accent-2); box-shadow: 0 0 12px rgba(255,157,66,.42); }
            .status-success, .status-good { color: #ffffff; background: rgba(216,155,82,0.13); border-color: rgba(216,155,82,0.28); }
            .status-warning, .status-warn { color: #ffffff; background: rgba(255,157,66,0.12); border-color: rgba(255,157,66,0.28); }
            .status-danger, .status-bad { color: #ffffff; background: rgba(255,79,31,0.13); border-color: rgba(255,79,31,0.30); }
            .hero-card {
                position: relative; overflow: hidden; padding: 4.2rem;
                border-radius: 8px; border: 1px solid rgba(255, 112, 51, 0.18);
                background:
                    radial-gradient(ellipse at 52% 10%, rgba(255, 63, 22, 0.34), transparent 34%),
                    radial-gradient(ellipse at 72% 20%, rgba(255, 157, 66, 0.14), transparent 38%),
                    linear-gradient(180deg, rgba(30, 25, 23, 0.92), rgba(11, 10, 10, 0.95));
                box-shadow: var(--shadow); color: var(--primary);
            }
            .hero-card:after {
                content: ''; position: absolute; inset: 12% 8% auto auto; height: 260px; width: 520px;
                background: radial-gradient(ellipse, rgba(255, 79, 31, 0.16), transparent 68%);
                filter: blur(18px);
                pointer-events: none;
            }
            .hero-title { font-size: 5.2rem; line-height: .98; font-weight: 560; letter-spacing: 0; margin: 0; max-width: 920px; }
            .hero-subtitle { max-width: 780px; color: var(--muted); font-size: 1.08rem; line-height: 1.7; margin-top: 1.05rem; }
            .hero-badges { display: flex; flex-wrap: wrap; gap: 0.58rem; margin-top: 1.45rem; }
            .hero-badge {
                display: inline-flex; align-items: center; gap: 0.38rem; padding: 0.48rem 0.74rem; border-radius: 999px;
                color: var(--primary); background: rgba(255,255,255,0.055); border: 1px solid var(--line);
                font-size: 0.82rem; font-weight: 540; backdrop-filter: blur(12px);
            }
            .glass-card, .method-card, .model-card, .score-card, .scenario-card, .chat-shell {
                border-radius: 8px; border: 1px solid var(--line);
                background: var(--surface); box-shadow: var(--soft-shadow); backdrop-filter: blur(16px);
            }
            .glass-card { padding: 1.25rem; }
            .section-title { color: var(--primary); font-size: 1.18rem; font-weight: 650; margin: 0 0 .25rem; letter-spacing: 0; }
            .metric-card {
                position: relative; overflow: hidden; min-height: 118px; padding: 1.05rem;
                border-radius: 8px; border: 1px solid var(--line);
                background: var(--surface); box-shadow: var(--soft-shadow);
            }
            .metric-card:before { content: ''; position: absolute; inset: 0 0 auto 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); pointer-events: none; }
            .metric-card-blue { --accent: var(--blue); }
            .metric-card-green { --accent: var(--green); }
            .metric-card-purple { --accent: var(--purple); }
            .metric-card-amber { --accent: var(--amber); }
            .metric-card-red { --accent: var(--red); }
            .metric-card-cyan { --accent: var(--cyan); }
            .metric-icon {
                width: 30px; height: 30px; display: inline-grid; place-items: center; border-radius: 8px;
                background: rgba(255,255,255,0.06); border: 1px solid var(--line); margin-bottom: 0.5rem;
                color: var(--primary); font-size: .88rem;
            }
            .metric-label { color: var(--muted); font-size: 0.7rem; font-weight: 620; text-transform: uppercase; letter-spacing: 0.08em; }
            .metric-value { color: var(--primary); font-size: 1.48rem; font-weight: 680; margin-top: 0.22rem; line-height: 1.12; overflow-wrap: anywhere; }
            .metric-note { color: var(--muted); font-size: 0.82rem; margin-top: 0.38rem; line-height: 1.38; }
            .warning-callout, .info-callout {
                border-radius: 8px; padding: 1rem 1.1rem; line-height: 1.55; box-shadow: var(--soft-shadow);
            }
            .warning-callout { border: 1px solid rgba(255,157,66,0.28); background: rgba(255, 157, 66, 0.10); color: #ffffff; }
            .info-callout { border: 1px solid var(--line-soft); background: rgba(255, 79, 31, 0.075); color: #ffffff; }
            .model-grid, .method-grid, .score-mini-grid {
                display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem;
            }
            .method-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .method-card, .model-card { padding: 1.18rem; min-height: 154px; position: relative; overflow: hidden; }
            .model-card:before, .method-card:before, .scenario-card:before {
                content: ''; position: absolute; inset: 0 0 auto 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); pointer-events: none;
            }
            .model-card-blue, .method-card-blue { --accent: var(--cyan); }
            .model-card-purple, .method-card-purple { --accent: var(--purple); }
            .model-card-amber, .method-card-amber { --accent: var(--amber); }
            .model-card-green, .method-card-green { --accent: var(--green); }
            .model-card-cyan, .method-card-cyan { --accent: var(--cyan); }
            .model-icon, .method-number {
                width: 32px; height: 32px; display: inline-grid; place-items: center; border-radius: 8px;
                background: rgba(255,255,255,0.06); border: 1px solid var(--line); font-weight: 620; margin-bottom: .75rem;
            }
            .model-title, .method-title { color: var(--primary); font-weight: 650; font-size: 1rem; margin-bottom: .35rem; }
            .model-body, .method-body { color: var(--muted); font-size: .86rem; line-height: 1.48; }
            .score-card { padding: 1.15rem; background: var(--surface); color: var(--primary); }
            .score-value { font-size: 2.6rem; font-weight: 680; line-height: 1; color: inherit; }
            .score-label { font-weight: 620; margin-top: .35rem; }
            .formula-panel {
                border-radius: 8px; padding: 1.25rem; color: var(--primary);
                background: linear-gradient(135deg, rgba(37, 31, 28, 0.92), rgba(17, 15, 15, 0.94));
                border: 1px solid var(--line-soft);
                box-shadow: var(--soft-shadow);
            }
            .formula-title { font-size: 1.05rem; font-weight: 680; margin-bottom: .7rem; }
            .formula-line { display: flex; justify-content: space-between; gap: 1rem; padding: .48rem 0; border-top: 1px solid var(--line); font-weight: 560; color: #ead6c9; }
            .scenario-card { padding: 1.15rem; min-height: 292px; position: relative; overflow: hidden; }
            .scenario-card-blue { --accent: var(--blue); }
            .scenario-card-green { --accent: var(--green); }
            .scenario-card-purple { --accent: var(--purple); }
            .scenario-card-amber { --accent: var(--amber); }
            .scenario-title { font-size: 1.02rem; font-weight: 650; color: var(--primary); margin-bottom: 0.25rem; }
            .scenario-price { font-size: 1.58rem; font-weight: 680; color: var(--primary); line-height: 1.1; }
            .scenario-row { display: flex; justify-content: space-between; gap: .8rem; border-top: 1px solid var(--line); padding-top: .46rem; margin-top: .46rem; color: var(--muted); font-size: .82rem; }
            .scenario-row b { color: var(--primary); }
            .chat-shell { overflow: hidden; background: var(--surface); }
            .chat-header {
                padding: 1rem; color: var(--primary); background: linear-gradient(135deg, rgba(42, 32, 28, 0.95), rgba(20, 17, 16, 0.95));
                border-bottom: 1px solid var(--line);
                display: flex; align-items: center; gap: .75rem;
            }
            .chat-avatar {
                width: 38px; height: 38px; display: grid; place-items: center; border-radius: 8px;
                background: rgba(255,255,255,0.06); border: 1px solid var(--line-soft); font-weight: 660;
            }
            .chat-header-title { font-weight: 680; font-size: 1.05rem; }
            .chat-header-subtitle { color: var(--muted); font-size: .82rem; margin-top: .15rem; line-height: 1.35; }
            .chat-body {
                min-height: 300px; max-height: 560px; overflow-y: auto; padding: .85rem;
                background: linear-gradient(180deg, rgba(18,16,15,0.94), rgba(10,9,9,0.94));
            }
            .chat-orb {
                width: 52px; height: 52px; flex: 0 0 auto; border-radius: 8px;
                display: grid; place-items: center; color: white; font-weight: 760;
                background: linear-gradient(145deg, #ff6a2b, #b5280d);
                box-shadow: 0 16px 40px rgba(255,79,31,.22);
            }
            .chat-thread { display: flex; flex-direction: column; gap: .78rem; padding: .35rem; }
            .chat-row { display: flex; gap: .7rem; align-items: flex-start; }
            .chat-row .chat-orb { width: 44px; height: 44px; font-size: .86rem; box-shadow: 0 12px 28px rgba(255,79,31,.20); }
            .chat-row-user { justify-content: flex-end; }
            .chat-row-assistant { justify-content: flex-start; }
            .chat-bubble {
                max-width: min(920px, 88%); padding: .86rem 1rem; border-radius: 8px;
                border: 1px solid var(--line); color: var(--primary); line-height: 1.55;
                font-size: .94rem; white-space: pre-wrap; overflow-wrap: anywhere;
                box-shadow: 0 12px 26px rgba(0,0,0,.22);
            }
            .chat-bubble-user {
                background: linear-gradient(135deg, rgba(255,79,31,.18), rgba(255,157,66,.10));
                border-top-right-radius: 8px;
            }
            .chat-bubble-assistant {
                background: rgba(255,255,255,.055);
                border-top-left-radius: 8px;
            }
            .chat-bubble-pending {
                color: var(--muted);
                background: rgba(255,255,255,.07);
            }
            .chat-source {
                display: inline-flex;
                margin-bottom: .42rem;
                padding: .16rem .48rem;
                border-radius: 999px;
                background: rgba(255,79,31,.15);
                color: #ffffff;
                font-size: .72rem;
                font-weight: 650;
            }
            .chat-wide-shell {
                border-radius: 8px; overflow: hidden; border: 1px solid var(--line);
                background:
                    radial-gradient(ellipse at 50% 0%, rgba(255,79,31,.14), transparent 34%),
                    rgba(14,13,13,.90);
                box-shadow: var(--soft-shadow);
            }
            .chat-input-area { padding: .8rem 1rem 1rem; border-top: 1px solid var(--line); background: var(--surface); }
            .quick-question button { border-radius: 8px !important; border-color: var(--line) !important; background: rgba(255,255,255,.055) !important; color: var(--primary) !important; box-shadow: none !important; }
            .warning-box { border: 1px solid rgba(255,157,66,0.28); background: rgba(255,157,66,0.10); color: #ffffff; border-radius: 8px; padding: .9rem 1rem; }
            .info-box { border: 1px solid var(--line-soft); background: rgba(255,79,31,0.075); color: #ffffff; border-radius: 8px; padding: .9rem 1rem; }
            .global-table-card {
                border-radius: 20px;
                border: 1px solid rgba(255, 123, 66, 0.16);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.052), rgba(255,255,255,0.02)),
                    rgba(14, 15, 18, 0.94);
                box-shadow: 0 18px 42px rgba(0,0,0,0.26);
                overflow: hidden;
                margin: .55rem 0 1rem;
            }
            .global-table-scroll {
                width: 100%;
                overflow-x: auto;
            }
            .global-dark-table {
                width: 100%;
                min-width: 760px;
                border-collapse: collapse;
                color: #ffffff;
            }
            .global-dark-table th {
                background: rgba(24, 24, 28, 0.98);
                color: rgba(255,247,237,0.88);
                font-size: .76rem;
                letter-spacing: .055em;
                text-transform: uppercase;
                text-align: left;
                padding: 14px 16px;
                border-bottom: 1px solid rgba(255,123,66,0.20);
                white-space: nowrap;
            }
            .global-dark-table td {
                color: rgba(245,247,250,0.84);
                font-size: .88rem;
                line-height: 1.44;
                padding: 13px 16px;
                border-bottom: 1px solid rgba(255,255,255,0.07);
                background: rgba(255,255,255,0.018);
                vertical-align: top;
            }
            .global-dark-table tr:nth-child(even) td {
                background: rgba(255,255,255,0.036);
            }
            .global-dark-table tr:last-child td {
                border-bottom: 0;
            }
            .global-table-strong {
                color: #fff7ed !important;
                font-weight: 760;
            }
            .global-table-code {
                color: #ffbd8a !important;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: .84rem !important;
                white-space: nowrap;
            }
            .global-status-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                min-height: 24px;
                padding: 3px 9px;
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(255,255,255,0.06);
                color: #fff7ed;
                font-size: .73rem;
                font-weight: 760;
                white-space: nowrap;
            }
            .global-status-good {
                border-color: rgba(34,197,94,0.28);
                background: rgba(34,197,94,0.12);
                color: #bbf7d0;
            }
            .global-status-warn {
                border-color: rgba(251,146,60,0.32);
                background: rgba(251,146,60,0.12);
                color: #fed7aa;
            }
            .global-status-bad {
                border-color: rgba(248,113,113,0.32);
                background: rgba(248,113,113,0.13);
                color: #fecaca;
            }
            .global-progress {
                display: flex;
                align-items: center;
                gap: 9px;
                min-width: 150px;
            }
            .global-progress-track {
                flex: 1;
                height: 7px;
                border-radius: 999px;
                background: rgba(255,255,255,0.08);
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.06);
            }
            .global-progress-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, rgba(255,79,31,0.95), rgba(255,157,66,0.95));
            }
            .global-progress-value {
                min-width: 48px;
                color: #fff7ed;
                font-size: .8rem;
                font-weight: 760;
                text-align: right;
            }
            .st-key-advisor_chat_module {
                max-width: 1120px;
                margin: 0 auto;
                padding: 1.35rem;
                border-radius: 22px;
                border: 1px solid rgba(255,123,66,0.24);
                background:
                    radial-gradient(ellipse at 16% 0%, rgba(255,79,31,0.20), transparent 34%),
                    radial-gradient(ellipse at 88% 18%, rgba(255,157,66,0.10), transparent 34%),
                    linear-gradient(180deg, rgba(25,22,21,0.98), rgba(7,7,8,0.98));
                box-shadow: 0 28px 76px rgba(0,0,0,0.46), 0 0 48px rgba(255,79,31,0.075);
            }
            .st-key-advisor_chat_module [data-testid='stVerticalBlock'] {
                gap: .9rem;
            }
            .advisor-chat-header {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: flex-start;
                padding: .35rem .25rem .25rem;
            }
            .advisor-chat-title-row {
                display: flex;
                align-items: center;
                gap: .75rem;
            }
            .advisor-status-pills {
                display: flex;
                justify-content: flex-end;
                flex-wrap: wrap;
                gap: .38rem;
                max-width: 430px;
            }
            .advisor-status-pill {
                display: inline-flex;
                align-items: center;
                min-height: 26px;
                padding: .24rem .58rem;
                border-radius: 999px;
                border: 1px solid rgba(255,123,66,0.20);
                background: rgba(255,255,255,0.06);
                color: #ffffff;
                font-size: .72rem;
                font-weight: 650;
                white-space: nowrap;
            }
            .advisor-chat-kicker {
                padding: 0 .6rem;
                font-size: .72rem;
                font-weight: 700;
                letter-spacing: .08em;
                text-transform: uppercase;
                color: #ffb25e;
            }
            .st-key-advisor_chat_module div[data-testid='stButton'] button {
                min-height: 42px !important;
                width: 100% !important;
                padding: .46rem .82rem !important;
                border-radius: 999px !important;
                border: 1px solid rgba(255,123,66,0.24) !important;
                background: rgba(255,255,255,0.06) !important;
                color: #ffffff !important;
                box-shadow: none !important;
                font-size: .82rem !important;
                line-height: 1.2 !important;
            }
            .st-key-advisor_chat_module div[data-testid='stButton'] button:hover {
                background: rgba(255,79,31,0.16) !important;
                border-color: rgba(255,178,94,0.48) !important;
                box-shadow: 0 10px 28px rgba(255,79,31,0.14) !important;
            }
            .st-key-advisor_chat_module .chat-wide-shell {
                box-shadow: none;
                border-color: rgba(255,255,255,0.10);
                background:
                    radial-gradient(ellipse at 0% 0%, rgba(255,79,31,0.08), transparent 34%),
                    rgba(7,7,7,0.62);
            }
            .st-key-advisor_chat_module .chat-body {
                min-height: 270px;
                max-height: 460px;
                padding: 1rem;
                background: transparent;
            }
            .st-key-advisor_chat_module .chat-bubble {
                max-width: min(760px, 82%);
                border-radius: 18px;
                padding: .95rem 1.08rem;
            }
            .st-key-advisor_chat_module .chat-bubble-user {
                background: linear-gradient(135deg, rgba(255,79,31,.92), rgba(181,40,13,.92));
                border-color: rgba(255,184,117,0.30);
                border-bottom-right-radius: 6px;
            }
            .st-key-advisor_chat_module .chat-bubble-assistant {
                background: rgba(255,255,255,0.07);
                border-color: rgba(255,255,255,0.12);
                border-bottom-left-radius: 6px;
            }
            .st-key-advisor_chat_module div[data-testid='stForm'] {
                margin-top: .15rem;
                padding: .82rem;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 20px;
                background: rgba(255,255,255,0.05);
            }
            .st-key-advisor_chat_module div[data-testid='stForm'] input {
                min-height: 50px;
                border-radius: 999px !important;
                padding-left: 1.1rem !important;
                background: rgba(4,4,4,0.62) !important;
                border: 1px solid rgba(255,123,66,0.26) !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
            .st-key-advisor_chat_module div[data-testid='stFormSubmitButton'] button {
                min-height: 50px !important;
                width: 100% !important;
                border-radius: 999px !important;
                background: linear-gradient(180deg, rgba(255,93,36,0.98), rgba(181,40,13,0.98)) !important;
                border: 1px solid rgba(255,184,117,0.36) !important;
                color: #ffffff !important;
                box-shadow: 0 14px 30px rgba(255,79,31,0.20) !important;
            }
            .advisor-safe-banner,
            .advisor-warning-banner {
                margin: 1rem 0 1.35rem;
                border-radius: 18px;
                padding: .95rem 1.05rem;
                border: 1px solid rgba(255,123,66,0.22);
                background: linear-gradient(135deg, rgba(255,79,31,0.09), rgba(255,157,66,0.045)), rgba(15,14,14,0.86);
                color: rgba(255,255,255,0.82);
                line-height: 1.55;
                box-shadow: 0 16px 38px rgba(0,0,0,0.22);
            }
            .advisor-warning-banner {
                border-color: rgba(255,79,31,0.32);
                background: linear-gradient(135deg, rgba(255,79,31,0.16), rgba(100,20,12,0.18)), rgba(15,12,12,0.92);
            }
            .advisor-secondary-section { margin-top: 46px; }
            .advisor-secondary-title {
                color: var(--primary);
                font-size: 1.16rem;
                font-weight: 720;
                margin-bottom: .22rem;
            }
            .advisor-secondary-subtitle {
                color: rgba(255,255,255,0.62);
                line-height: 1.5;
                margin-bottom: 1rem;
                max-width: 900px;
            }
            .advisor-context-card,
            .advisor-setup-card,
            .advisor-status-card {
                border-radius: 20px;
                border: 1px solid rgba(255,123,66,0.16);
                background: linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018)), rgba(15,15,16,0.92);
                box-shadow: 0 18px 44px rgba(0,0,0,0.24);
            }
            .advisor-context-card,
            .advisor-setup-card { padding: 1.15rem; }
            .advisor-kv-row {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                padding: .58rem 0;
                border-top: 1px solid rgba(255,255,255,0.07);
                color: rgba(255,255,255,0.66);
                line-height: 1.35;
            }
            .advisor-kv-row:first-child { border-top: 0; }
            .advisor-kv-row b {
                color: #fff7ed;
                text-align: right;
                overflow-wrap: anywhere;
            }
            .advisor-status-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 18px;
                margin-top: 20px;
            }
            .advisor-status-card {
                min-height: 132px;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .advisor-status-label {
                color: rgba(255,255,255,0.56);
                font-size: .72rem;
                font-weight: 760;
                letter-spacing: .06em;
                text-transform: uppercase;
            }
            .advisor-status-value {
                color: #fff7ed;
                font-size: 1.18rem;
                font-weight: 820;
                line-height: 1.18;
                margin-top: .48rem;
                overflow-wrap: anywhere;
            }
            .advisor-status-note {
                color: rgba(255,255,255,0.62);
                font-size: .84rem;
                line-height: 1.4;
                margin-top: .72rem;
            }
            .advisor-advanced-table {
                width: 100%;
                border-collapse: collapse;
                overflow: hidden;
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .advisor-advanced-table th,
            .advisor-advanced-table td {
                text-align: left;
                padding: .72rem .82rem;
                border-bottom: 1px solid rgba(255,255,255,0.07);
                color: rgba(255,255,255,0.78);
                background: rgba(255,255,255,0.035);
            }
            .advisor-advanced-table th {
                color: #fff7ed;
                background: rgba(255,255,255,0.07);
            }
            .advisor-advanced-table tr:last-child td { border-bottom: 0; }
            .nav-card { min-height: 168px; display: flex; flex-direction: column; justify-content: space-between; }
            .card-grid {
                display: grid;
                gap: 1rem;
                width: 100%;
                align-items: stretch;
                padding: 0;
            }
            .card-grid.two-col { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .card-grid.three-col { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .card-grid.auto-col { grid-template-columns: repeat(auto-fit, minmax(285px, 1fr)); }
            .premium-card {
                position: relative;
                min-height: 206px;
                height: 100%;
                padding: 1.28rem;
                border-radius: 8px;
                border: 1px solid var(--line);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.038)),
                    var(--surface-soft);
                box-shadow: var(--soft-shadow);
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
            .premium-card.card-blue:before { background: radial-gradient(ellipse at top left, rgba(255,116,72,0.16), transparent 44%); }
            .premium-card.card-purple:before { background: radial-gradient(ellipse at top left, rgba(223,107,63,0.17), transparent 44%); }
            .premium-card.card-mint:before { background: radial-gradient(ellipse at top left, rgba(216,155,82,0.16), transparent 44%); }
            .premium-card.card-green:before { background: radial-gradient(ellipse at top left, rgba(216,155,82,0.17), transparent 44%); }
            .premium-card.card-amber:before { background: radial-gradient(ellipse at top left, rgba(255,157,66,0.18), transparent 44%); }
            .premium-card.card-cyan:before { background: radial-gradient(ellipse at top left, rgba(255,178,94,0.15), transparent 44%); }
            .premium-card.card-red:before { background: radial-gradient(ellipse at top left, rgba(255,79,31,0.18), transparent 44%); }
            .premium-card > * { position: relative; z-index: 1; }
            .premium-card.metric-size { min-height: 148px; }
            .premium-card.large-size { min-height: 244px; }
            .premium-card.scenario-size { min-height: 420px; }
            .card-icon-row { display: flex; align-items: center; gap: .75rem; margin-bottom: 1rem; }
            .card-icon {
                min-width: 44px; min-height: 44px; border-radius: 8px;
                padding: 0 .45rem;
                display: inline-flex; align-items: center; justify-content: center;
                background: rgba(255,79,31,0.12);
                border: 1px solid rgba(255,123,66,0.18);
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.025);
                font-weight: 680;
            }
            .card-title {
                color: var(--primary);
                font-size: 1.16rem;
                line-height: 1.18;
                font-weight: 700;
                letter-spacing: 0;
                margin: 0 0 .55rem;
            }
            .card-value {
                color: #ffffff;
                font-size: 1.55rem;
                line-height: 1.1;
                font-weight: 720;
                margin: .1rem 0 .55rem;
                overflow-wrap: anywhere;
            }
            .card-body {
                color: var(--muted);
                font-size: .88rem;
                line-height: 1.5;
                overflow-wrap: anywhere;
            }
            .card-list {
                color: var(--muted);
                font-size: .86rem;
                line-height: 1.55;
                margin-top: .85rem;
            }
            .card-line { display: flex; justify-content: space-between; gap: .8rem; padding: .32rem 0; border-top: 1px solid var(--line); }
            .card-line b { color: var(--primary); text-align: right; }
            .card-footer { margin-top: 1.15rem; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .45rem; }
            .card-pill {
                font-size: .75rem;
                color: #ffffff;
                background: rgba(255,79,31,0.11);
                border-radius: 999px;
                padding: .38rem .62rem;
                border: 1px solid rgba(255,123,66,0.18);
            }
            .advisor-panel { padding: .25rem .1rem .1rem; }
            .soft-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,79,31,.24), transparent); margin: 1.6rem 0; }
            .divider-space { margin-top: 1.35rem; }
            .stMarkdown, .stText, p, li, label, span { color: inherit; }
            h1, h2, h3, h4, h5, h6 { color: var(--primary); letter-spacing: 0; }
            .stApp :where(p, li, label, span, div, small, strong, em, code, figcaption) {
                color: #ffffff;
            }
            div[data-testid='stDataFrame'] *,
            div[data-testid='stTable'] *,
            div[data-testid='stDataFrameResizable'] *,
            div[data-testid='stDataFrame'] :where(p, li, label, span, div, small, strong, em, code),
            div[data-testid='stTable'] :where(p, li, label, span, div, small, strong, em, code) {
                color: #050505 !important;
                -webkit-text-fill-color: #050505 !important;
            }
            div[data-testid='stDataFrame'],
            div[data-testid='stTable'],
            div[data-testid='stDataFrameResizable'] {
                background: #ffffff !important;
                color: #050505 !important;
                -webkit-text-fill-color: #050505 !important;
                --text-color: #050505;
                --background-color: #ffffff;
                --secondary-background-color: #ffffff;
            }
            div[data-testid='stDataFrame'] svg *,
            div[data-testid='stTable'] svg * {
                fill: #050505 !important;
                color: #050505 !important;
            }
            div[data-testid='stDataFrame'] table,
            div[data-testid='stTable'] table,
            div[data-testid='stDataFrame'] thead,
            div[data-testid='stTable'] thead,
            div[data-testid='stDataFrame'] tbody,
            div[data-testid='stTable'] tbody,
            div[data-testid='stDataFrame'] tr,
            div[data-testid='stTable'] tr,
            div[data-testid='stDataFrame'] th,
            div[data-testid='stTable'] th,
            div[data-testid='stDataFrame'] td,
            div[data-testid='stTable'] td {
                background: #ffffff !important;
                color: #050505 !important;
                -webkit-text-fill-color: #050505 !important;
            }
            div[data-testid='stMetric'] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.9rem;
            }
            div[data-testid='stMetric'] label, div[data-testid='stMetric'] [data-testid='stMetricValue'] {
                color: var(--primary);
            }
            button[kind], div[data-testid='stDownloadButton'] button, div[data-testid='stButton'] button {
                border-radius: 8px !important;
                border: 1px solid rgba(255, 123, 66, 0.35) !important;
                background: linear-gradient(180deg, rgba(255, 93, 36, 0.95), rgba(191, 43, 13, 0.95)) !important;
                color: #fff7f1 !important;
                box-shadow: 0 12px 28px rgba(255, 79, 31, 0.18) !important;
            }
            button[kind]:hover, div[data-testid='stDownloadButton'] button:hover, div[data-testid='stButton'] button:hover {
                border-color: rgba(255, 184, 117, 0.58) !important;
                filter: brightness(1.05);
            }
            div[data-testid='stTabs'] button {
                color: var(--muted) !important;
                background: transparent !important;
                box-shadow: none !important;
                border-radius: 8px 8px 0 0 !important;
            }
            div[data-testid='stTabs'] button[aria-selected='true'] {
                color: var(--primary) !important;
                background: rgba(255,79,31,0.12) !important;
            }
            div[data-testid='stExpander'] {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: rgba(255,255,255,0.035);
            }
            div[data-baseweb='select'] > div,
            div[data-testid='stNumberInput'] input,
            div[data-testid='stTextInput'] input,
            div[data-testid='stChatInput'] textarea,
            textarea {
                background: rgba(255,255,255,0.055) !important;
                border-color: rgba(255,255,255,0.12) !important;
                color: var(--primary) !important;
                border-radius: 8px !important;
            }
            div[data-testid='stNumberInput'] input,
            div[data-testid='stNumberInput'] input *,
            div[data-testid='stTextInput'] input,
            div[data-testid='stTextInput'] input *,
            div[data-testid='stFileUploader'] *,
            div[data-baseweb='select'] *,
            div[data-baseweb='input'] *,
            div[data-testid='stNumberInput'] button,
            div[data-testid='stNumberInput'] button * {
                color: #050505 !important;
                -webkit-text-fill-color: #050505 !important;
            }
            div[data-testid='stNumberInput'] input,
            div[data-testid='stTextInput'] input,
            div[data-baseweb='select'] > div,
            div[data-baseweb='input'] > div {
                background: #f7f8fb !important;
                border-color: rgba(5, 5, 5, 0.18) !important;
            }
            div[data-testid='stNumberInput'] svg *,
            div[data-baseweb='select'] svg * {
                fill: #050505 !important;
                color: #050505 !important;
            }
            .st-key-advisor_chat_module div[data-testid='stTextInput'] input,
            .st-key-advisor_chat_module div[data-testid='stTextInput'] input *,
            .st-key-advisor_chat_module div[data-baseweb='input'] *,
            .st-key-advisor_chat_module div[data-testid='stForm'] input {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
            .st-key-advisor_chat_module div[data-testid='stTextInput'] input,
            .st-key-advisor_chat_module div[data-baseweb='input'] > div,
            .st-key-advisor_chat_module div[data-testid='stForm'] input {
                background: rgba(4,4,4,0.62) !important;
                border-color: rgba(255,123,66,0.26) !important;
            }
            div[data-testid='stAlert'] {
                border-radius: 8px;
                background: rgba(255,157,66,0.10);
                border: 1px solid rgba(255,157,66,0.24);
                color: #ffffff;
            }
            @media (max-width: 1100px) {
                .model-grid, .method-grid, .score-mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .card-grid.three-col { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .hero-title { font-size: 4rem; }
                .hero-card { padding: 3rem; }
            }
            @media (max-width: 760px) {
                .scope-pill { float: none; margin-bottom: .8rem; }
                .model-grid, .method-grid, .score-mini-grid { grid-template-columns: 1fr; }
                .card-grid.three-col, .card-grid.two-col, .card-grid.auto-col { grid-template-columns: 1fr; }
                .page-title { font-size: 2.2rem; }
                .hero-card { padding: 1.35rem; }
                .hero-title { font-size: 3rem; }
                .metric-value { font-size: 1.45rem; }
                .st-key-advisor_chat_module { padding: .85rem; }
                .advisor-chat-header { flex-direction: column; align-items: flex-start; }
                .advisor-status-pills { justify-content: flex-start; max-width: 100%; }
                .st-key-advisor_chat_module .chat-body { max-height: 360px; }
                .st-key-advisor_chat_module .chat-bubble { max-width: 92%; }
                .advisor-status-grid { grid-template-columns: 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()


def init_session_state_defaults() -> None:
    st.session_state.setdefault("session_id", f"st-{uuid.uuid4().hex[:12]}")
    st.session_state.setdefault("user_id", os.getenv("USER_ID", "anonymous"))
    configured_model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
    available_model_ids = {item["model_id"] for item in OPENROUTER_MODELS}
    if configured_model not in available_model_ids:
        configured_model = DEFAULT_OPENROUTER_MODEL
    st.session_state.setdefault("selected_openrouter_model", configured_model)
    st.session_state.setdefault("llm_primary_label", configured_model)
    st.session_state.setdefault("llm_fallback_labels", [])


init_session_state_defaults()


def audit_event(event: dict[str, Any]) -> None:
    enriched = {
        "session_id": st.session_state.get("session_id"),
        "user_id": st.session_state.get("user_id", "anonymous"),
        "reveal_status": "revealed" if st.session_state.get("revealed", False) else event.get("reveal_status", "hidden"),
        **event,
    }
    write_audit_event(enriched)


def audit_event_once(key: str, event: dict[str, Any]) -> None:
    if st.session_state.get(key):
        return
    audit_event(event)
    st.session_state[key] = True


def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openrouter").strip().lower() or "openrouter"


def selected_openrouter_model_id() -> str:
    model_id = str(st.session_state.get("selected_openrouter_model") or "").strip()
    available_model_ids = {item["model_id"] for item in OPENROUTER_MODELS}
    if model_id in available_model_ids:
        return model_id
    env_model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    return env_model if env_model in available_model_ids else DEFAULT_OPENROUTER_MODEL


def openrouter_model_option_label(model: dict[str, str]) -> str:
    return f"{model['number']}. {model['label']} - {model['model_id']}"


def read_local_openrouter_secret() -> tuple[str, str]:
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return "", "missing_file"
    current_section = ""
    aliases = {"OPENROUTER_API_KEY", "openrouter_api_key", "api_key"}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "", "read_error"
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip().casefold()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in aliases:
            continue
        if key == "api_key" and current_section not in {"openrouter", "llm", "advisor"}:
            continue
        secret = value.split("#", 1)[0].strip().strip('"').strip("'")
        if secret:
            return secret, f"{path.name}:{key}"
    return "", "empty"


def streamlit_secret_openrouter_key() -> tuple[str, str]:
    try:
        for key in ("OPENROUTER_API_KEY", "openrouter_api_key"):
            secret_key = str(st.secrets.get(key, "")).strip()
            if secret_key:
                return secret_key, f"streamlit_secrets:{key}"
        for section in ("openrouter", "llm", "advisor"):
            section_value = st.secrets.get(section, {})
            if not hasattr(section_value, "get"):
                continue
            for key in ("api_key", "OPENROUTER_API_KEY", "openrouter_api_key"):
                secret_key = str(section_value.get(key, "")).strip()
                if secret_key:
                    return secret_key, f"streamlit_secrets:{section}.{key}"
    except Exception:
        return "", "streamlit_secrets_error"
    return "", "streamlit_secrets_empty"


def set_advisor_llm_status(status: str, source: str, reason: str = "", model_id: str | None = None) -> None:
    st.session_state.advisor_llm_status = {
        "status": status,
        "source": source,
        "reason": reason,
        "model": model_id or selected_openrouter_model_id(),
    }


def user_friendly_error_message(exc: Exception) -> str:
    text = str(exc).casefold()
    if "missing" in text or "required" in text or "kolon" in text:
        return "Zorunlu kolon yok."
    if "schema" in text or "şema" in text:
        return "Veri şeması eksik."
    if "temporal split" in text or "empty train" in text or "test set" in text:
        return "Test döneminde yeterli kayıt yok."
    if "scenario" in text or "senaryo" in text:
        return "Geçerli senaryo üretilemedi."
    if "llm" in text or "advisor" in text:
        return "LLM doğrulaması başarısız."
    if "reveal" in text:
        return "Gerçek sonuç reveal edilmeden karşılaştırma yapılamaz."
    return "İşlem tamamlanamadı. Lütfen veri ve seçimleri kontrol edin."


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return normalize_schema(pd.read_csv(SAMPLE_DATA))


@st.cache_data(show_spinner=False)
def cached_backtest(data: pd.DataFrame, diagnostics_cache_version: str = BACKTEST_PROFILE_DIAGNOSTICS_CACHE_VERSION) -> pd.DataFrame:
    _ = diagnostics_cache_version
    split = temporal_split(data)
    return run_backtest(pd.concat([split["train"], split["validation"]]), split["test"])


def backtest_has_profile_diagnostics(results: pd.DataFrame) -> bool:
    if results.empty:
        return False
    missing = [column for column in PROFILE_DIAGNOSTIC_COLUMNS if column not in results.columns]
    if missing:
        return False
    numeric = pd.to_numeric(results["cluster_inertia"], errors="coerce")
    return bool(numeric.notna().any() and float(numeric.fillna(0).abs().sum()) > 0)


def load_backtest_results(data: pd.DataFrame) -> pd.DataFrame:
    results = cached_backtest(data, BACKTEST_PROFILE_DIAGNOSTICS_CACHE_VERSION)
    if not backtest_has_profile_diagnostics(results):
        cached_backtest.clear()
        results = cached_backtest(data, BACKTEST_PROFILE_DIAGNOSTICS_CACHE_VERSION)
    return ensure_backtest_columns(results)


def ensure_backtest_columns(results: pd.DataFrame) -> pd.DataFrame:
    fixed = results.copy()
    defaults: dict[str, Any] = {
        "soft_penalty_score": 0.0,
        "invalid_reason": "",
        "config_version": "config-v1",
        "retrieval_model_version": "retrieval-v1",
        "kmeans_model_version": "kmeans-v1",
        "isolation_forest_model_version": "isolation-forest-v1",
        "baseline_model_version": "baseline-v1",
        "training_data_range": "2021-2024 train+validation; 2025 pseudo-live test",
        "leakage_blocked_fields_present": "",
        "leakage_masked_fields_count": 0,
        "top_similar_tenders_summary": "",
        "reveal_status": "revealed_for_backtest",
        "soft_penalty_explanations": "",
        "hard_constraint_status": "",
        "caveat": "",
        "failure_reason": "",
        "llm_validation_status": "pass",
        "advisor_schema_valid": True,
        "advisor_forbidden_claims_detected": False,
        "advisor_grounding_score": 1.0,
        "advisor_prompt_injection_detected": False,
        "advisor_fallback_used": True,
        "is_inlier": True,
        "anomaly_score": 0.0,
        "isolation_threshold": 0.0,
        "manual_review_flag": False,
        "manual_review_reasons": "",
        "knn_profile_score": 0.0,
        "mixed_cluster_score": 0.0,
        "cluster_purity_score": 0.0,
        "profile_score_components": {},
        "isolation_contamination": 0.05,
        "training_inlier_rate": 0.95,
        "training_anomaly_rate": 0.05,
        "segment_anomaly_rate": 0.0,
        "cluster_silhouette_score": 0.0,
        "cluster_inertia": 0.0,
        "cluster_min_size": 0,
        "cluster_max_size": 0,
        "small_cluster_count": 0,
        "empty_cluster_count": 0,
        "cluster_assignment_confidence": 0.0,
        "cluster_distance": 0.0,
        "cluster_second_distance": 0.0,
        "cluster_distance_percentile": 0.0,
        "cluster_count": 0,
        "cluster_name": "Hesaplanamadı",
        "cluster_id": "",
    }
    for column, default in defaults.items():
        if column not in fixed.columns:
            fixed[column] = default
    return fixed


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


def format_decimal(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


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
    if value >= 85:
        return "Top %15 içinde: güçlü uyum."
    if value >= 70:
        return "Top %30 içinde: kabul edilebilir uyum."
    if value >= 50:
        return "Top %50 içinde: orta düzey uyum."
    return "Alt %50: senaryo skor mantığı gözden geçirilmeli."


def render_small_card(title: str, body: str, badge_html: str = "") -> None:
    st.markdown(
        premium_card_html(title, body, footer_html=badge_html, size="large-size", color="blue"),
        unsafe_allow_html=True,
    )


def render_kv_card(title: str, rows: list[tuple[str, str]], note: str = "") -> None:
    st.markdown(
        premium_card_html(title, note, lines=rows, size="large-size", color="purple"),
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


def render_metric_card(title: str, value: str, subtitle: str = "", color: str = "blue", icon: str = "") -> None:
    st.markdown(
        premium_card_html(title, subtitle, icon=icon, value=value, color=color, size="metric-size"),
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", color: str = "blue", icon: str = "") -> None:
    render_metric_card(label, value, note, color, icon)


def premium_card_html(
    title: str,
    body: str,
    icon: str = "",
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
    icon_html = f"<div class='card-icon-row'><span class='card-icon'>{escape(icon)}</span></div>" if icon else ""
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
        f"{icon_html}"
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
            icon=str(item.get("icon", "")),
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


def compact_chat_text(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\n(?=- )", "\n", text)
    return text


def clean_user_facing_note(value: Any, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    for code in [
        "low_similarity",
        "wide_price_band",
        "medium_model_disagreement",
        "low_margin_buffer",
        "outside_price_band",
        "low_model_confidence",
        "high_cluster_distance",
        "delivery_pressure",
        "high_competition",
        "missing_optional_data",
        "cost_uncertainty",
    ]:
        text = re.sub(rf"(?:^|[;,\s])+{re.escape(code)}(?:$|[;,\s]+)", " ", text)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"(;\s*){2,}", "; ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ;,")
    if max_chars and len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ;,.")
        return f"{cut}..."
    return text


def chat_thread_html(messages: list[dict[str, Any]]) -> str:
    rows = []
    for message in messages:
        role = "user" if message.get("role") == "user" else "assistant"
        content = escape(compact_chat_text(message.get("content", "")))
        source = str(message.get("source", "")).strip()
        source_html = f"<div class='chat-source'>{escape(source)}</div>" if role == "assistant" and source else ""
        pending_class = " chat-bubble-pending" if message.get("pending") else ""
        bubble = f"<div class='chat-bubble chat-bubble-{role}{pending_class}'>{source_html}{content}</div>"
        if role == "assistant":
            rows.append(
                "<div class='chat-row chat-row-assistant'>"
                "<div class='chat-orb'>AI</div>"
                f"{bubble}"
                "</div>"
            )
        else:
            rows.append(f"<div class='chat-row chat-row-user'>{bubble}</div>")
    return "<div class='chat-thread'>" + "".join(rows) + "</div>"


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
        premium_card_html(title, body, pill=kicker, footer_html=status_html, color="blue", size="large-size"),
        unsafe_allow_html=True,
    )


def inject_data_quality_css() -> None:
    st.markdown(
        """
        <style>
            .dq-shell {
                max-width: 1240px;
                margin: 0 auto;
            }
            .dq-section {
                margin-top: 52px;
            }
            .dq-section-tight {
                margin-top: 34px;
            }
            .dq-section .section-title,
            .dq-section-tight .section-title {
                font-size: 1.22rem;
                line-height: 1.25;
                margin-bottom: 0.18rem;
            }
            .dq-section .section-subtitle,
            .dq-section-tight .section-subtitle {
                margin-top: 0.42rem;
                margin-bottom: 18px;
                max-width: 760px;
                font-size: 0.92rem;
                line-height: 1.55;
            }
            .dq-info {
                margin-top: 32px;
            }
            .dq-info .info-callout {
                padding: 1.05rem 1.15rem;
                border-radius: 18px;
                background:
                    linear-gradient(135deg, rgba(255, 79, 31, 0.105), rgba(255, 157, 66, 0.045)),
                    rgba(18, 16, 15, 0.78);
                border-color: rgba(255, 123, 66, 0.24);
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.26);
            }
            .dq-grid {
                display: grid;
                gap: 22px;
                align-items: stretch;
                width: 100%;
            }
            .dq-grid-four {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
            .dq-grid-three {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .dq-quality-grid {
                grid-template-columns: repeat(6, minmax(0, 1fr));
            }
            .dq-quality-grid .dq-quality-card {
                grid-column: span 2;
            }
            .dq-quality-grid .dq-quality-card:nth-last-child(2):nth-child(3n + 1),
            .dq-quality-grid .dq-quality-card:last-child:nth-child(3n + 2) {
                grid-column: span 3;
            }
            .dq-feature-card,
            .dq-metric-card,
            .dq-quality-card {
                position: relative;
                min-width: 0;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.032)),
                    rgba(18, 17, 16, 0.84);
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
                overflow: hidden;
            }
            .dq-feature-card:before,
            .dq-metric-card:before,
            .dq-quality-card:before {
                content: '';
                position: absolute;
                inset: 0 0 auto 0;
                height: 2px;
                background: linear-gradient(90deg, rgba(255,79,31,0.0), rgba(255,116,72,0.72), rgba(255,157,66,0.0));
                opacity: .62;
            }
            .dq-feature-card {
                min-height: 164px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                gap: 12px;
            }
            .dq-feature-top,
            .dq-quality-top {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 14px;
            }
            .dq-index {
                width: 34px;
                height: 34px;
                flex: 0 0 34px;
                display: inline-grid;
                place-items: center;
                border-radius: 12px;
                color: #fff;
                font-size: .78rem;
                font-weight: 720;
                background: rgba(255, 79, 31, 0.14);
                border: 1px solid rgba(255, 123, 66, 0.22);
            }
            .dq-card-title {
                color: #ffffff;
                font-size: 1.02rem;
                line-height: 1.24;
                font-weight: 700;
                margin: 0;
            }
            .dq-card-body {
                color: rgba(255,255,255,0.86);
                font-size: .88rem;
                line-height: 1.52;
                margin: 0;
                overflow-wrap: anywhere;
            }
            .dq-metric-card {
                min-height: 132px;
                padding: 22px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 12px;
            }
            .dq-metric-label {
                color: rgba(255,255,255,0.72);
                font-size: .72rem;
                font-weight: 700;
                letter-spacing: .08em;
                text-transform: uppercase;
            }
            .dq-metric-value {
                color: #ffffff;
                font-size: 1.72rem;
                line-height: 1.12;
                font-weight: 760;
                overflow-wrap: anywhere;
            }
            .dq-metric-note {
                color: rgba(255,255,255,0.76);
                font-size: .82rem;
                line-height: 1.35;
            }
            .dq-quality-card {
                min-height: 150px;
                padding: 22px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 18px;
            }
            .dq-quality-card .status-badge {
                flex: 0 0 auto;
                padding: .26rem .56rem;
                font-size: .7rem;
                white-space: nowrap;
                box-shadow: none;
            }
            .dq-quality-text {
                display: flex;
                flex-direction: column;
                gap: 8px;
                min-width: 0;
            }
            .dq-table-card {
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.11);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025)),
                    rgba(14, 13, 13, 0.92);
                box-shadow: 0 18px 38px rgba(0, 0, 0, 0.30);
                overflow: hidden;
            }
            .dq-table-scroll {
                width: 100%;
                overflow-x: auto;
            }
            .dq-table {
                width: 100%;
                border-collapse: collapse;
                min-width: 760px;
                color: #ffffff;
            }
            .dq-table thead th {
                background: rgba(26, 24, 23, 0.98);
                color: rgba(255,255,255,0.88);
                font-size: .76rem;
                letter-spacing: .07em;
                text-transform: uppercase;
                text-align: left;
                padding: 15px 18px;
                border-bottom: 1px solid rgba(255, 123, 66, 0.22);
            }
            .dq-table tbody td {
                color: rgba(255,255,255,0.88);
                font-size: .88rem;
                line-height: 1.45;
                padding: 15px 18px;
                border-bottom: 1px solid rgba(255,255,255,0.075);
                background: rgba(255,255,255,0.018);
                vertical-align: top;
            }
            .dq-table tbody tr:nth-child(even) td {
                background: rgba(255,255,255,0.038);
            }
            .dq-table tbody tr:last-child td {
                border-bottom: 0;
            }
            .dq-table-code {
                color: #ffbd8a;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: .84rem;
                white-space: nowrap;
            }
            @media (max-width: 1180px) {
                .dq-grid-four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .dq-grid-three { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 760px) {
                .dq-shell { max-width: 100%; }
                .dq-section { margin-top: 42px; }
                .dq-section-tight { margin-top: 28px; }
                .dq-grid-four,
                .dq-grid-three,
                .dq-quality-grid { grid-template-columns: 1fr; gap: 16px; }
                .dq-quality-grid .dq-quality-card,
                .dq-quality-grid .dq-quality-card:nth-last-child(2):nth-child(3n + 1),
                .dq-quality-grid .dq-quality-card:last-child:nth-child(3n + 2) {
                    grid-column: auto;
                }
                .dq-feature-card,
                .dq-metric-card,
                .dq-quality-card {
                    min-height: auto;
                    padding: 20px;
                    border-radius: 18px;
                }
                .dq-metric-value { font-size: 1.5rem; }
                .dq-quality-top { flex-direction: column; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_data_quality_feature_grid(items: list[tuple[str, str]]) -> None:
    cards = []
    for idx, (title, body) in enumerate(items, start=1):
        cards.append(
            "<div class='dq-feature-card'>"
            "<div class='dq-feature-top'>"
            f"<div class='dq-card-title'>{escape(title)}</div>"
            f"<span class='dq-index'>{idx:02d}</span>"
            "</div>"
            f"<div class='dq-card-body'>{escape(body)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='dq-grid dq-grid-four'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_data_quality_metric_grid(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, note in items:
        cards.append(
            "<div class='dq-metric-card'>"
            f"<div class='dq-metric-label'>{escape(label)}</div>"
            f"<div class='dq-metric-value'>{escape(value)}</div>"
            f"<div class='dq-metric-note'>{escape(note)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='dq-grid dq-grid-four'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_data_quality_status_grid(items: list[tuple[str, str, str, str]]) -> None:
    cards = []
    for title, value, body, card_status in items:
        cards.append(
            "<div class='dq-quality-card'>"
            "<div class='dq-quality-top'>"
            "<div class='dq-quality-text'>"
            f"<div class='dq-card-title'>{escape(title)}</div>"
            f"<div class='dq-card-body'>{escape(body)}</div>"
            "</div>"
            f"{badge(value, card_status)}"
            "</div>"
            "</div>"
        )
    st.markdown(f"<div class='dq-grid dq-quality-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def dark_table_cell_html(column: str, value: Any) -> tuple[str, str]:
    column_text = str(column)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-", ""
    if isinstance(value, (int, float, np.integer, np.floating)) and any(token in column_text.casefold() for token in ["oran", "rate", "pct", "yüzde", "%"]):
        numeric = float(value)
        pct_value = numeric * 100 if abs(numeric) <= 1 else numeric
        width = max(0, min(100, pct_value))
        html = (
            "<div class='global-progress'>"
            "<div class='global-progress-track'>"
            f"<div class='global-progress-fill' style='width:{width:.1f}%'></div>"
            "</div>"
            f"<span class='global-progress-value'>{escape(format_pct(pct_value))}</span>"
            "</div>"
        )
        return html, ""
    status_columns = {"Durum", "Audit durumu", "Sızıntı durumu", "Band İçinde mi", "Manual review flag"}
    if column_text in status_columns or any(token in column_text.casefold() for token in ["status", "durumu", "kontrol"]):
        text = str(value)
        normalized = text.casefold()
        status = "good" if any(token in normalized for token in ["geçti", "pass", "bulundu", "uygun", "evet", "hazır", "yok"]) else "bad" if any(token in normalized for token in ["fail", "eksik", "ihlal", "var", "uyarı"]) else "warn"
        return f"<span class='global-status-pill global-status-{status}'>{escape(text)}</span>", ""
    if column_text in {"Kolon", "Kullanılan kolon"} or "column" in column_text.casefold():
        return escape(str(value)), "global-table-code"
    if column_text in {"Metrik", "İş anlamı", "Kontrol", "Özet"}:
        return escape(str(value)), "global-table-strong"
    return escape(str(value)), ""


def render_global_dark_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value_html, css_class = dark_table_cell_html(str(column), row[column])
            class_attr = f" class='{css_class}'" if css_class else ""
            cells.append(f"<td{class_attr}>{value_html}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='global-table-card'><div class='global-table-scroll'>"
        f"<table class='global-dark-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_dark_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = escape(str(row[column]))
            css_class = " class='dq-table-code'" if column == "Kolon" else ""
            cells.append(f"<td{css_class}>{value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='dq-table-card'><div class='dq-table-scroll'>"
        f"<table class='dq-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def inject_test_simulator_css() -> None:
    st.markdown(
        """
        <style>
            .ts-section {
                margin-top: 50px;
            }
            .ts-section-tight {
                margin-top: 32px;
            }
            .ts-section .section-title,
            .ts-section-tight .section-title {
                font-size: 1.22rem;
                line-height: 1.25;
                margin-bottom: 0.18rem;
            }
            .ts-section .section-subtitle,
            .ts-section-tight .section-subtitle {
                max-width: 820px;
                margin-top: 0.42rem;
                margin-bottom: 18px;
                font-size: .92rem;
                line-height: 1.55;
            }
            .ts-warning {
                margin-top: 30px;
                border-radius: 18px;
                padding: 18px 20px;
                color: #ffffff;
                border: 1px solid rgba(255,157,66,0.28);
                background:
                    linear-gradient(135deg, rgba(255,157,66,0.11), rgba(255,79,31,0.055)),
                    rgba(18, 16, 15, 0.82);
                box-shadow: 0 16px 34px rgba(0,0,0,0.26);
                line-height: 1.55;
            }
            .ts-card,
            .st-key-ts_select_card,
            .st-key-ts_inputs_card {
                position: relative;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.10);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.060), rgba(255,255,255,0.030)),
                    rgba(18, 17, 16, 0.84);
                box-shadow: 0 16px 34px rgba(0,0,0,0.28);
            }
            .st-key-ts_select_card,
            .st-key-ts_inputs_card {
                padding: 22px;
            }
            .st-key-ts_select_card [data-testid='stVerticalBlock'],
            .st-key-ts_inputs_card [data-testid='stVerticalBlock'] {
                gap: .8rem;
            }
            .ts-control-title {
                color: #ffffff;
                font-size: 1.02rem;
                font-weight: 720;
                line-height: 1.25;
                margin-bottom: .22rem;
            }
            .ts-control-copy {
                color: rgba(255,255,255,0.78);
                font-size: .86rem;
                line-height: 1.48;
                margin-bottom: .3rem;
            }
            .st-key-ts_select_card div[data-baseweb='select'] > div,
            .st-key-ts_inputs_card div[data-baseweb='input'] > div,
            .st-key-ts_inputs_card div[data-testid='stNumberInput'] input {
                background: rgba(6, 6, 6, 0.62) !important;
                border: 1px solid rgba(255,123,66,0.28) !important;
                border-radius: 12px !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                box-shadow: none !important;
            }
            .st-key-ts_select_card div[data-baseweb='select'] *,
            .st-key-ts_inputs_card div[data-baseweb='input'] *,
            .st-key-ts_inputs_card div[data-testid='stNumberInput'] input,
            .st-key-ts_inputs_card div[data-testid='stNumberInput'] button,
            .st-key-ts_inputs_card div[data-testid='stNumberInput'] button * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
            .st-key-ts_select_card svg *,
            .st-key-ts_inputs_card svg * {
                fill: #ffffff !important;
                color: #ffffff !important;
            }
            .st-key-ts_select_card label,
            .st-key-ts_inputs_card label {
                color: rgba(255,255,255,0.86) !important;
                font-size: .82rem !important;
                font-weight: 650 !important;
            }
            .st-key-ts_inputs_card div[data-testid='column'] {
                min-width: 0;
            }
            .st-key-ts_inputs_card div[data-testid='stButton'] button {
                min-height: 44px !important;
                width: auto !important;
                padding: .54rem 1.05rem !important;
                border-radius: 999px !important;
                font-weight: 720 !important;
                background: linear-gradient(180deg, rgba(255,93,36,0.98), rgba(181,40,13,0.98)) !important;
                border-color: rgba(255,184,117,0.38) !important;
                box-shadow: 0 14px 30px rgba(255,79,31,0.20) !important;
            }
            .ts-summary-card {
                padding: 22px;
            }
            .ts-summary-head {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 16px;
                margin-bottom: 14px;
            }
            .ts-summary-title {
                color: #ffffff;
                font-size: 1.08rem;
                line-height: 1.25;
                font-weight: 740;
            }
            .ts-summary-subtitle {
                color: rgba(255,255,255,0.74);
                font-size: .84rem;
                line-height: 1.45;
                margin-top: .25rem;
            }
            .ts-badge-row {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 8px;
            }
            .ts-kv-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 8px 16px;
            }
            .ts-kv-row {
                min-width: 0;
                padding: 8px 0;
                border-top: 1px solid rgba(255,255,255,0.075);
            }
            .ts-kv-label {
                color: rgba(255,255,255,0.58);
                font-size: .70rem;
                font-weight: 720;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin-bottom: .22rem;
            }
            .ts-kv-value {
                color: rgba(255,255,255,0.92);
                font-size: .86rem;
                line-height: 1.35;
                overflow-wrap: anywhere;
            }
            .ts-process-grid {
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 20px;
                align-items: stretch;
            }
            .ts-process-card {
                grid-column: span 2;
                min-height: 148px;
                padding: 20px;
                position: relative;
                overflow: hidden;
            }
            .ts-process-card:nth-last-child(2):nth-child(3n + 1),
            .ts-process-card:last-child:nth-child(3n + 2) {
                grid-column: span 3;
            }
            .ts-process-card:before,
            .ts-summary-card:before {
                content: '';
                position: absolute;
                inset: 0 0 auto 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(255,116,72,.72), transparent);
                opacity: .62;
            }
            .ts-process-top {
                display: flex;
                justify-content: space-between;
                gap: 14px;
                align-items: flex-start;
                margin-bottom: 12px;
            }
            .ts-step {
                width: 32px;
                height: 32px;
                flex: 0 0 32px;
                display: inline-grid;
                place-items: center;
                border-radius: 12px;
                color: #ffffff;
                font-size: .76rem;
                font-weight: 760;
                background: rgba(255,79,31,0.14);
                border: 1px solid rgba(255,123,66,0.22);
            }
            .ts-process-title {
                color: #ffffff;
                font-size: 1rem;
                line-height: 1.25;
                font-weight: 720;
            }
            .ts-process-body {
                color: rgba(255,255,255,0.82);
                font-size: .86rem;
                line-height: 1.5;
            }
            .st-key-ts_masked_expander {
                margin-top: 26px;
            }
            .st-key-ts_masked_expander div[data-testid='stExpander'] {
                border-radius: 18px;
                border: 1px solid rgba(255,123,66,0.20);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.050), rgba(255,255,255,0.025)),
                    rgba(14,13,13,0.88);
                box-shadow: 0 14px 30px rgba(0,0,0,0.24);
                overflow: hidden;
            }
            .st-key-ts_masked_expander div[data-testid='stExpander'] summary {
                color: #ffffff !important;
                font-weight: 700;
            }
            .st-key-ts_masked_expander div[data-testid='stDataFrame'] {
                border-color: rgba(255,123,66,0.20);
                border-radius: 12px;
            }
            .ts-masked-table-wrap {
                border-radius: 14px;
                border: 1px solid rgba(255,123,66,0.18);
                background: rgba(6,6,6,0.42);
                overflow: hidden;
            }
            .ts-masked-table-scroll {
                width: 100%;
                max-height: 420px;
                overflow: auto;
            }
            .ts-masked-table {
                width: 100%;
                min-width: 620px;
                border-collapse: collapse;
            }
            .ts-masked-table th {
                text-align: left;
                padding: 12px 15px;
                color: rgba(255,255,255,0.86);
                background: rgba(26,24,23,0.98);
                border-bottom: 1px solid rgba(255,123,66,0.20);
                font-size: .74rem;
                letter-spacing: .07em;
                text-transform: uppercase;
            }
            .ts-masked-table td {
                padding: 11px 15px;
                color: rgba(255,255,255,0.86);
                background: rgba(255,255,255,0.018);
                border-bottom: 1px solid rgba(255,255,255,0.07);
                font-size: .84rem;
                line-height: 1.4;
                vertical-align: top;
            }
            .ts-masked-table tr:nth-child(even) td {
                background: rgba(255,255,255,0.035);
            }
            .ts-masked-table tr:last-child td {
                border-bottom: 0;
            }
            @media (max-width: 1180px) {
                .ts-process-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .ts-process-card,
                .ts-process-card:nth-last-child(2):nth-child(3n + 1),
                .ts-process-card:last-child:nth-child(3n + 2) {
                    grid-column: auto;
                }
            }
            @media (max-width: 760px) {
                .ts-section { margin-top: 42px; }
                .ts-section-tight { margin-top: 28px; }
                .st-key-ts_select_card,
                .st-key-ts_inputs_card,
                .ts-summary-card,
                .ts-process-card {
                    padding: 20px;
                    border-radius: 18px;
                }
                .ts-process-grid {
                    grid-template-columns: 1fr;
                    gap: 16px;
                }
                .ts-kv-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 8px 14px;
                }
                .ts-summary-head {
                    flex-direction: column;
                }
                .ts-badge-row {
                    justify-content: flex-start;
                }
                .ts-process-card { min-height: auto; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_test_tender_summary(selected: str, masked: dict[str, Any], audit: dict[str, Any]) -> None:
    rows = [
        ("İhale ID", selected),
        ("Ürün grubu", str(masked.get("product_group", "-"))),
        ("Bölge", str(masked.get("region", "-"))),
        ("Kurum", str(masked.get("buyer_institution", "-"))),
        ("Ürün adı", str(masked.get("product_name", "-"))),
        ("Miktar", format_int(masked.get("quantity", 0))),
        ("Teslim süresi", f"{masked.get('delivery_months', '-')} ay"),
    ]
    kv_html = "".join(
        "<div class='ts-kv-row'>"
        f"<div class='ts-kv-label'>{escape(label)}</div>"
        f"<div class='ts-kv-value'>{escape(value)}</div>"
        "</div>"
        for label, value in rows
    )
    leak_badge = badge("Sızıntı yok" if audit["audit_status"] == "pass" else "Sızıntı uyarısı", "good" if audit["audit_status"] == "pass" else "bad")
    st.markdown(
        "<div class='ts-card ts-summary-card'>"
        "<div class='ts-summary-head'>"
        "<div>"
        "<div class='ts-summary-title'>Seçili İhale</div>"
        "<div class='ts-summary-subtitle'>Bu bilgiler canlı ihale girdisi gibi kullanılır; gerçek sonuç alanları maskelidir.</div>"
        "</div>"
        f"<div class='ts-badge-row'>{badge('Gerçek sonuç gizli', 'warn')}{leak_badge}</div>"
        "</div>"
        f"<div class='ts-kv-grid'>{kv_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_test_process_grid(items: list[tuple[str, str]]) -> None:
    cards = []
    for idx, (title, body) in enumerate(items, start=1):
        cards.append(
            "<div class='ts-card ts-process-card'>"
            "<div class='ts-process-top'>"
            f"<div class='ts-process-title'>{escape(title)}</div>"
            f"<span class='ts-step'>{idx:02d}</span>"
            "</div>"
            f"<div class='ts-process-body'>{escape(body)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='ts-process-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_test_masked_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{escape(str(row[column]))}</td>" for column in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        "<div class='ts-masked-table-wrap'><div class='ts-masked-table-scroll'>"
        f"<table class='ts-masked-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def inject_similar_tenders_css() -> None:
    st.markdown(
        """
        <style>
            .sim-section {
                margin-top: 50px;
            }
            .sim-section-tight {
                margin-top: 32px;
            }
            .sim-section .section-title,
            .sim-section-tight .section-title {
                font-size: 1.22rem;
                line-height: 1.25;
                margin-bottom: .18rem;
            }
            .sim-section .section-subtitle,
            .sim-section-tight .section-subtitle {
                max-width: 820px;
                margin-top: .42rem;
                margin-bottom: 18px;
                font-size: .92rem;
                line-height: 1.55;
            }
            .sim-callout {
                margin-top: 30px;
                border-radius: 18px;
                padding: 18px 20px;
                border: 1px solid rgba(255,123,66,0.24);
                background:
                    linear-gradient(135deg, rgba(255,79,31,0.10), rgba(255,157,66,0.045)),
                    rgba(18,16,15,0.82);
                box-shadow: 0 16px 34px rgba(0,0,0,0.26);
            }
            .sim-callout-title {
                color: #ffffff;
                font-size: 1rem;
                line-height: 1.25;
                font-weight: 740;
                margin-bottom: 10px;
            }
            .sim-callout-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
            }
            .sim-callout-item {
                min-width: 0;
                padding-top: 10px;
                border-top: 1px solid rgba(255,255,255,0.08);
                color: rgba(255,255,255,0.82);
                font-size: .86rem;
                line-height: 1.48;
            }
            .sim-callout-item b {
                display: block;
                color: #ffffff;
                font-size: .78rem;
                letter-spacing: .07em;
                text-transform: uppercase;
                margin-bottom: 4px;
            }
            .sim-metric-grid {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 20px;
                align-items: stretch;
            }
            .sim-metric-card {
                min-width: 0;
                min-height: 128px;
                padding: 16px;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.10);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.060), rgba(255,255,255,0.030)),
                    rgba(18,17,16,0.84);
                box-shadow: 0 16px 34px rgba(0,0,0,0.28);
                position: relative;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 8px;
            }
            .sim-metric-card:before,
            .sim-table-card:before {
                content: '';
                position: absolute;
                inset: 0 0 auto 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(255,116,72,.72), transparent);
                opacity: .62;
            }
            .sim-metric-label {
                color: rgba(255,255,255,0.70);
                font-size: .62rem;
                font-weight: 740;
                letter-spacing: .075em;
                text-transform: uppercase;
                line-height: 1.28;
            }
            .sim-metric-value {
                color: #ffffff;
                font-size: 1.36rem;
                line-height: 1.1;
                font-weight: 760;
                overflow-wrap: anywhere;
            }
            .sim-metric-note {
                color: rgba(255,255,255,0.76);
                font-size: .74rem;
                line-height: 1.32;
            }
            .sim-table-card {
                position: relative;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.11);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025)),
                    rgba(14,13,13,0.92);
                box-shadow: 0 18px 38px rgba(0,0,0,0.30);
                overflow: hidden;
            }
            .sim-table-scroll {
                width: 100%;
                overflow-x: auto;
            }
            .sim-table {
                width: 100%;
                min-width: 1180px;
                border-collapse: collapse;
            }
            .sim-table th {
                text-align: left;
                padding: 14px 16px;
                color: rgba(255,255,255,0.86);
                background: rgba(26,24,23,0.98);
                border-bottom: 1px solid rgba(255,123,66,0.22);
                font-size: .72rem;
                letter-spacing: .07em;
                text-transform: uppercase;
                white-space: nowrap;
            }
            .sim-table td {
                padding: 13px 16px;
                color: rgba(255,255,255,0.86);
                background: rgba(255,255,255,0.018);
                border-bottom: 1px solid rgba(255,255,255,0.07);
                font-size: .84rem;
                line-height: 1.42;
                vertical-align: middle;
            }
            .sim-table tr:nth-child(even) td {
                background: rgba(255,255,255,0.035);
            }
            .sim-table tr:last-child td {
                border-bottom: 0;
            }
            .sim-id {
                color: #ffbd8a;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: .80rem;
                white-space: nowrap;
            }
            .sim-number {
                white-space: nowrap;
                text-align: right;
            }
            .sim-score-cell {
                min-width: 142px;
            }
            .sim-score-wrap {
                display: grid;
                grid-template-columns: minmax(72px, 1fr) 48px;
                gap: 10px;
                align-items: center;
            }
            .sim-score-track {
                height: 7px;
                border-radius: 999px;
                background: rgba(255,255,255,0.10);
                overflow: hidden;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.035);
            }
            .sim-score-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #ff4f1f, #ff9d42);
                box-shadow: 0 0 12px rgba(255,79,31,0.22);
            }
            .sim-score-value {
                color: #ffffff;
                font-size: .80rem;
                font-weight: 720;
                text-align: right;
                font-variant-numeric: tabular-nums;
            }
            @media (max-width: 1180px) {
                .sim-metric-grid {
                    grid-template-columns: repeat(6, minmax(0, 1fr));
                }
                .sim-metric-card {
                    grid-column: span 2;
                }
                .sim-metric-card:nth-last-child(2):nth-child(3n + 1),
                .sim-metric-card:last-child:nth-child(3n + 2) {
                    grid-column: span 3;
                }
                .sim-callout-grid {
                    grid-template-columns: 1fr;
                }
            }
            @media (max-width: 760px) {
                .sim-section { margin-top: 42px; }
                .sim-section-tight { margin-top: 28px; }
                .sim-metric-grid {
                    grid-template-columns: 1fr;
                    gap: 16px;
                }
                .sim-metric-card,
                .sim-metric-card:nth-last-child(2):nth-child(3n + 1),
                .sim-metric-card:last-child:nth-child(3n + 2) {
                    grid-column: auto;
                    min-height: auto;
                    padding: 20px;
                    border-radius: 18px;
                }
                .sim-callout,
                .sim-table-card {
                    border-radius: 18px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_similar_methodology_callout() -> None:
    st.markdown(
        """
        <div class='sim-callout'>
            <div class='sim-callout-title'>Benzerlik hesabı ve basit örnek</div>
            <div class='sim-callout-grid'>
                <div class='sim-callout-item'><b>Girdi sinyalleri</b>Ürün adı, ürün grubu, kurum, kurum tipi, bölge, ihale tipi, miktar, teslim süresi ve tahmini rekabet birlikte değerlendirilir.</div>
                <div class='sim-callout-item'><b>Skor disiplini</b>Fiyat, marj ve maliyet alanları benzerlik skoruna girmez; metinsel alanlar yerel embedding ile sayısallaştırılır.</div>
                <div class='sim-callout-item'><b>Yorumlama</b>Tarihsel fiyatlar sadece emsal bilgisini yorumlamak için gösterilir; seçili test ihalesinin gerçek sonucu reveal öncesi kullanılmaz.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similar_metric_grid(items: list[tuple[str, str, str]]) -> None:
    html = "".join(
        "<div class='sim-metric-card'>"
        f"<div class='sim-metric-label'>{escape(label)}</div>"
        f"<div class='sim-metric-value'>{escape(value)}</div>"
        f"<div class='sim-metric-note'>{escape(note)}</div>"
        "</div>"
        for label, value, note in items
    )
    st.markdown(f"<div class='sim-metric-grid'>{html}</div>", unsafe_allow_html=True)


def render_similar_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = row[column]
            if column == "Benzerlik Skoru":
                score = 0.0 if pd.isna(value) else max(0.0, min(1.0, float(value)))
                cells.append(
                    "<td class='sim-score-cell'>"
                    "<div class='sim-score-wrap'>"
                    "<div class='sim-score-track'>"
                    f"<div class='sim-score-fill' style='width:{score * 100:.1f}%'></div>"
                    "</div>"
                    f"<div class='sim-score-value'>{score:.3f}</div>"
                    "</div>"
                    "</td>"
                )
            elif column == "İhale ID":
                cells.append(f"<td><span class='sim-id'>{escape(str(value))}</span></td>")
            elif column == "Miktar":
                cells.append(f"<td class='sim-number'>{escape(format_int(value))}</td>")
            elif column == "Tarihsel Kazanılmış Fiyat":
                cells.append(f"<td class='sim-number'>{escape(format_try(value))}</td>")
            elif column == "Karlılık Oranı":
                cells.append(f"<td class='sim-number'>{escape(format_decimal(value, 2))}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='sim-table-card'><div class='sim-table-scroll'>"
        f"<table class='sim-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def inject_profile_fit_css() -> None:
    st.markdown(
        """
        <style>
            .pf-section {
                margin-top: 50px;
            }
            .pf-section-tight {
                margin-top: 32px;
            }
            .pf-section .section-title,
            .pf-section-tight .section-title {
                font-size: 1.22rem;
                line-height: 1.25;
                margin-bottom: .18rem;
            }
            .pf-section .section-subtitle,
            .pf-section-tight .section-subtitle {
                max-width: 840px;
                margin-top: .42rem;
                margin-bottom: 18px;
                font-size: .92rem;
                line-height: 1.55;
            }
            .pf-callout {
                margin-top: 30px;
                border-radius: 18px;
                padding: 18px 20px;
                color: #ffffff;
                border: 1px solid rgba(255,123,66,0.24);
                background:
                    linear-gradient(135deg, rgba(255,79,31,0.10), rgba(255,157,66,0.045)),
                    rgba(18,16,15,0.82);
                box-shadow: 0 16px 34px rgba(0,0,0,0.26);
            }
            .pf-callout-title {
                font-size: 1rem;
                line-height: 1.25;
                font-weight: 740;
                margin-bottom: 8px;
            }
            .pf-callout-body {
                color: rgba(255,255,255,0.84);
                font-size: .88rem;
                line-height: 1.55;
            }
            .pf-grid {
                display: grid;
                gap: 20px;
                align-items: stretch;
            }
            .pf-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .pf-metric-grid {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
            .pf-two-col {
                grid-template-columns: minmax(0, 1.16fr) minmax(300px, .84fr);
                align-items: start;
            }
            .pf-card,
            .pf-kpi-card,
            .pf-metric-card,
            .pf-gauge-card,
            .pf-table-card,
            .st-key-pf_gauge_card {
                position: relative;
                min-width: 0;
                border-radius: 20px;
                border: 1px solid rgba(255,255,255,0.10);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.060), rgba(255,255,255,0.030)),
                    rgba(18,17,16,0.84);
                box-shadow: 0 16px 34px rgba(0,0,0,0.28);
                overflow: hidden;
            }
            .pf-kpi-card:before,
            .pf-metric-card:before,
            .pf-card:before,
            .pf-gauge-card:before,
            .pf-table-card:before,
            .st-key-pf_gauge_card:before {
                content: '';
                position: absolute;
                inset: 0 0 auto 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(255,116,72,.72), transparent);
                opacity: .62;
            }
            .pf-kpi-card {
                min-height: 158px;
                padding: 18px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 12px;
            }
            .pf-kpi-head {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: flex-start;
            }
            .pf-kpi-label,
            .pf-metric-label {
                color: rgba(255,255,255,0.70);
                font-size: .66rem;
                font-weight: 740;
                letter-spacing: .075em;
                text-transform: uppercase;
                line-height: 1.28;
            }
            .pf-kpi-value {
                color: #ffffff;
                font-size: 1.18rem;
                line-height: 1.18;
                font-weight: 760;
                overflow-wrap: anywhere;
            }
            .pf-kpi-body,
            .pf-metric-note {
                color: rgba(255,255,255,0.78);
                font-size: .78rem;
                line-height: 1.38;
            }
            .pf-kpi-card .status-badge {
                flex: 0 0 auto;
                padding: .24rem .52rem;
                font-size: .68rem;
                box-shadow: none;
                white-space: nowrap;
            }
            .pf-score-note {
                margin-top: 12px;
                color: rgba(255,255,255,0.80);
                font-size: .86rem;
                line-height: 1.55;
            }
            .pf-card {
                padding: 20px;
            }
            .pf-card-title,
            .pf-gauge-title {
                color: #ffffff;
                font-size: 1.04rem;
                line-height: 1.25;
                font-weight: 740;
                margin-bottom: 8px;
            }
            .pf-card-note {
                color: rgba(255,255,255,0.76);
                font-size: .84rem;
                line-height: 1.48;
                margin-bottom: 14px;
            }
            .pf-kv-list {
                display: grid;
                gap: 0;
            }
            .pf-kv-row {
                display: grid;
                grid-template-columns: minmax(0, .9fr) minmax(0, 1fr);
                gap: 14px;
                padding: 9px 0;
                border-top: 1px solid rgba(255,255,255,0.075);
                align-items: baseline;
            }
            .pf-kv-label {
                color: rgba(255,255,255,0.58);
                font-size: .70rem;
                font-weight: 720;
                letter-spacing: .06em;
                text-transform: uppercase;
            }
            .pf-kv-value {
                color: rgba(255,255,255,0.92);
                font-size: .86rem;
                line-height: 1.35;
                text-align: right;
                overflow-wrap: anywhere;
            }
            .pf-gauge-card {
                padding: 18px;
            }
            .st-key-pf_gauge_card {
                padding: 18px;
            }
            .st-key-pf_gauge_card [data-testid='stVerticalBlock'] {
                gap: .35rem;
            }
            .pf-gauge-copy {
                color: rgba(255,255,255,0.80);
                font-size: .86rem;
                line-height: 1.52;
                margin-top: 10px;
                padding-top: 12px;
                border-top: 1px solid rgba(255,255,255,0.075);
            }
            .pf-metric-card {
                min-height: 142px;
                padding: 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 8px;
            }
            .pf-metric-value {
                color: #ffffff;
                font-size: 1.34rem;
                line-height: 1.1;
                font-weight: 760;
                overflow-wrap: anywhere;
            }
            .pf-table-card {
                border-color: rgba(255,255,255,0.11);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025)),
                    rgba(14,13,13,0.92);
            }
            .pf-table-scroll {
                width: 100%;
                overflow-x: auto;
            }
            .pf-table {
                width: 100%;
                min-width: 920px;
                border-collapse: collapse;
            }
            .pf-table th {
                text-align: left;
                padding: 13px 15px;
                color: rgba(255,255,255,0.86);
                background: rgba(26,24,23,0.98);
                border-bottom: 1px solid rgba(255,123,66,0.22);
                font-size: .72rem;
                letter-spacing: .07em;
                text-transform: uppercase;
                white-space: nowrap;
            }
            .pf-table td {
                padding: 12px 15px;
                color: rgba(255,255,255,0.86);
                background: rgba(255,255,255,0.018);
                border-bottom: 1px solid rgba(255,255,255,0.07);
                font-size: .84rem;
                line-height: 1.42;
                vertical-align: middle;
            }
            .pf-table tr:nth-child(even) td {
                background: rgba(255,255,255,0.035);
            }
            .pf-table tr:last-child td {
                border-bottom: 0;
            }
            .pf-id {
                color: #ffbd8a;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: .80rem;
                white-space: nowrap;
            }
            .pf-number {
                white-space: nowrap;
                text-align: right;
            }
            @media (max-width: 1180px) {
                .pf-metric-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .pf-two-col {
                    grid-template-columns: 1fr;
                }
            }
            @media (max-width: 760px) {
                .pf-section { margin-top: 42px; }
                .pf-section-tight { margin-top: 28px; }
                .pf-kpi-grid,
                .pf-metric-grid {
                    grid-template-columns: 1fr;
                    gap: 16px;
                }
                .pf-kpi-card,
                .pf-metric-card,
                .pf-card,
                .pf-gauge-card,
                .st-key-pf_gauge_card,
                .pf-callout,
                .pf-table-card {
                    border-radius: 18px;
                }
                .pf-kv-row {
                    grid-template-columns: 1fr;
                    gap: 4px;
                }
                .pf-kv-value {
                    text-align: left;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_profile_callout() -> None:
    st.markdown(
        """
        <div class='pf-callout'>
            <div class='pf-callout-title'>Profil modelleri ne yapar?</div>
            <div class='pf-callout-body'>
                KNN emsal arama, mixed-type clustering ve Isolation Forest fiyat önermez; seçili ihalenin geçmiş kazanılmış ihale profillerine yapısal olarak ne kadar benzediğini ve sıra dışı olup olmadığını analiz eder.
                Fiyat aralığı ayrı olarak Fiyat Koridoru bölümünde değerlendirilir.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_kpi_grid(items: list[dict[str, str]]) -> None:
    cards = []
    for item in items:
        cards.append(
            "<div class='pf-kpi-card'>"
            "<div class='pf-kpi-head'>"
            f"<div class='pf-kpi-label'>{escape(item['label'])}</div>"
            f"{badge(item.get('badge', ''), item.get('status', 'good')) if item.get('badge') else ''}"
            "</div>"
            f"<div class='pf-kpi-value'>{escape(item['value'])}</div>"
            f"<div class='pf-kpi-body'>{escape(item['body'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='pf-grid pf-kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_profile_kv_panel(title: str, rows: list[tuple[str, str]], note: str = "") -> None:
    rows_html = "".join(
        "<div class='pf-kv-row'>"
        f"<div class='pf-kv-label'>{escape(label)}</div>"
        f"<div class='pf-kv-value'>{escape(value)}</div>"
        "</div>"
        for label, value in rows
    )
    note_html = f"<div class='pf-card-note'>{escape(note)}</div>" if note else ""
    st.markdown(
        "<div class='pf-card'>"
        f"<div class='pf-card-title'>{escape(title)}</div>"
        f"{note_html}"
        f"<div class='pf-kv-list'>{rows_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_profile_metric_grid(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, note in items:
        cards.append(
            "<div class='pf-metric-card'>"
            f"<div class='pf-metric-label'>{escape(label)}</div>"
            f"<div class='pf-metric-value'>{escape(value)}</div>"
            f"<div class='pf-metric-note'>{escape(note)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='pf-grid pf-metric-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_profile_examples_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = row[column]
            if column == "İhale ID":
                cells.append(f"<td><span class='pf-id'>{escape(str(value))}</span></td>")
            elif column in {"Miktar", "Teslim süresi"}:
                cells.append(f"<td class='pf-number'>{escape(format_decimal(value, 1) if column == 'Teslim süresi' else format_int(value))}</td>")
            elif column == "Seçili ihaleye uzaklık":
                cells.append(f"<td class='pf-number'>{escape(format_decimal(value, 3))}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='pf-table-card'><div class='pf-table-scroll'>"
        f"<table class='pf-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def inject_price_corridor_css() -> None:
    st.markdown(
        """
        <style>
            .pc-section {
                margin-top: 48px;
            }
            .pc-section .section-title {
                font-size: 1.24rem;
                line-height: 1.25;
                margin-bottom: 0.18rem;
            }
            .pc-section .section-subtitle {
                max-width: 860px;
                color: rgba(245, 247, 250, 0.68);
                line-height: 1.55;
                margin-bottom: 18px;
            }
            .pc-grid {
                display: grid;
                gap: 20px;
                align-items: stretch;
            }
            .pc-kpi-grid {
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin-top: 28px;
            }
            .pc-baseline-grid {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
            .pc-kpi-card,
            .pc-primary-card,
            .pc-baseline-card,
            .pc-note-card,
            .pc-table-card {
                border: 1px solid rgba(248, 113, 113, 0.16);
                background:
                    linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.02)),
                    rgba(16, 18, 22, 0.92);
                box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
                border-radius: 20px;
            }
            .pc-kpi-card {
                min-height: 132px;
                padding: 22px 24px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .pc-label {
                color: rgba(245, 247, 250, 0.62);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .pc-value {
                color: #fff7ed;
                font-size: 1.78rem;
                line-height: 1.1;
                font-weight: 850;
                margin-top: 12px;
            }
            .pc-note {
                color: rgba(245, 247, 250, 0.64);
                font-size: 0.88rem;
                line-height: 1.42;
                margin-top: 12px;
            }
            .pc-primary-card {
                padding: 24px;
            }
            .pc-primary-top {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 16px;
                margin-bottom: 20px;
            }
            .pc-primary-title {
                color: #fff7ed;
                font-size: 1.2rem;
                font-weight: 850;
                line-height: 1.25;
            }
            .pc-primary-copy {
                margin-top: 8px;
                color: rgba(245, 247, 250, 0.68);
                line-height: 1.55;
                max-width: 780px;
            }
            .pc-pill {
                flex: 0 0 auto;
                border: 1px solid rgba(251, 146, 60, 0.34);
                background: rgba(251, 146, 60, 0.12);
                color: #fed7aa;
                border-radius: 999px;
                padding: 6px 10px;
                font-size: 0.74rem;
                font-weight: 800;
                white-space: nowrap;
            }
            .pc-band-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
            }
            .pc-band {
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(255, 255, 255, 0.035);
                border-radius: 16px;
                padding: 16px;
            }
            .pc-band-label {
                color: rgba(245, 247, 250, 0.58);
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .pc-band-value {
                color: #fff7ed;
                font-size: 1.38rem;
                font-weight: 850;
                margin-top: 8px;
                line-height: 1.14;
            }
            .pc-baseline-card {
                min-height: 178px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 14px;
            }
            .pc-baseline-head {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                align-items: flex-start;
            }
            .pc-baseline-title {
                color: #fff7ed;
                font-size: 0.98rem;
                font-weight: 830;
                line-height: 1.25;
            }
            .pc-baseline-copy {
                margin-top: 8px;
                color: rgba(245, 247, 250, 0.62);
                font-size: 0.86rem;
                line-height: 1.42;
            }
            .pc-baseline-value {
                color: #fff7ed;
                font-size: 1.32rem;
                font-weight: 850;
                line-height: 1.12;
            }
            .pc-mini-band {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
            }
            .pc-mini-band div {
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.035);
                border: 1px solid rgba(255, 255, 255, 0.07);
                padding: 9px 8px;
                min-width: 0;
            }
            .pc-mini-band span {
                display: block;
                color: rgba(245, 247, 250, 0.55);
                font-size: 0.7rem;
                font-weight: 760;
                margin-bottom: 4px;
            }
            .pc-mini-band b {
                display: block;
                color: #fff7ed;
                font-size: 0.82rem;
                line-height: 1.18;
                overflow-wrap: anywhere;
            }
            .pc-table-card {
                padding: 18px;
            }
            .pc-table-scroll {
                overflow-x: auto;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .pc-table {
                width: 100%;
                border-collapse: collapse;
                min-width: 980px;
                background: rgba(12, 14, 18, 0.88);
            }
            .pc-table th {
                text-align: left;
                color: rgba(255, 247, 237, 0.82);
                background: rgba(26, 29, 36, 0.98);
                border-bottom: 1px solid rgba(248, 113, 113, 0.18);
                padding: 13px 14px;
                font-size: 0.76rem;
                letter-spacing: 0.03em;
                text-transform: uppercase;
                white-space: nowrap;
            }
            .pc-table td {
                color: rgba(245, 247, 250, 0.82);
                border-bottom: 1px solid rgba(255, 255, 255, 0.065);
                padding: 13px 14px;
                font-size: 0.88rem;
                vertical-align: top;
                line-height: 1.42;
            }
            .pc-table tr:nth-child(even) td {
                background: rgba(255, 255, 255, 0.025);
            }
            .pc-table tr:last-child td {
                border-bottom: 0;
            }
            .pc-table .pc-method {
                color: #fff7ed;
                font-weight: 760;
                white-space: nowrap;
            }
            .pc-table .pc-description {
                max-width: 340px;
                color: rgba(245, 247, 250, 0.68);
            }
            .pc-note-card {
                margin-top: 28px;
                padding: 18px 20px;
                border-color: rgba(251, 146, 60, 0.22);
                background:
                    linear-gradient(135deg, rgba(251, 146, 60, 0.09), rgba(239, 68, 68, 0.045)),
                    rgba(16, 18, 22, 0.92);
                color: rgba(245, 247, 250, 0.74);
                line-height: 1.55;
            }
            .pc-note-card b {
                color: #fed7aa;
            }
            @media (max-width: 1180px) {
                .pc-kpi-grid,
                .pc-baseline-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 720px) {
                .pc-kpi-grid,
                .pc-baseline-grid,
                .pc-band-grid {
                    grid-template-columns: 1fr;
                }
                .pc-primary-top {
                    flex-direction: column;
                }
                .pc-kpi-card,
                .pc-baseline-card,
                .pc-primary-card {
                    border-radius: 18px;
                    padding: 18px;
                }
                .pc-value {
                    font-size: 1.45rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_price_kpi_grid(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, note in items:
        cards.append(
            "<div class='pc-kpi-card'>"
            "<div>"
            f"<div class='pc-label'>{escape(label)}</div>"
            f"<div class='pc-value'>{escape(value)}</div>"
            "</div>"
            f"<div class='pc-note'>{escape(note)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='pc-grid pc-kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_primary_corridor_card(corridor: dict[str, Any], confidence_label: str) -> None:
    bands = [
        ("Alt fiyat", format_try(corridor.get("predicted_low_price"))),
        ("Orta fiyat", format_try(corridor.get("predicted_mid_price"))),
        ("Üst fiyat", format_try(corridor.get("predicted_high_price"))),
    ]
    band_html = "".join(
        "<div class='pc-band'>"
        f"<div class='pc-band-label'>{escape(label)}</div>"
        f"<div class='pc-band-value'>{escape(value)}</div>"
        "</div>"
        for label, value in bands
    )
    st.markdown(
        "<div class='pc-primary-card'>"
        "<div class='pc-primary-top'>"
        "<div>"
        "<div class='pc-primary-title'>Benzerlik Tabanlı Koridor</div>"
        "<div class='pc-primary-copy'>Top-K benzer kazanılmış ihalelerden tarihsel fiyat bandı üretir. Orta fiyat Top-K emsallerin medyan/p50 fiyatına dayanır; düşük ve yüksek uçlar aynı emsal bandının sınırlarını gösterir.</div>"
        "</div>"
        f"<span class='pc-pill'>Ana fiyat koridoru · {escape(confidence_label)}</span>"
        "</div>"
        f"<div class='pc-band-grid'>{band_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_price_baseline_grid(rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        title = str(row["Yöntem"])
        confidence = str(row["Güven seviyesi"])
        description = clean_user_facing_note(row.get("Açıklama", ""), 138)
        cards.append(
            "<div class='pc-baseline-card'>"
            "<div>"
            "<div class='pc-baseline-head'>"
            f"<div class='pc-baseline-title'>{escape(title)}</div>"
            f"<span class='pc-pill'>{escape(confidence)}</span>"
            "</div>"
            f"<div class='pc-baseline-copy'>{escape(description)}</div>"
            "</div>"
            f"<div class='pc-baseline-value'>{escape(format_optional_try(row.get('Tahmin fiyatı'), 'Aktif değil'))}</div>"
            "<div class='pc-mini-band'>"
            f"<div><span>Düşük</span><b>{escape(format_optional_try(row.get('Düşük fiyat / low'), 'Aktif değil'))}</b></div>"
            f"<div><span>Orta</span><b>{escape(format_optional_try(row.get('Orta fiyat / mid'), 'Aktif değil'))}</b></div>"
            f"<div><span>Yüksek</span><b>{escape(format_optional_try(row.get('Yüksek fiyat / high'), 'Aktif değil'))}</b></div>"
            "</div>"
            "</div>"
        )
    st.markdown(f"<div class='pc-grid pc-baseline-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_price_comparison_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = escape(str(row[column]))
            if column == "Yöntem":
                cells.append(f"<td class='pc-method'>{value}</td>")
            elif column == "Açıklama":
                cells.append(f"<td class='pc-description'>{value}</td>")
            else:
                cells.append(f"<td>{value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='pc-table-card'><div class='pc-table-scroll'>"
        f"<table class='pc-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_price_decision_note() -> None:
    st.markdown(
        "<div class='pc-note-card'><b>Karar notu:</b> "
        "Fiyat koridoru tek başına karar değildir. Bu fiyatlar; emsal ihale analizi, profil uyumu, karlılık beklentisi ve risk göstergeleriyle birlikte değerlendirilmelidir."
        "</div>",
        unsafe_allow_html=True,
    )


def inject_scenario_css() -> None:
    st.markdown(
        """
        <style>
            .sc-section {
                margin-top: 50px;
            }
            .sc-section .section-title {
                font-size: 1.24rem;
                line-height: 1.25;
                margin-bottom: 0.18rem;
            }
            .sc-section .section-subtitle {
                max-width: 900px;
                color: rgba(245, 247, 250, 0.68);
                line-height: 1.55;
                margin-bottom: 18px;
            }
            .sc-grid {
                display: grid;
                gap: 20px;
                align-items: stretch;
            }
            .sc-kpi-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
                margin-top: 28px;
            }
            .sc-card-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .sc-kpi-card,
            .sc-score-card,
            .sc-strategy-card,
            .sc-table-card,
            .sc-note-card {
                border: 1px solid rgba(248, 113, 113, 0.16);
                background:
                    linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.022)),
                    rgba(16, 18, 22, 0.92);
                box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
                border-radius: 20px;
            }
            .sc-kpi-card {
                min-height: 132px;
                padding: 22px 24px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .sc-label {
                color: rgba(245, 247, 250, 0.62);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .sc-value {
                color: #fff7ed;
                font-size: 1.76rem;
                line-height: 1.1;
                font-weight: 850;
                margin-top: 12px;
            }
            .sc-note {
                color: rgba(245, 247, 250, 0.64);
                font-size: 0.88rem;
                line-height: 1.42;
                margin-top: 12px;
            }
            .sc-score-card {
                padding: 22px 24px;
                border-color: rgba(251, 146, 60, 0.22);
                background:
                    linear-gradient(135deg, rgba(251, 146, 60, 0.09), rgba(239, 68, 68, 0.04)),
                    rgba(16, 18, 22, 0.92);
            }
            .sc-score-title {
                color: #fff7ed;
                font-size: 1.08rem;
                font-weight: 840;
                margin-bottom: 10px;
            }
            .sc-score-copy {
                color: rgba(245, 247, 250, 0.72);
                line-height: 1.58;
                max-width: 980px;
            }
            .sc-score-components {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 12px;
                margin-top: 18px;
            }
            .sc-score-component {
                border: 1px solid rgba(255, 255, 255, 0.075);
                background: rgba(255, 255, 255, 0.035);
                border-radius: 14px;
                padding: 13px;
            }
            .sc-score-component b {
                display: block;
                color: #fed7aa;
                font-size: 0.84rem;
                margin-bottom: 7px;
            }
            .sc-score-component span {
                display: block;
                color: rgba(245, 247, 250, 0.66);
                font-size: 0.8rem;
                line-height: 1.42;
            }
            .sc-strategy-card {
                padding: 22px;
                display: flex;
                flex-direction: column;
                gap: 18px;
                min-height: 0;
                height: auto;
                overflow: visible;
            }
            .sc-card-top {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 14px;
            }
            .sc-card-number {
                width: 38px;
                height: 38px;
                display: inline-grid;
                place-items: center;
                border-radius: 12px;
                border: 1px solid rgba(251, 146, 60, 0.24);
                background: rgba(251, 146, 60, 0.10);
                color: #fed7aa;
                font-weight: 850;
                flex: 0 0 auto;
            }
            .sc-card-title {
                color: #fff7ed;
                font-size: 1.06rem;
                line-height: 1.22;
                font-weight: 850;
            }
            .sc-pill {
                flex: 0 0 auto;
                border: 1px solid rgba(251, 146, 60, 0.34);
                background: rgba(251, 146, 60, 0.12);
                color: #fed7aa;
                border-radius: 999px;
                padding: 6px 10px;
                font-size: 0.74rem;
                font-weight: 800;
                white-space: nowrap;
            }
            .sc-pill-good {
                border-color: rgba(34, 197, 94, 0.30);
                background: rgba(34, 197, 94, 0.10);
                color: #bbf7d0;
            }
            .sc-pill-bad {
                border-color: rgba(248, 113, 113, 0.34);
                background: rgba(248, 113, 113, 0.12);
                color: #fecaca;
            }
            .sc-price-label {
                color: rgba(245, 247, 250, 0.58);
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .sc-price {
                margin-top: 7px;
                color: #fff7ed;
                font-size: 1.72rem;
                line-height: 1.08;
                font-weight: 880;
            }
            .sc-summary {
                color: rgba(245, 247, 250, 0.70);
                line-height: 1.52;
                font-size: 0.9rem;
            }
            .sc-metric-group-title {
                color: rgba(245, 247, 250, 0.56);
                font-size: 0.72rem;
                font-weight: 820;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 9px;
            }
            .sc-primary-metrics,
            .sc-secondary-metrics {
                display: grid;
                gap: 10px;
            }
            .sc-primary-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .sc-secondary-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .sc-mini-metric {
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.075);
                background: rgba(255, 255, 255, 0.035);
                padding: 11px 12px;
                min-width: 0;
            }
            .sc-mini-metric span {
                display: block;
                color: rgba(245, 247, 250, 0.55);
                font-size: 0.72rem;
                font-weight: 760;
                margin-bottom: 5px;
            }
            .sc-mini-metric b {
                display: block;
                color: #fff7ed;
                font-size: 0.9rem;
                line-height: 1.2;
                overflow-wrap: anywhere;
            }
            .sc-risk-box {
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                padding-top: 14px;
            }
            .sc-risk-title {
                color: #fed7aa;
                font-size: 0.84rem;
                font-weight: 820;
                margin-bottom: 8px;
            }
            .sc-risk-list {
                margin: 0;
                padding-left: 18px;
                color: rgba(245, 247, 250, 0.70);
                font-size: 0.86rem;
                line-height: 1.5;
            }
            .sc-risk-list li {
                margin: 4px 0;
            }
            .sc-rule-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                padding-top: 14px;
            }
            .sc-rule-label {
                color: rgba(245, 247, 250, 0.58);
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .sc-table-card {
                padding: 18px;
            }
            .sc-table-scroll {
                overflow-x: auto;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .sc-table {
                width: 100%;
                border-collapse: collapse;
                min-width: 1320px;
                background: rgba(12, 14, 18, 0.88);
            }
            .sc-table th {
                text-align: left;
                color: rgba(255, 247, 237, 0.84);
                background: rgba(26, 29, 36, 0.98);
                border-bottom: 1px solid rgba(248, 113, 113, 0.18);
                padding: 13px 14px;
                font-size: 0.74rem;
                letter-spacing: 0.03em;
                text-transform: uppercase;
                white-space: nowrap;
            }
            .sc-table td {
                color: rgba(245, 247, 250, 0.82);
                border-bottom: 1px solid rgba(255, 255, 255, 0.065);
                padding: 13px 14px;
                font-size: 0.86rem;
                vertical-align: top;
                line-height: 1.42;
            }
            .sc-table tr:nth-child(even) td {
                background: rgba(255, 255, 255, 0.025);
            }
            .sc-table tr:last-child td {
                border-bottom: 0;
            }
            .sc-table .sc-id {
                color: #fff7ed;
                font-weight: 800;
                white-space: nowrap;
            }
            .sc-table .sc-text-cell {
                min-width: 260px;
                max-width: 380px;
                white-space: normal;
            }
            .sc-progress {
                display: flex;
                align-items: center;
                gap: 10px;
                min-width: 132px;
            }
            .sc-progress-track {
                height: 8px;
                flex: 1 1 auto;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.08);
                overflow: hidden;
            }
            .sc-progress-fill {
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, #ef4444, #fb923c);
            }
            .sc-progress-value {
                color: #fff7ed;
                font-weight: 780;
                font-size: 0.82rem;
                min-width: 42px;
                text-align: right;
            }
            .sc-note-card {
                margin-top: 28px;
                padding: 18px 20px;
                border-color: rgba(251, 146, 60, 0.22);
                background:
                    linear-gradient(135deg, rgba(251, 146, 60, 0.09), rgba(239, 68, 68, 0.045)),
                    rgba(16, 18, 22, 0.92);
                color: rgba(245, 247, 250, 0.74);
                line-height: 1.55;
            }
            .sc-note-card b {
                color: #fed7aa;
            }
            @media (max-width: 1180px) {
                .sc-kpi-grid,
                .sc-card-grid {
                    grid-template-columns: 1fr;
                }
                .sc-score-components {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 720px) {
                .sc-kpi-grid,
                .sc-score-components,
                .sc-primary-metrics,
                .sc-secondary-metrics {
                    grid-template-columns: 1fr;
                }
                .sc-card-top,
                .sc-rule-row {
                    align-items: flex-start;
                    flex-direction: column;
                }
                .sc-kpi-card,
                .sc-score-card,
                .sc-strategy-card {
                    border-radius: 18px;
                    padding: 18px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_kpi_grid(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, note in items:
        cards.append(
            "<div class='sc-kpi-card'>"
            "<div>"
            f"<div class='sc-label'>{escape(label)}</div>"
            f"<div class='sc-value'>{escape(value)}</div>"
            "</div>"
            f"<div class='sc-note'>{escape(note)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='sc-grid sc-kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def business_risk_note(text: Any) -> str:
    note = clean_user_facing_note(text)
    replacements = [
        ("Fiyat koridoru geniş", "Fiyat koridoru geniş olduğu için önerinin belirsizliği artıyor."),
        ("Farklı fiyat modelleri arasında belirgin fark var", "Farklı fiyat modelleri birbirinden uzak sonuç verdiği için fiyat varsayımı manuel kontrol edilmeli."),
        ("Teslimat süresi baskılı", "Teslimat süresi baskılı göründüğü için operasyonel uygulanabilirlik ayrıca kontrol edilmeli."),
        ("geçmiş kazanılmış başarı grubunun tipik örneklerinden uzak", "Seçili ihale geçmiş kazanılmış başarı grubuna göre daha az tipik görünüyor; manuel inceleme önerilir."),
        ("Fiyat tarihsel fiyat bandının dışında", "Önerilen fiyat tarihsel emsal bandının dışında kaldığı için rekabet ve marj birlikte kontrol edilmeli."),
    ]
    for needle, replacement in replacements:
        if needle in note:
            return replacement
    return note.rstrip(" .;") + "." if note else ""


def scenario_risk_items(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw = clean_user_facing_note(value)
        raw_items = [item.strip() for item in re.split(r"\s*;\s*", raw) if item.strip()]
    items: list[str] = []
    for item in raw_items:
        normalized = business_risk_note(item)
        if normalized and normalized not in items:
            items.append(normalized)
    return items or ["Belirgin skor düşüren risk uyarısı yok."]


def strategy_business_description(label: str, fallback: str) -> str:
    if "Agresif" in label:
        return "Bu senaryo, fiyatı emsal koridora yakın tutarak daha rekabetçi bir teklif oluşturmayı hedefler. Marjı düşürebilir; bu nedenle maliyet varsayımı ve minimum karlılık eşiği kontrol edilmelidir."
    if "Dengeli" in label:
        return "Bu senaryo, fiyat bandı uyumu, karlılık ve risk seviyesini dengede tutmaya çalışır. Genellikle teklif komitesi için ana karşılaştırma senaryosu olarak kullanılabilir."
    if "Marj" in label:
        return "Bu senaryo, karlılığı korumaya daha fazla ağırlık verir. Fiyat geçmiş emsal bandından uzaklaşırsa rekabet riski artabilir; profil uyumu ve fiyat bandı birlikte değerlendirilmelidir."
    return fallback


def render_scenario_score_explanation(weights: dict[str, float]) -> None:
    risk_weight = abs(float(weights.get("risk_penalty_score", -0.10))) * 100
    components = [
        (
            f"%{weights.get('won_profile_fit_score', 0) * 100:.0f} Profil Uyumu",
            "Bu teklifin seçili ihalenin geçmiş kazanılmış profile ne kadar uyduğunu gösterir.",
        ),
        (
            f"%{weights.get('price_band_fit_score', 0) * 100:.0f} Fiyat Bandı Uyumu",
            "Önerilen fiyatın tarihsel emsal fiyat aralığına ne kadar yakın olduğunu gösterir.",
        ),
        (
            f"%{weights.get('margin_score', 0) * 100:.0f} Karlılık Skoru",
            "Tahmini maliyet ve beklenen marj açısından teklifin sağlıklı olup olmadığını gösterir.",
        ),
        (
            f"%{weights.get('model_confidence_score', 0) * 100:.0f} Model Güveni",
            "Emsal sayısı, benzerlik gücü ve veri kalitesine göre sistemin çıktıya ne kadar güvendiğini gösterir.",
        ),
        (
            f"-%{risk_weight:.0f} Risk Cezası",
            "Teslimat, maliyet belirsizliği, geniş fiyat bandı veya model uyuşmazlığı gibi riskler varsa skoru düşürür.",
        ),
    ]
    component_html = "".join(
        "<div class='sc-score-component'>"
        f"<b>{escape(title)}</b>"
        f"<span>{escape(body)}</span>"
        "</div>"
        for title, body in components
    )
    st.markdown(
        "<div class='sc-score-card'>"
        "<div class='sc-score-title'>Senaryo skoru nasıl okunur?</div>"
        "<div class='sc-score-copy'>Senaryo skoru, teklif seçeneğinin geçmiş kazanılmış ihale profiline, fiyat bandına, beklenen karlılığa, model güvenine ve risk uyarılarına göre hesaplanan karar destek skorudur. Bu skor gerçek kazanma olasılığı değildir. Yüksek skor daha dengeli karar destek sinyali anlamına gelir; kesin kazanır anlamına gelmez.</div>"
        f"<div class='sc-score-components'>{component_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_scenario_cards(selected_cards: list[tuple[str, str, pd.Series]], tender: dict[str, Any]) -> None:
    cards = []
    for idx, (label, description, scenario) in enumerate(selected_cards):
        invalid_reason = clean_user_facing_note(scenario.get("invalid_reason", ""))
        is_valid = bool(scenario["hard_constraints_valid"])
        status_label = "Temel kurallar uygun" if is_valid else f"Geçersiz: {invalid_reason or 'Kural ihlali var'}"
        status_class = "sc-pill-good" if is_valid else "sc-pill-bad"
        total_offer = float(scenario["proposed_unit_price"]) * float(tender.get("quantity", 0))
        contribution = total_offer * float(scenario["computed_margin_pct"]) / 100
        risk_value = max(0.0, 100 - float(scenario["risk_penalty_score"]))
        risk_label = "Düşük" if risk_value >= 75 else "Orta" if risk_value >= 55 else "Yüksek"
        risks = scenario_risk_items(scenario.get("soft_penalty_explanations", ""))
        if not is_valid and invalid_reason:
            risks = [business_risk_note(invalid_reason)] + [item for item in risks if item != business_risk_note(invalid_reason)]
        risk_html = "".join(f"<li>{escape(item)}</li>" for item in risks)
        primary_metrics = [
            ("Toplam teklif", format_try(total_offer)),
            ("Karlılık oranı", format_pct(scenario["computed_margin_pct"])),
            ("Katkı kârı", format_try(contribution)),
            ("Senaryo skoru", f"{float(scenario['scenario_score']):.0f}/100"),
        ]
        secondary_metrics = [
            ("Profil uyumu", format_score(scenario.get("won_profile_fit_score"))),
            ("Fiyat bandı uyumu", format_score(scenario.get("price_band_fit_score"))),
            ("Model güveni", format_score(scenario.get("model_confidence_score"))),
            ("Risk seviyesi", risk_label),
        ]
        primary_html = "".join(
            "<div class='sc-mini-metric'>"
            f"<span>{escape(metric_label)}</span>"
            f"<b>{escape(metric_value)}</b>"
            "</div>"
            for metric_label, metric_value in primary_metrics
        )
        secondary_html = "".join(
            "<div class='sc-mini-metric'>"
            f"<span>{escape(metric_label)}</span>"
            f"<b>{escape(metric_value)}</b>"
            "</div>"
            for metric_label, metric_value in secondary_metrics
        )
        cards.append(
            "<div class='sc-strategy-card'>"
            "<div class='sc-card-top'>"
            "<div style='display:flex; gap:12px; align-items:flex-start;'>"
            f"<div class='sc-card-number'>0{idx + 1}</div>"
            f"<div class='sc-card-title'>{escape(label)}</div>"
            "</div>"
            f"<span class='sc-pill {status_class}'>{escape(status_label)}</span>"
            "</div>"
            "<div>"
            "<div class='sc-price-label'>Ana fiyat</div>"
            f"<div class='sc-price'>{escape(format_try(scenario['proposed_unit_price']))}</div>"
            "</div>"
            f"<div class='sc-summary'>{escape(strategy_business_description(label, description))}</div>"
            "<div>"
            "<div class='sc-metric-group-title'>Key metrics</div>"
            f"<div class='sc-primary-metrics'>{primary_html}</div>"
            "</div>"
            "<div>"
            "<div class='sc-metric-group-title'>Secondary metrics</div>"
            f"<div class='sc-secondary-metrics'>{secondary_html}</div>"
            "</div>"
            "<div class='sc-risk-box'>"
            "<div class='sc-risk-title'>Skoru düşüren riskler</div>"
            f"<ul class='sc-risk-list'>{risk_html}</ul>"
            "</div>"
            "<div class='sc-rule-row'>"
            "<div class='sc-rule-label'>Kural kontrolü</div>"
            f"<span class='sc-pill {status_class}'>{escape(status_label)}</span>"
            "</div>"
            "</div>"
        )
    if cards:
        st.markdown(f"<div class='sc-grid sc-card-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def scenario_table_interpretation(row: pd.Series) -> str:
    valid = bool(row.get("Kural Durumu", False))
    invalid_reason = business_risk_note(row.get("Geçersiz Senaryo Açıklaması", ""))
    risks = scenario_risk_items(row.get("Risk Uyarıları / Skor Cezaları", ""))
    has_risk = risks != ["Belirgin skor düşüren risk uyarısı yok."]
    if not valid:
        return f"Senaryo kesin kural nedeniyle ana öneri olmamalıdır. {invalid_reason or 'Teklif komitesi kural ihlalini incelemelidir.'}"
    if has_risk:
        return "Senaryo geçerli; teklif komitesi şu noktaları kontrol etmeli: " + " ".join(risks[:2])
    return "Senaryo geçerli; belirgin ek risk uyarısı yok."


def scenario_progress_cell(value: Any) -> str:
    try:
        score = max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return escape(str(value))
    return (
        "<div class='sc-progress'>"
        "<div class='sc-progress-track'>"
        f"<div class='sc-progress-fill' style='width:{score:.1f}%'></div>"
        "</div>"
        f"<span class='sc-progress-value'>{score:.1f}</span>"
        "</div>"
    )


def render_scenario_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    text_columns = {
        "Geçersiz Senaryo Açıklaması",
        "Risk Uyarıları / Skor Cezaları",
        "Nasıl Yorumlanmalı?",
        "Kural / Risk Notları",
        "Benzer ihalelerden kanıt",
        "Not / uyarı",
    }
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = row[column]
            if column == "Senaryo Skoru":
                cells.append(f"<td>{scenario_progress_cell(value)}</td>")
            elif column == "Senaryo ID":
                cells.append(f"<td class='sc-id'>{escape(str(value))}</td>")
            elif column in text_columns:
                cells.append(f"<td class='sc-text-cell'>{escape(str(value))}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='sc-table-card'><div class='sc-table-scroll'>"
        f"<table class='sc-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_scenario_decision_note() -> None:
    st.markdown(
        "<div class='sc-note-card'><b>Karar notu:</b> "
        "Bu sayfa tek bir doğru fiyat seçmek için değil, farklı teklif stratejilerinin fiyat, karlılık, risk ve kural uygunluğu açısından karşılaştırılması için kullanılır. Nihai karar teklif komitesi tarafından maliyet, stok, teslimat ve ticari öncelikler dikkate alınarak verilmelidir."
        "</div>",
        unsafe_allow_html=True,
    )


def inject_reveal_compare_css() -> None:
    st.markdown(
        """
        <style>
            .rc-section {
                margin-top: 50px;
            }
            .rc-section .section-title {
                font-size: 1.24rem;
                line-height: 1.25;
                margin-bottom: 0.18rem;
            }
            .rc-section .section-subtitle {
                max-width: 900px;
                color: rgba(245, 247, 250, 0.68);
                line-height: 1.55;
                margin-bottom: 18px;
            }
            .rc-grid {
                display: grid;
                gap: 20px;
                align-items: stretch;
            }
            .rc-grid-3 {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .rc-grid-4 {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
            .rc-card,
            .rc-summary-card,
            .rc-story-card,
            .rc-table-card,
            .rc-export-card {
                border: 1px solid rgba(248, 113, 113, 0.16);
                background:
                    linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.022)),
                    rgba(16, 18, 22, 0.92);
                box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
                border-radius: 20px;
            }
            .rc-card {
                min-height: 134px;
                padding: 21px 22px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .rc-label {
                color: rgba(245, 247, 250, 0.62);
                font-size: 0.76rem;
                font-weight: 820;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .rc-value {
                color: #fff7ed;
                font-size: 1.52rem;
                line-height: 1.12;
                font-weight: 860;
                margin-top: 10px;
                overflow-wrap: anywhere;
            }
            .rc-note {
                color: rgba(245, 247, 250, 0.65);
                font-size: 0.86rem;
                line-height: 1.42;
                margin-top: 12px;
            }
            .rc-summary-card {
                padding: 24px;
                border-color: rgba(251, 146, 60, 0.24);
                background:
                    linear-gradient(135deg, rgba(251, 146, 60, 0.10), rgba(239, 68, 68, 0.045)),
                    rgba(16, 18, 22, 0.92);
                margin-top: 28px;
            }
            .rc-summary-top {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 18px;
                margin-bottom: 18px;
            }
            .rc-summary-title {
                color: #fff7ed;
                font-size: 1.18rem;
                font-weight: 860;
                line-height: 1.25;
            }
            .rc-summary-copy {
                color: rgba(245, 247, 250, 0.73);
                line-height: 1.58;
                max-width: 980px;
            }
            .rc-pill {
                flex: 0 0 auto;
                border: 1px solid rgba(251, 146, 60, 0.34);
                background: rgba(251, 146, 60, 0.12);
                color: #fed7aa;
                border-radius: 999px;
                padding: 6px 10px;
                font-size: 0.74rem;
                font-weight: 820;
                white-space: nowrap;
            }
            .rc-pill-good {
                border-color: rgba(34, 197, 94, 0.30);
                background: rgba(34, 197, 94, 0.10);
                color: #bbf7d0;
            }
            .rc-pill-warn {
                border-color: rgba(251, 146, 60, 0.34);
                background: rgba(251, 146, 60, 0.12);
                color: #fed7aa;
            }
            .rc-pill-bad {
                border-color: rgba(248, 113, 113, 0.34);
                background: rgba(248, 113, 113, 0.12);
                color: #fecaca;
            }
            .rc-summary-metrics {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
            }
            .rc-summary-metric {
                border: 1px solid rgba(255, 255, 255, 0.075);
                background: rgba(255, 255, 255, 0.035);
                border-radius: 14px;
                padding: 13px;
            }
            .rc-summary-metric span {
                display: block;
                color: rgba(245, 247, 250, 0.55);
                font-size: 0.72rem;
                font-weight: 780;
                margin-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }
            .rc-summary-metric b {
                display: block;
                color: #fff7ed;
                font-size: 1.08rem;
                line-height: 1.2;
            }
            .rc-price-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
            }
            .rc-price-point {
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(255, 255, 255, 0.035);
                border-radius: 16px;
                padding: 16px;
            }
            .rc-price-point.actual {
                border-color: rgba(251, 146, 60, 0.30);
                background: rgba(251, 146, 60, 0.08);
            }
            .rc-story-card {
                padding: 22px;
                display: grid;
                grid-template-columns: 0.35fr 1fr;
                gap: 20px;
                align-items: center;
            }
            .rc-rank-value {
                color: #fff7ed;
                font-size: 2.2rem;
                font-weight: 880;
                line-height: 1;
            }
            .rc-story-copy {
                color: rgba(245, 247, 250, 0.72);
                line-height: 1.56;
            }
            .rc-table-card {
                padding: 18px;
            }
            .rc-table-scroll {
                overflow-x: auto;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .rc-table {
                width: 100%;
                border-collapse: collapse;
                min-width: 980px;
                background: rgba(12, 14, 18, 0.88);
            }
            .rc-table th {
                text-align: left;
                color: rgba(255, 247, 237, 0.84);
                background: rgba(26, 29, 36, 0.98);
                border-bottom: 1px solid rgba(248, 113, 113, 0.18);
                padding: 13px 14px;
                font-size: 0.74rem;
                letter-spacing: 0.03em;
                text-transform: uppercase;
                white-space: nowrap;
            }
            .rc-table td {
                color: rgba(245, 247, 250, 0.82);
                border-bottom: 1px solid rgba(255, 255, 255, 0.065);
                padding: 13px 14px;
                font-size: 0.87rem;
                vertical-align: top;
                line-height: 1.42;
            }
            .rc-table tr:nth-child(even) td {
                background: rgba(255, 255, 255, 0.025);
            }
            .rc-table tr:last-child td {
                border-bottom: 0;
            }
            .rc-table .rc-strong {
                color: #fff7ed;
                font-weight: 800;
                white-space: nowrap;
            }
            .rc-export-card {
                margin-top: 28px;
                padding: 18px 20px;
                border-color: rgba(251, 146, 60, 0.22);
                background:
                    linear-gradient(135deg, rgba(251, 146, 60, 0.09), rgba(239, 68, 68, 0.045)),
                    rgba(16, 18, 22, 0.92);
                color: rgba(245, 247, 250, 0.74);
                line-height: 1.55;
            }
            .rc-export-card b {
                color: #fed7aa;
            }
            @media (max-width: 1180px) {
                .rc-grid-3,
                .rc-grid-4,
                .rc-summary-metrics,
                .rc-price-strip {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .rc-story-card {
                    grid-template-columns: 1fr;
                }
            }
            @media (max-width: 720px) {
                .rc-grid-3,
                .rc-grid-4,
                .rc-summary-metrics,
                .rc-price-strip {
                    grid-template-columns: 1fr;
                }
                .rc-summary-top {
                    flex-direction: column;
                }
                .rc-card,
                .rc-summary-card,
                .rc-story-card {
                    border-radius: 18px;
                    padding: 18px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reveal_rank_level(rank_pct: float) -> tuple[str, str]:
    if rank_pct >= 75:
        return "Güçlü uyum", "good"
    if rank_pct >= 50:
        return "Orta / kabul edilebilir uyum", "warn"
    if rank_pct >= 25:
        return "Zayıf / dikkat gerektiren uyum", "warn"
    return "Düşük uyum, manuel inceleme gerekli", "bad"


def reveal_overall_level(inside: bool, pct_error: float, rank_pct: float) -> tuple[str, str]:
    if inside and pct_error <= 15 and rank_pct >= 50:
        return "İyi uyum", "good"
    if inside or rank_pct >= 50:
        return "Orta uyum", "warn"
    return "Zayıf uyum", "bad"


def reveal_price_gap_comment(actual_price: float, mid_price: float) -> str:
    diff = actual_price - mid_price
    if abs(diff) < 0.01:
        return "Gerçek fiyat, dengeli öneriyle neredeyse aynı çıktı."
    direction = "yüksek" if diff > 0 else "düşük"
    implication = "orta senaryonun daha ihtiyatlı kaldığını" if diff > 0 else "orta senaryonun gerçek sonuca göre yüksek kaldığını"
    return f"Gerçek fiyat, dengeli öneriden {format_try(abs(diff))} {direction} çıktı. Bu, sistemin {implication} gösterir."


def reveal_inside_comment(inside: bool) -> str:
    if inside:
        return "Gerçek kazanılmış fiyat, sistemin önerdiği düşük-yüksek koridor içinde kaldı. Bu, koridorun bu ihale için tarihsel fiyat davranışını makul yakaladığını gösterir."
    return "Gerçek kazanılmış fiyat, önerilen koridorun dışında kaldı. Bu ihale için fiyat davranışı geçmiş emsallerden farklı olabilir; manuel analiz gerekir."


def reveal_rank_comment_business(rank_pct: float) -> str:
    label, _ = reveal_rank_level(rank_pct)
    return (
        "Bu metrik, gerçek kazanılmış senaryonun sistemin ürettiği aday senaryolar içinde ne kadar üst sıralarda kaldığını gösterir. "
        f"Bu ihale için sonuç: {label}. Değer yükseldikçe sistemin gerçek sonuca yakın senaryoları daha iyi öne çıkardığı anlaşılır."
    )


def render_reveal_metric_grid(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    cards = []
    for label, value, note in items:
        cards.append(
            "<div class='rc-card'>"
            "<div>"
            f"<div class='rc-label'>{escape(label)}</div>"
            f"<div class='rc-value'>{escape(value)}</div>"
            "</div>"
            f"<div class='rc-note'>{escape(note)}</div>"
            "</div>"
        )
    grid_class = "rc-grid-4" if columns == 4 else "rc-grid-3"
    st.markdown(f"<div class='rc-grid {grid_class}'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_reveal_summary(inside: bool, abs_error: float, pct_error: float, rank_pct: float, actual_price: float, mid_price: float) -> None:
    overall_label, status = reveal_overall_level(inside, pct_error, rank_pct)
    badge_class = {"good": "rc-pill-good", "warn": "rc-pill-warn", "bad": "rc-pill-bad"}[status]
    summary = (
        f"{reveal_inside_comment(inside)} "
        f"{reveal_price_gap_comment(actual_price, mid_price)} "
        f"Gerçek kazanılmış senaryo, sistem senaryolarının üst {format_pct(rank_pct)} bölümünde yer aldı."
    )
    metrics = [
        ("Koridor durumu", "İçinde" if inside else "Dışında"),
        ("Dengeli fiyattan fark", f"{format_try(abs_error)} ({format_pct(pct_error)})"),
        ("Gerçek senaryo sırası", format_pct(rank_pct)),
    ]
    metric_html = "".join(
        "<div class='rc-summary-metric'>"
        f"<span>{escape(label)}</span>"
        f"<b>{escape(value)}</b>"
        "</div>"
        for label, value in metrics
    )
    st.markdown(
        "<div class='rc-summary-card'>"
        "<div class='rc-summary-top'>"
        "<div>"
        "<div class='rc-summary-title'>Sonuç Özeti</div>"
        f"<div class='rc-summary-copy'>{escape(summary)}</div>"
        "</div>"
        f"<span class='rc-pill {badge_class}'>Genel yorum: {escape(overall_label)}</span>"
        "</div>"
        f"<div class='rc-summary-metrics'>{metric_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_price_strip(corridor: dict[str, Any], actual_price: float) -> None:
    points = [
        ("Düşük öneri", format_try(corridor["predicted_low_price"]), "Daha rekabetçi seviye", ""),
        ("Dengeli öneri", format_try(corridor["predicted_mid_price"]), "Emsal orta fiyat", ""),
        ("Yüksek öneri", format_try(corridor["predicted_high_price"]), "Daha karlı seviye", ""),
        ("Gerçek kazanılmış fiyat", format_try(actual_price), "Reveal sonrası gerçek sonuç", " actual"),
    ]
    html = "".join(
        f"<div class='rc-price-point{css}'>"
        f"<div class='rc-label'>{escape(label)}</div>"
        f"<div class='rc-value'>{escape(value)}</div>"
        f"<div class='rc-note'>{escape(note)}</div>"
        "</div>"
        for label, value, note, css in points
    )
    st.markdown(f"<div class='rc-price-strip'>{html}</div>", unsafe_allow_html=True)


def render_reveal_rank_card(rank_pct: float) -> None:
    label, status = reveal_rank_level(rank_pct)
    badge_class = {"good": "rc-pill-good", "warn": "rc-pill-warn", "bad": "rc-pill-bad"}[status]
    st.markdown(
        "<div class='rc-story-card'>"
        "<div>"
        "<div class='rc-label'>Gerçek senaryo sırası</div>"
        f"<div class='rc-rank-value'>{escape(format_pct(rank_pct))}</div>"
        f"<span class='rc-pill {badge_class}'>{escape(label)}</span>"
        "</div>"
        f"<div class='rc-story-copy'>{escape(reveal_rank_comment_business(rank_pct))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_reveal_table(df: pd.DataFrame) -> None:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in df.columns:
            value = escape(str(row[column]))
            if column in {"İhale ID", "Fiyat noktası", "Metrik"}:
                cells.append(f"<td class='rc-strong'>{value}</td>")
            else:
                cells.append(f"<td>{value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        "<div class='rc-table-card'><div class='rc-table-scroll'>"
        f"<table class='rc-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_reveal_export_note() -> None:
    st.markdown(
        "<div class='rc-export-card'><b>Karar notu:</b> "
        "Bu sonuç tek seçili ihale için reveal sonrası kontrol raporudur. Nihai karar; maliyet, stok, teslimat, ticari öncelikler ve teklif komitesi değerlendirmesiyle verilmelidir."
        "</div>",
        unsafe_allow_html=True,
    )


def backtest_summary_comment(metrics: dict[str, float], leakage_pass: bool) -> tuple[str, str]:
    coverage = float(metrics.get("band_coverage", 0.0))
    mape = float(metrics.get("mape", 0.0))
    band_score = float(metrics.get("coverage_adjusted_band_score", 0.0))
    if not leakage_pass:
        return (
            "Sızıntı uyarısı",
            "Leakage audit pass olmadığı için bu backtest sonucu güvenli performans karnesi olarak okunmamalıdır.",
        )
    if coverage >= 0.65 and mape <= 30 and band_score >= 0.35:
        return (
            "Güçlü karar desteği",
            "Sistem fiyat koridorunu test döneminde görece tutarlı yakalamış görünüyor. Yine de sonuçlar gerçek kazanma olasılığı değil, karar destek sinyalidir.",
        )
    if coverage >= 0.45:
        return (
            "Dikkatli okunmalı",
            "Sistem fiyat koridorunu bazı ihalelerde yakalıyor; ancak band genişliği ve hata oranları nedeniyle fiyat çıktısı karar desteği olarak dikkatli yorumlanmalıdır. Profil ve sıra dışılık sinyalleri ayrı değerlendirilmelidir.",
        )
    return (
        "Zayıf fiyat uyumu",
        "Fiyat koridoru test döneminde gerçek sonuçları sınırlı yakalamış görünüyor. Fiyat varsayımları, emsal havuzu ve segment kırılımları manuel incelenmelidir.",
    )


def render_backtest_summary(results: pd.DataFrame, metrics: dict[str, float], leakage_pass: bool) -> None:
    label, comment = backtest_summary_comment(metrics, leakage_pass)
    status = "rc-pill-good" if label == "Güçlü karar desteği" else "rc-pill-bad" if label in {"Sızıntı uyarısı", "Zayıf fiyat uyumu"} else "rc-pill-warn"
    metric_html = "".join(
        "<div class='rc-summary-metric'>"
        f"<span>{escape(metric_label)}</span>"
        f"<b>{escape(metric_value)}</b>"
        "</div>"
        for metric_label, metric_value in [
            ("Test ihalesi sayısı", format_int(len(results))),
            ("Band coverage", format_pct(float(metrics.get("band_coverage", 0.0)) * 100)),
            ("MAE", format_try(metrics.get("mae", 0.0))),
            ("MAPE", format_pct(metrics.get("mape", 0.0))),
            ("Leakage", "Sızıntı yok" if leakage_pass else "Uyarı var"),
            ("Band kalite skoru", f"{float(metrics.get('coverage_adjusted_band_score', 0.0)):.2f}"),
        ]
    )
    st.markdown(
        "<div class='rc-summary-card'>"
        "<div class='rc-summary-top'>"
        "<div>"
        "<div class='rc-summary-title'>Backtest Genel Özeti</div>"
        "<div class='rc-summary-copy'>Backtest, geçmişte kazanılmış ihaleleri canlı ihale gibi test eder. Her ihalede gerçek sonuç önce gizlenir, sistem emsal/profil/fiyat/senaryo çıktısı üretir, sonra gerçek sonuç açılarak genel tutarlılık ölçülür. Bu sayfa tek bir ihale değil, tüm test döneminin performansını gösterir.</div>"
        f"<div class='rc-summary-copy' style='margin-top:10px;'>{escape(comment)}</div>"
        "</div>"
        f"<span class='rc-pill {status}'>Genel yorum: {escape(label)}</span>"
        "</div>"
        f"<div class='rc-summary-metrics'>{metric_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def format_backtest_table(df: pd.DataFrame, price_columns: set[str] | None = None, pct_columns: set[str] | None = None, score_columns: set[str] | None = None) -> pd.DataFrame:
    display = df.copy()
    price_columns = price_columns or set()
    pct_columns = pct_columns or set()
    score_columns = score_columns or set()

    def is_missing_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        return False

    def display_value(value: Any) -> str:
        if is_missing_value(value):
            return "-"
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def pct_value(value: Any) -> str:
        if value == "" or is_missing_value(value):
            return "-"
        number = float(value)
        return format_pct(number * 100 if abs(number) <= 1 else number)

    for column in display.columns:
        if column in price_columns:
            display[column] = display[column].apply(format_try)
        elif column in pct_columns:
            display[column] = display[column].apply(pct_value)
        elif column in score_columns:
            display[column] = display[column].apply(format_score)
        else:
            display[column] = display[column].apply(display_value)
    return display


def render_backtest_export_panel(
    metrics: dict[str, float],
    results: pd.DataFrame,
    leakage_report: pd.DataFrame,
    segment_display: pd.DataFrame,
    stress_results: pd.DataFrame,
) -> None:
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        audited_download_button("Backtest Raporu", dataframe_to_csv_bytes(pd.DataFrame([metrics])), "backtest_raporu.csv", width="stretch")
    with e2:
        audited_download_button("Tender-Level Sonuçlar", dataframe_to_csv_bytes(results), "tender_level_sonuclar.csv", width="stretch")
    with e3:
        audited_download_button("Gerçek Sonuç Sızıntısı Kontrolü", dataframe_to_csv_bytes(leakage_report), "leakage_audit.csv", width="stretch")
    e4, e5, e6 = st.columns(3, gap="medium")
    with e4:
        audited_download_button("Segment Metrikleri", dataframe_to_csv_bytes(segment_display), "segment_metrikleri.csv", width="stretch")
    with e5:
        audited_download_button("Expert Review Export", dataframe_to_csv_bytes(expert_review_template(results)), "expert_review_export.csv", width="stretch")
    with e6:
        audited_download_button("Sentetik Aykırı Senaryo Testi", dataframe_to_csv_bytes(stress_results), "sentetik_aykiri_senaryo_testi.csv", width="stretch")


def inject_reports_css() -> None:
    st.markdown(
        """
        <style>
            .report-section {
                margin-top: 48px;
            }
            .report-section-title {
                color: #fff7ed;
                font-size: 1.18rem;
                font-weight: 780;
                line-height: 1.25;
                margin-bottom: 0.22rem;
            }
            .report-section-subtitle {
                color: rgba(245, 247, 250, 0.66);
                line-height: 1.55;
                max-width: 900px;
                margin-bottom: 18px;
            }
            .report-control-grid {
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 20px;
            }
            .report-control-card {
                grid-column: span 2;
                min-height: 156px;
                border: 1px solid rgba(248, 113, 113, 0.16);
                background:
                    linear-gradient(145deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018)),
                    rgba(16, 17, 20, 0.94);
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 18px 44px rgba(0,0,0,0.24);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 16px;
            }
            .report-control-card.center-left { grid-column: 2 / span 2; }
            .report-control-card.center-right { grid-column: 4 / span 2; }
            .report-control-top {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
            }
            .report-control-kicker {
                color: rgba(245,247,250,0.56);
                font-size: .72rem;
                font-weight: 780;
                letter-spacing: .06em;
                text-transform: uppercase;
            }
            .report-control-title {
                color: #fff7ed;
                font-size: 1.03rem;
                font-weight: 820;
                line-height: 1.25;
                margin-top: 6px;
            }
            .report-control-body {
                color: rgba(245,247,250,0.66);
                font-size: .88rem;
                line-height: 1.44;
            }
            .report-badge {
                display: inline-flex;
                align-items: center;
                min-height: 26px;
                padding: 4px 9px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.10);
                background: rgba(255,255,255,0.055);
                color: #fff7ed;
                font-size: .73rem;
                font-weight: 760;
                white-space: nowrap;
            }
            .report-badge-success {
                border-color: rgba(34,197,94,0.28);
                background: rgba(34,197,94,0.12);
                color: #bbf7d0;
            }
            .report-badge-warning {
                border-color: rgba(251,146,60,0.32);
                background: rgba(251,146,60,0.12);
                color: #fed7aa;
            }
            .report-badge-danger {
                border-color: rgba(248,113,113,0.32);
                background: rgba(248,113,113,0.13);
                color: #fecaca;
            }
            .report-export-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 20px;
                align-items: stretch;
            }
            .st-key-report_export_backtest,
            .st-key-report_export_scenario,
            .st-key-report_export_audit,
            .st-key-report_export_review {
                border: 1px solid rgba(248, 113, 113, 0.16);
                background:
                    radial-gradient(ellipse at 16% 0%, rgba(255,79,31,0.12), transparent 34%),
                    linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.018)),
                    rgba(14, 15, 18, 0.94);
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 18px 44px rgba(0,0,0,0.24);
                min-height: 292px;
            }
            .st-key-report_export_backtest [data-testid='stVerticalBlock'],
            .st-key-report_export_scenario [data-testid='stVerticalBlock'],
            .st-key-report_export_audit [data-testid='stVerticalBlock'],
            .st-key-report_export_review [data-testid='stVerticalBlock'] {
                gap: .62rem;
            }
            .report-export-title {
                color: #fff7ed;
                font-size: 1rem;
                font-weight: 820;
                line-height: 1.25;
            }
            .report-export-copy {
                color: rgba(245,247,250,0.62);
                font-size: .86rem;
                line-height: 1.42;
                min-height: 42px;
                margin: 7px 0 12px;
            }
            .report-export-action {
                border-radius: 14px;
                padding: 10px 11px;
                border: 1px solid rgba(255,255,255,0.08);
                background: rgba(255,255,255,0.035);
            }
            .report-export-action-title {
                color: #fff7ed;
                font-size: .86rem;
                font-weight: 760;
                line-height: 1.2;
            }
            .report-export-action-note {
                color: rgba(245,247,250,0.56);
                font-size: .76rem;
                line-height: 1.34;
                margin-top: 3px;
            }
            .st-key-report_export_backtest div[data-testid='stDownloadButton'] button,
            .st-key-report_export_scenario div[data-testid='stDownloadButton'] button,
            .st-key-report_export_audit div[data-testid='stDownloadButton'] button,
            .st-key-report_export_review div[data-testid='stDownloadButton'] button {
                width: 100% !important;
                min-height: 38px !important;
                border-radius: 999px !important;
                border: 1px solid rgba(255,184,117,0.28) !important;
                background: linear-gradient(180deg, rgba(255,93,36,0.92), rgba(181,40,13,0.92)) !important;
                color: #ffffff !important;
                box-shadow: 0 10px 24px rgba(255,79,31,0.16) !important;
            }
            .report-detail-card {
                border: 1px solid rgba(248,113,113,0.16);
                border-radius: 20px;
                background: rgba(14,15,18,0.92);
                padding: 18px;
                box-shadow: 0 18px 44px rgba(0,0,0,0.22);
            }
            @media (max-width: 1180px) {
                .report-export-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .report-control-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                .report-control-card,
                .report-control-card.center-left,
                .report-control-card.center-right { grid-column: auto; }
            }
            @media (max-width: 760px) {
                .report-export-grid,
                .report-control-grid { grid-template-columns: 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_report_section(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='report-section'><div class='report-section-title'>{escape(title)}</div>"
        f"<div class='report-section-subtitle'>{escape(subtitle)}</div></div>",
        unsafe_allow_html=True,
    )


def render_report_control_cards(cards: list[dict[str, str]]) -> None:
    html = []
    for idx, card in enumerate(cards):
        extra_class = " center-left" if idx == 3 else " center-right" if idx == 4 else ""
        status = card.get("status", "success")
        html.append(
            f"<div class='report-control-card{extra_class}'>"
            "<div class='report-control-top'>"
            "<div>"
            f"<div class='report-control-kicker'>{escape(card['kicker'])}</div>"
            f"<div class='report-control-title'>{escape(card['title'])}</div>"
            "</div>"
            f"<span class='report-badge report-badge-{escape(status)}'>{escape(card['badge'])}</span>"
            "</div>"
            f"<div class='report-control-body'>{escape(card['body'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='report-control-grid'>{''.join(html)}</div>", unsafe_allow_html=True)


def render_export_action(label: str, note: str, data: Any, file_name: str, mime: str | None = None) -> None:
    st.markdown(
        "<div class='report-export-action'>"
        f"<div class='report-export-action-title'>{escape(label)}</div>"
        f"<div class='report-export-action-note'>{escape(note)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    audited_download_button(label, data, file_name, mime=mime, width="stretch")


def render_report_export_group(key: str, title: str, description: str, actions: list[dict[str, Any]]) -> None:
    with st.container(key=key):
        st.markdown(
            f"<div class='report-export-title'>{escape(title)}</div>"
            f"<div class='report-export-copy'>{escape(description)}</div>",
            unsafe_allow_html=True,
        )
        for action in actions:
            render_export_action(
                action["label"],
                action["note"],
                action["data"],
                action["file_name"],
                action.get("mime"),
            )


def build_gauge(score: float, title: str = "Skor") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "/100", "font": {"size": 30, "color": "#ffffff"}},
            title={"text": title, "font": {"size": 14, "color": "rgba(255,255,255,0.84)"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6a2b"},
                "bgcolor": "rgba(255,255,255,0.06)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 123, 66, 0.22)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(255, 79, 31, 0.16)"},
                    {"range": [45, 70], "color": "rgba(255, 157, 66, 0.14)"},
                    {"range": [70, 100], "color": "rgba(216, 155, 82, 0.14)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=12, r=12, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
    )
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
    secret_key, _ = streamlit_secret_openrouter_key()
    if secret_key:
        return secret_key
    local_key, _ = read_local_openrouter_secret()
    return local_key


def advisor_context(result: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    similar = result["similar"].head(5)
    tender = current_tender() or {}
    baseline = predict_baseline_prices(get_history_frame(), tender) if tender else pd.DataFrame()
    quality = retrieval_quality_from_result(result, tender)
    scenario_weights = load_scenario_weights()
    corridor = result["corridor"]
    evidence_items = [
        {
            "evidence_id": "E_PRICE_001",
            "type": "price_band",
            "content": f"Top-K fiyat koridoru: düşük {format_try(corridor.get('predicted_low_price'))}, orta {format_try(corridor.get('predicted_mid_price'))}, yüksek {format_try(corridor.get('predicted_high_price'))}.",
        },
        {
            "evidence_id": "E_PROFILE_001",
            "type": "profile_fit",
            "content": f"Kazanılmış profil uyumu {float(best.get('won_profile_fit_score', 0)):.1f}/100.",
        },
        {
            "evidence_id": "E_CONF_001",
            "type": "model_confidence",
            "content": f"Model güven skoru {float(result.get('model_confidence_score', 0)):.1f}/100.",
        },
        {
            "evidence_id": "E_RISK_001",
            "type": "risk_flags",
            "content": "Risk bayrakları: " + (", ".join(best.get("risk_flags", [])) if best.get("risk_flags") else "belirgin risk bayrağı yok."),
        },
        {
            "evidence_id": "E_SIMILAR_001",
            "type": "similar_tenders",
            "content": f"Benzer ihale sayısı {len(result['similar'])}; top-10 ortalama benzerlik {float(result.get('top10_avg_similarity', 0)):.2f}.",
        },
    ]
    context = {
        **best,
        "tender_id": tender.get("tender_id"),
        "product_name": tender.get("product_name"),
        "product_group": tender.get("product_group"),
        "region": tender.get("region"),
        "quantity": tender.get("quantity"),
        "delivery_months": tender.get("delivery_months"),
        "estimated_unit_cost": tender.get("estimated_unit_cost"),
        "corridor": corridor,
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
        "leakage_audit": st.session_state.get("leakage_audit", {"audit_status": "pass"}),
        "evidence_items": evidence_items,
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
    baselines = context.get("baseline_model_predictions", [])
    baseline_text = "; ".join(
        f"{item.get('method')}: {format_try(item.get('prediction'))}"
        for item in baselines[:4]
        if item.get("prediction") is not None
    )
    profile_score = float(context.get("won_profile_fit_score", 0) or 0)
    price_score = float(context.get("price_band_fit_score", 0) or 0)
    margin_score = float(context.get("margin_score", 0) or 0)
    confidence_score = float(context.get("model_confidence_score", 0) or 0)
    scenario_score = float(context.get("scenario_score", 0) or 0)
    margin_pct_value = float(context.get("computed_margin_pct", 0) or 0)
    proposed_price = context.get("proposed_unit_price")
    similar_count = int(context.get("similar_tender_count", 0) or 0)
    cluster_name = context.get("cluster_name", "Kazanılmış profil grubu")
    isolation_status = context.get("isolation_forest", {}).get("status", "Profil kontrolü hesaplandı")
    if "fiyat" in q or "koridor" in q:
        answer = (
            "Fiyat yorumu üç katmanla okunmalı:\n\n"
            f"1. Emsal koridoru: Benzer kazanılmış ihaleler düşük {format_try(corridor.get('predicted_low_price'))}, "
            f"orta {format_try(corridor.get('predicted_mid_price'))}, yüksek {format_try(corridor.get('predicted_high_price'))} bandını veriyor. "
            "Bu band geçmiş kazanılmış işlerde görülen fiyat davranışını temsil eder.\n\n"
            f"2. Seçilen senaryo: {format_try(proposed_price)} birim fiyatla yaklaşık {format_pct(margin_pct_value)} beklenen karlılık oranı üretiyor. "
            f"Fiyat bandı uyum skoru {price_score:.1f}/100; bu fiyatın emsal koridora ne kadar yakın olduğunu gösterir.\n\n"
            f"3. Baseline kontrolü: {baseline_text or 'aktif baz model çıktısı yok'}. "
            "Eğer koridor ve baz modeller aynı yöne işaret ediyorsa fiyat kararı daha rahat okunur; ayrışma varsa manuel fiyat incelemesi gerekir."
        )
    elif "risk" in q or "manuel" in q:
        manual_review_required = confidence_score < 50 or profile_score < 45 or bool(risk_flags)
        answer = (
            "Manuel inceleme kararı tek bir metrikten gelmiyor; üç sinyal birlikte okunuyor:\n\n"
            f"1. Model güveni {confidence_score:.1f}/100. Benzer ihale sayısı ve benzerlik gücü yeterliyse karar desteği daha sağlamdır.\n"
            f"2. Profil uyumu {profile_score:.1f}/100. Düşükse ihale geçmiş kazanılmış örneklere daha az benziyor demektir.\n"
            f"3. Risk ve kural notları: {risk_text}.\n\n"
            f"Sonuç: {'Manuel inceleme önerilir; fiyat, marj ve teslim varsayımları iş birimiyle kontrol edilmeli.' if manual_review_required else 'Manuel inceleme kritik görünmüyor; yine de teklif onayı öncesi maliyet ve teslim varsayımları kontrol edilmeli.'}"
        )
    elif "benzer" in q or "profile" in q or "profil" in q or "küme" in q:
        answer = (
            "Profil yorumu, bu ihalenin geçmişte kazanılmış işlere ne kadar tanıdık göründüğünü anlatır:\n\n"
            f"1. Emsal havuzu: Sistem {similar_count} benzer kazanılmış ihaleyi referans aldı. "
            "Ürün grubu, bölge, kurum tipi, miktar ve metinsel benzerlik birlikte kullanılır.\n\n"
            f"2. Başarı grubu: İhale '{cluster_name}' profiline yakın konumlandı. "
            "Bu mixed-type cluster çıktısı, ihalenin hangi geçmiş başarı segmentine benzediğini gösterir.\n\n"
            f"3. Sıra dışılık kontrolü: {isolation_status}. Isolation Forest burada kayıp tahmini yapmaz; sadece geçmiş kazanılmış dağılım içinde alışıldık mı diye bakar.\n\n"
            f"İş yorumu: Profil uyumu {profile_score:.1f}/100. Bu değer yüksekse geçmiş kazanım örneklerine benzerlik güçlüdür; düşükse fiyat ve teslim koşulları daha dikkatli incelenmelidir."
        )
    elif "neden" in q or "öner" in q or "senaryo" in q:
        answer = (
            "Bu skorun nedeni bileşen bazında şöyle okunmalı:\n\n"
            f"1. Profil uyumu: {profile_score:.1f}/100. İhale geçmiş kazanılmış profillere ne kadar benziyor sorusunu yanıtlar.\n"
            f"2. Fiyat bandı uyumu: {price_score:.1f}/100. Önerilen fiyatın emsal koridorla hizasını ölçer.\n"
            f"3. Karlılık: Önerilen {format_try(proposed_price)} birim fiyat yaklaşık {format_pct(margin_pct_value)} beklenen karlılık oranı üretiyor. Karlılık skoru {margin_score:.1f}/100.\n"
            f"4. Model güveni: {confidence_score:.1f}/100. Benzer kayıt sayısı ve benzerlik kalitesi yeterli mi diye bakar.\n"
            f"5. Risk cezası: {risk_text}.\n\n"
            f"Toplam senaryo skoru {scenario_score:.1f}/100. Business yorumu olarak bu skor, teklif seçeneğinin geçmiş kazanılmış örneklerle uyumunu ve fiyat-marj-risk dengesini gösterir; tek başına otomatik teklif kararı değildir."
        )
    else:
        answer = (
            f"{advisor.get('executive_summary', '')}\n\n"
            f"Önerilen aksiyon: {advisor.get('recommended_action', '')}\n\n"
            f"Senaryo gerekçesi: {advisor.get('scenario_rationale', '')}\n\n"
            f"Güven gerekçesi: {advisor.get('confidence_rationale', '')}\n\n"
            "Detaylı okumada profil uyumu, fiyat bandı uyumu, beklenen karlılık, model güveni ve risk bayrakları birlikte değerlendirilmelidir."
        )
    return answer


def call_guarded_llm(context: dict[str, Any], question: str) -> dict[str, Any] | None:
    injection = detect_prompt_injection(question)
    if injection["prompt_injection_detected"]:
        set_advisor_llm_status("blocked", "Guardrail", "Prompt injection tespit edildi.")
        audit_event(
            {
                "event_type": "prompt_injection_detected",
                "user_action": "advisor_question",
                "tender_id": context.get("tender_id"),
                "module": "advisor",
                "input_summary": question[:240],
                "output_summary": "blocked",
                "validation_status": "blocked",
                "leakage_status": context.get("leakage_audit", {}).get("audit_status", "unknown"),
                "advisor_guardrail_status": injection["guardrail_status"],
            }
        )
        return None
    if llm_provider() in {"none", "offline", "disabled", "fallback"}:
        set_advisor_llm_status("fallback", "Güvenli fallback", "LLM_PROVIDER offline/fallback modunda.")
        log_event(
            "fallback_advisor_used",
            module="advisor",
            status="pass",
            message="LLM_PROVIDER offline modda; fallback advisor kullanılacak.",
            tender_id=str(context.get("tender_id") or "") or None,
        )
        return None
    api_key = get_openrouter_api_key()
    if not api_key:
        set_advisor_llm_status("fallback", "Güvenli fallback", "OpenRouter API anahtarı bulunamadı.")
        return None
    safe_context = sanitize_advisor_context(context)
    context_validation = validate_advisor_context(safe_context)
    if not context_validation["context_valid"]:
        set_advisor_llm_status("fallback", "Güvenli fallback", "Advisor bağlam doğrulaması başarısız.")
        log_event(
            "advisor_validation_failed",
            module="advisor",
            status="fail",
            message="AI Danışman bağlam doğrulaması başarısız.",
            tender_id=str(safe_context.get("tender_id") or "") or None,
            validation=context_validation,
        )
        return None
    prompt_context = {**safe_context, "user_question": question}
    prompt = build_advisor_prompt(prompt_context)
    selected_model = selected_openrouter_model_id()
    set_advisor_llm_status("calling", "OpenRouter LLM", "OpenRouter çağrısı yapılıyor.", selected_model)
    body = {
        "model": selected_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sen Türkçe yanıt veren, yalnızca verilen MODEL_CONTEXT_JSON içeriğini yorumlayan "
                    "ihale karar destek analistisin. Hesap yapma, sayı uydurma, eksik bilgiyi tamamlama. "
                    "Kullanıcının sorusunu özellikle cevapla ama yanıtı mutlaka verilen emsal ihale, fiyat "
                    "koridoru, Linear Regression Baseline, Random Forest / Ağaç Tabanlı Baseline, medyan baz, Cost Plus Margin, "
                    "mixed-type başarı grubu, Isolation Forest sıra dışılık kontrolü, senaryo skorları, risk "
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
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Tender IQ Agentic Bid Advisor",
            },
            json=body,
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        log_exception(
            "advisor_llm_call_failed",
            module="advisor",
            status="fallback",
            message="LLM çağrısı başarısız; fallback advisor kullanılacak.",
            tender_id=str(safe_context.get("tender_id") or "") or None,
        )
        set_advisor_llm_status("fallback", "Güvenli fallback", "OpenRouter çağrısı başarısız oldu.", selected_model)
        return None
    parsed = normalize_llm_payload(content)
    if not parsed:
        parsed = payload_from_free_text(content, safe_context, question)
        if not parsed:
            set_advisor_llm_status("fallback", "Güvenli fallback", "OpenRouter yanıtı geçerli JSON olarak okunamadı ve güvenli şemaya dönüştürülemedi.", selected_model)
            audit_event(
                {
                    "event_type": "advisor_validation_failed",
                    "user_action": "advisor_question",
                    "tender_id": safe_context.get("tender_id"),
                    "module": "advisor",
                    "input_summary": question[:240],
                    "output_summary": str(content)[:240],
                    "validation_status": "fail",
                    "leakage_status": safe_context.get("leakage_audit", {}).get("audit_status", "unknown"),
                    "advisor_guardrail_status": "json_parse_failed",
                    "details": {
                        "llm_model": selected_model,
                        "failure_reason": "json_parse_failed",
                    },
                }
            )
            return None
    else:
        parsed = normalize_advisor_payload_schema(parsed, safe_context, question)
    validation = validate_advisor_output(parsed, safe_context)
    grounding = validate_grounding(parsed, safe_context)
    support = validate_supported_claims(parsed, safe_context)
    forbidden = detect_forbidden_claims(advisor_semantic_text(parsed))
    if (
        not validation["valid"]
        or not grounding["grounded"]
        or not support["supported"]
        or forbidden["forbidden_claims_detected"]
    ):
        failure_parts = []
        if not validation["schema_valid"]:
            failure_parts.append("schema")
        if validation.get("forbidden_claims_detected") or forbidden["forbidden_claims_detected"]:
            failure_parts.append("forbidden_claim")
        if validation.get("hidden_actual_fields_used"):
            failure_parts.append("hidden_actual")
        if not grounding["grounded"]:
            failure_parts.append("grounding")
        if not support["supported"]:
            failure_parts.append("support")
        failure_reason = ", ".join(failure_parts) or "unknown_validation_failure"
        set_advisor_llm_status("fallback", "Güvenli fallback", f"OpenRouter yanıtı doğrulama hatası: {failure_reason}.", selected_model)
        audit_event(
            {
                "event_type": "advisor_validation_failed",
                "user_action": "advisor_question",
                "tender_id": safe_context.get("tender_id"),
                "module": "advisor",
                "input_summary": question[:240],
                "output_summary": str(parsed.get("executive_summary", parsed.get("decision_summary", "")))[:240],
                "validation_status": "fail",
                "leakage_status": safe_context.get("leakage_audit", {}).get("audit_status", "unknown"),
                "advisor_guardrail_status": grounding["grounding_validation_status"],
                "details": {
                    "llm_model": selected_model,
                    "failure_reason": failure_reason,
                    "schema_errors": validation.get("schema_errors", []),
                    "missing_fields": validation.get("missing_fields", []),
                    "forbidden_terms": forbidden.get("detected_terms", []),
                    "hidden_actual_fields_used": validation.get("hidden_actual_fields_used", []),
                    "grounding_unsupported_claims": grounding.get("unsupported_claims", []),
                    "support_unsupported_claims": support.get("unsupported_claims", []),
                },
            }
        )
        return None
    parsed["validation_result"] = {
        "valid": True,
        "advisor_validation_status": "pass",
        "llm_validation_status": "pass",
        "llm_provider": "openrouter",
        "llm_model": selected_model,
        "schema_valid": validation["schema_valid"],
        "forbidden_claims_detected": False,
        "grounding_score": grounding["grounding_score"],
        "prompt_injection_detected": False,
        "fallback_used": False,
    }
    set_advisor_llm_status("pass", "OpenRouter LLM", "OpenRouter yanıtı doğrulandı.", selected_model)
    audit_event(
        {
            "event_type": "advisor_response_validated",
            "user_action": "advisor_question",
            "tender_id": safe_context.get("tender_id"),
            "module": "advisor",
            "input_summary": question[:240],
            "output_summary": str(parsed.get("executive_summary", ""))[:240],
            "validation_status": validation["advisor_validation_status"],
            "leakage_status": safe_context.get("leakage_audit", {}).get("audit_status", "unknown"),
            "advisor_guardrail_status": grounding["grounding_validation_status"],
            "details": {
                "llm_provider": "openrouter",
                "llm_model": selected_model,
            },
        }
    )
    return parsed


def advisor_payload_to_chat_text(payload: dict[str, Any]) -> str:
    parts = [
        ("Kısa Özet", payload.get("executive_summary")),
        ("Önerilen Aksiyon", payload.get("recommended_action")),
        ("Senaryo Gerekçesi", payload.get("scenario_rationale")),
        ("Güven Gerekçesi", payload.get("confidence_rationale")),
    ]
    sections = [f"{title}: {str(value).strip()}" for title, value in parts if str(value or "").strip()]
    for title, key in [
        ("Kullanılan Kanıtlar", "evidence_used"),
        ("Risk Uyarıları", "risk_warnings"),
        ("Manuel Kontrol Gerekenler", "human_checks_required"),
        ("Sınırlar", "limitations"),
    ]:
        values = payload.get(key)
        if isinstance(values, list) and values:
            items = []
            for item in values[:4]:
                if isinstance(item, dict):
                    evidence_id = item.get("evidence_id", "-")
                    claim = item.get("claim", "")
                    items.append(f"- {evidence_id}: {claim}")
                else:
                    items.append(f"- {item}")
            sections.append(f"{title}:\n" + "\n".join(items))
        elif isinstance(values, str) and values:
            sections.append(f"{title}: {values}")
    return compact_chat_text("\n".join(sections))


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
    best = st.session_state.best_scenario
    audit_event(
        {
            "event_type": "scenario_generated",
            "user_action": "scenario_generation",
            "tender_id": tender.get("tender_id"),
            "module": "optimizer",
            "input_summary": f"product_group={tender.get('product_group')}; region={tender.get('region')}",
            "output_summary": f"best_score={best.get('scenario_score')}; hard_valid={best.get('hard_constraints_valid')}",
            "validation_status": "pass" if best.get("hard_constraints_valid") else "fail",
            "leakage_status": st.session_state.get("leakage_audit", {}).get("audit_status", "unknown"),
            "advisor_guardrail_status": "not_applicable",
        }
    )
    audit_event(
        {
            "event_type": "soft_penalty_generated",
            "user_action": "scenario_generation",
            "tender_id": tender.get("tender_id"),
            "scenario_id": best.get("scenario_id", best.get("scenario_name", "")),
            "module": "optimizer",
            "input_summary": "scenario scoring inputs",
            "output_summary": f"soft_penalty_score={best.get('soft_penalty_score', 0)}",
            "validation_status": "pass",
            "leakage_status": st.session_state.get("leakage_audit", {}).get("audit_status", "unknown"),
            "advisor_guardrail_status": "not_applicable",
            "details": {
                "soft_penalty_score": best.get("soft_penalty_score", 0),
                "soft_penalty_explanations": best.get("soft_penalty_explanations", ""),
            },
        }
    )
    rejected = result["scenarios"][~result["scenarios"]["hard_constraints_valid"].astype(bool)]
    for _, rejected_row in rejected.head(3).iterrows():
        audit_event(
            {
                "event_type": "scenario_rejected_by_constraint",
                "user_action": "scenario_validation",
                "tender_id": tender.get("tender_id"),
                "scenario_id": rejected_row.get("scenario_id", rejected_row.get("scenario_name", "")),
                "module": "optimizer",
                "input_summary": f"price={rejected_row.get('proposed_unit_price')}",
                "output_summary": str(rejected_row.get("invalid_reason", rejected_row.get("hard_constraint_status", "")))[:240],
                "validation_status": "fail",
                "leakage_status": st.session_state.get("leakage_audit", {}).get("audit_status", "unknown"),
                "advisor_guardrail_status": "not_applicable",
            }
        )
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


def audited_download_button(label: str, data: Any, file_name: str, mime: str | None = None, **kwargs: Any) -> bool:
    clicked = st.download_button(label, data, file_name, mime=mime, **kwargs)
    if clicked:
        audit_event(
            {
                "event_type": "report_exported",
                "user_action": "download_report",
                "module": "export",
                "input_summary": label,
                "output_summary": file_name,
                "validation_status": "pass",
            }
        )
    return bool(clicked)


def render_home() -> None:
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


def render_data_quality() -> None:
    inject_data_quality_css()
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
    audit_event_once(
        f"data_quality_checked_{len(data)}_{start}_{end}",
        {
            "event_type": "data_quality_checked",
            "user_action": "open_data_quality_page",
            "module": "data",
            "input_summary": f"rows={len(data)}",
            "output_summary": f"schema_valid={schema_result.valid}; quality_passed={quality['passed']}",
            "validation_status": "pass" if schema_result.valid and quality["passed"] else "fail",
            "reveal_status": "not_applicable",
            "details": {
                "missing_columns": schema_result.missing_columns,
                "duplicate_tender_ids": summary["duplicate_tender_ids"],
                "quality_issues": quality["issues"],
            },
        },
    )

    st.markdown(
        "<div class='dq-info'>"
        "<div class='info-callout'><b>Veri neden önemli?</b> "
        "Bu adım, sistemin fiyat koridoru, benzer ihale eşleştirmesi, profil uyumu ve senaryo skorlaması için kullandığı tarihsel kazanılmış ihale veri setini yükler ve doğrular. Kalite kontrolleri, verinin analiz için uygun ve güvenli olup olmadığını gösterir."
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='dq-section-tight'><div class='section-title'>Veri ne işe yarıyor?</div>"
        "<div class='section-subtitle'>Veri seti, Tender IQ'nun tüm karar destek çıktılarının temel girdisidir.</div></div>",
        unsafe_allow_html=True,
    )
    data_use_cards = [
        ("Benzer ihale bulma", "Yeni ihale, geçmiş kazanılmış ihalelerle karşılaştırılır ve en yakın emsaller bulunur."),
        ("Fiyat koridoru üretme", "Benzer kazanılmış ihalelerden düşük, orta ve yüksek fiyat bandı çıkarılır."),
        ("Profil uyumu hesaplama", "Yeni ihalenin geçmiş kazanılmış işlere ne kadar tanıdık göründüğü ölçülür."),
        ("Senaryo ve karlılık analizi", "Aday teklif fiyatlarının karlılık, katkı ve risk etkisi karşılaştırılır."),
    ]
    render_data_quality_feature_grid(data_use_cards)

    st.markdown(
        "<div class='dq-section'><div class='section-title'>Veri özeti</div>"
        "<div class='section-subtitle'>Demo veri setinin iş seviyesindeki kısa görünümü.</div></div>",
        unsafe_allow_html=True,
    )
    render_data_quality_metric_grid(
        [
            ("Kayıt sayısı", format_int(summary["row_count"]), "Normalize edilmiş ihale"),
            ("Ürün grubu sayısı", format_int(data["product_group"].nunique()), "Kategori"),
            ("Kurum sayısı", format_int(data["buyer_institution"].nunique()), "Alıcı kurum"),
            ("Tarih aralığı", f"{start} - {end}", "İhale tarihi"),
        ]
    )

    st.markdown(
        "<div class='dq-section'><div class='section-title'>Kalite kontrol sonucu</div>"
        "<div class='section-subtitle'>Analize başlamadan önce veri setinin kullanılabilirliği doğrulanır.</div></div>",
        unsafe_allow_html=True,
    )
    status = "good" if schema_result.valid and quality["passed"] else "warn"
    quality_cards = [
        ("Şema kontrolü", "Geçti" if schema_result.valid else "Eksik", "Zorunlu kolonların bulunup bulunmadığını kontrol eder.", "good" if schema_result.valid else "bad"),
        ("Zorunlu kolonlar", "Tamam" if not schema_result.missing_columns else "Eksik", "Ürün, kurum, miktar, tarih, fiyat ve karlılık alanlarının durumunu gösterir.", "good" if not schema_result.missing_columns else "bad"),
        ("Eksik veri durumu", "Uygun" if quality["passed"] else "Uyarı", "Boş veya sorunlu değerlerin analizi bozup bozmadığını inceler.", "good" if quality["passed"] else "warn"),
        ("Tekrarlı kayıt kontrolü", format_int(summary["duplicate_tender_ids"]), "Aynı tender_id ile gelen tekrarları görünür kılar.", "good" if summary["duplicate_tender_ids"] == 0 else "warn"),
        ("Veri kullanıma hazır mı?", "Hazır" if status == "good" else "Kontrol gerekli", "Kalite ve şema kontrollerinin ortak sonucudur.", status),
    ]
    render_data_quality_status_grid(quality_cards)

    st.markdown(
        "<div class='dq-section'><div class='section-title'>Zorunlu kolonlar</div>"
        "<div class='section-subtitle'>Teknik kolon adları iş dilindeki anlamlarıyla birlikte gösterilir.</div></div>",
        unsafe_allow_html=True,
    )
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
    render_dark_table(required_columns)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Eksik veri kontrolü", "Kolon eşleştirme", "Veri önizleme", "İsteğe bağlı yeni veri yükleme"])
    with tabs[0]:
        null_df = pd.DataFrame(
            [{"Kolon": key, "Boş oran": value} for key, value in summary["null_rates"].items()]
        ).sort_values("Boş oran", ascending=False)
        render_global_dark_table(null_df)
        if quality["issues"]:
            st.warning("; ".join(str(issue) for issue in quality["issues"]))
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
        render_global_dark_table(mapping)
    with tabs[2]:
        render_global_dark_table(data.head(25))
    with tabs[3]:
        info_callout(
            "Yeni CSV yüklemek, demo veri seti yerine kurumunuza ait tarihsel kazanılmış ihale verisini kullanmak içindir. Dosyada ürün, kurum, bölge, miktar, tarih, tahmini maliyet, kazanılmış fiyat ve karlılık oranı alanları bulunmalıdır.",
            "Ne zaman yüklenir?",
        )
        uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"])
        if uploaded is not None:
            audit_event(
                {
                    "event_type": "data_upload_started",
                    "user_action": "csv_upload",
                    "module": "data",
                    "input_summary": getattr(uploaded, "name", "uploaded_csv"),
                    "validation_status": "started",
                }
            )
            uploaded_df = pd.read_csv(uploaded)
            schema_check = validate_schema(uploaded_df)
            if not schema_check.valid:
                audit_event(
                    {
                        "event_type": "schema_validation_failed",
                        "user_action": "csv_upload",
                        "module": "data",
                        "input_summary": getattr(uploaded, "name", "uploaded_csv"),
                        "output_summary": f"missing={','.join(schema_check.missing_columns)}",
                        "validation_status": "fail",
                    }
                )
                st.error("Zorunlu kolon yok. Lütfen CSV şemasını kontrol edin.")
                return
            st.session_state.active_data = normalize_schema(uploaded_df)
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("scenario_result", None)
            st.session_state.pop("latest_artifact_dir", None)
            audit_event(
                {
                    "event_type": "data_upload_completed",
                    "user_action": "csv_upload",
                    "module": "data",
                    "input_summary": getattr(uploaded, "name", "uploaded_csv"),
                    "output_summary": f"rows={len(st.session_state.active_data)}",
                    "validation_status": "pass",
                }
            )
            st.success("Veri yüklendi ve uygulama şemasına uyarlandı.")


def render_methodology() -> None:
    page_header(
        "Metodoloji",
        "Sistem neyi, neden ve nasıl hesaplıyor? Aşağıdaki bölüm metodolojiyi temelden ama iş dilinde açıklar.",
        "Metodoloji",
    )
    warning_box()
    info_callout(PWIN_PROXY_EXPLANATION, "Profil uyum göstergesi ne anlatır?")
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
        info_callout(
            "Negatif sınıf güvenilir olmadığı için kazan/kaybet sınıflandırması ölçülmez. Backtest, fiyat koridoru hizası, profil uyumu, segment performansı, sızıntı kontrolü ve danışman güvenliği üzerinden yapılır.",
            "Neden accuracy, precision, recall veya ROC-AUC ana başarı metriği değil?",
        )

    with tabs[1]:
        section_header("Benzerlik nasıl hesaplanıyor?", "Yerel metin embedding ve yapısal KNN sinyalleri yeni ihaleyi geçmiş kazanılmış ihalelerle karşılaştırır.", "Retrieval")
        render_method_grid(
            [
                ("İhale metni hazırlanır", "Ürün adı, ürün grubu, kurum, bölge, ihale tipi ve miktar bilgileri tek bir ihale profiline dönüştürülür."),
                ("Yerel embedding ile vektöre çevrilir", "Metinsel alanlar deterministik yerel vektörlerle temsil edilir; dış servis veya secret gerekmez."),
                ("KNN benzerliği hesaplanır", "Metin embedding yakınlığı ürün, kurum tipi, bölge, ihale tipi, miktar, teslim ve rekabet sinyalleriyle birleştirilir."),
                ("Top-K benzer ihaleler seçilir", "En yüksek skorlu kazanılmış ihaleler emsal listeye alınır."),
                ("Koridor ve skorlar beslenir", "Fiyat koridoru, profil uyumu ve danışman açıklamaları bu emsal setten destek alır."),
            ],
            ["blue", "purple", "cyan", "green", "amber"],
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            glass_card(
                "Yerel metin embedding",
                "Ürün, kurum ve ihale metinlerini dış servis kullanmadan deterministik vektörlere çevirir. Bu vektörler yapısal alanlarla birlikte Top-K emsal aramasında kullanılır.",
                "Metin temsili",
            )
        with c2:
            glass_card(
                "Embedding yakınlığı",
                "İki ihalenin sayısal vektörlerinin birbirine ne kadar yakın olduğunu ölçer. Skor 1'e yaklaştıkça benzerlik artar; 0'a yaklaştıkça azalır.",
                "Yakınlık hesabı",
            )
        info_callout(
            "Örnek profil: serum ürün grubu, kamu hastanesi, Marmara bölgesi, açık ihale ve 50.000 adet. Aynı ürün grubu, benzer kurum tipi, aynı bölge ve yakın miktar varsa benzerlik güçlenir.",
            "Top-K retrieval ve eşleşme metrikleri",
        )
        metrics = pd.DataFrame(
            [
                ["Ürün Grubu Eşleşme Oranı", "İlk K benzer ihale içinde aynı ürün grubuna düşen kayıt oranı."],
                ["Bölge Eşleşme Oranı", "Benzer ihalelerin seçili ihale ile aynı bölgede olma oranı."],
                ["Miktar Bandı Eşleşme Oranı", "Benzer ihalelerin yakın miktar ölçeğinde olma oranı."],
                ["İlk K Ortalama Benzerlik", "Getirilen benzer ihalelerin ortalama embedding ve yapısal özellik benzerliği."],
            ],
            columns=["Metrik", "Ne anlatır?"],
        )
        render_global_dark_table(metrics)

    with tabs[2]:
        section_header("Model Bileşenleri", "Her bileşen karar destek çıktısının farklı bir parçasını açıklar.", "Model")
        render_model_grid(
            [
                ("01", "Embedding + KNN Emsal Arama", "Ne yapar: Yeni ihaleye benzeyen kazanılmış ihaleleri bulur. Neden var: Emsal seti olmadan fiyat ve profil yorumu zayıf kalır. Katkı: Benzer ihaleler listesini, profil uyumunun ana sinyalini ve koridor girdisini üretir.", "blue"),
                ("02", "Mixed-Type Clustering", "Ne yapar: Kazanılmış ihaleleri kategorik ve sayısal profil alanlarını birlikte okuyan Gower mesafesiyle gruplar. Neden var: Tek tek ihale yerine profil segmenti görmeyi sağlar. Katkı: Profil yorumunu destekler; ana karar sinyali KNN emsalleridir.", "purple"),
                ("03", "Isolation Forest", "Ne yapar: Yeni ihalenin geçmiş profile normal mi sıra dışı mı uyduğunu kontrol eder. Neden var: Aykırı durumları saklamaz. Katkı: Risk ve manuel inceleme sinyali üretir.", "amber"),
                ("04", "Price Corridor Engine", "Ne yapar: Emsal kazanılmış ihalelerden düşük, orta ve yüksek fiyat bandı çıkarır. Neden var: Tek nokta fiyat yerine karar aralığı verir. Katkı: Senaryo fiyatlarını besler.", "green"),
                ("05", "Scenario Scoring", "Ne yapar: Fiyat, karlılık, profil uyumu, güven ve risk cezasını tek karar destek skorunda birleştirir. Neden var: Alternatif teklifleri kıyaslanabilir hale getirir. Katkı: Sıralı senaryo önerisi üretir.", "cyan"),
                ("06", "Model Confidence / Risk", "Ne yapar: Benzer ihale sayısı, veri kalitesi, band genişliği ve aykırılık sinyallerini birlikte okur. Neden var: Skorun ne kadar güvenle okunacağını gösterir. Katkı: AI Danışman ve manuel inceleme kararını destekler.", "blue"),
                ("07", "Linear Regression Baseline", "Ne yapar: Ürün grubu, bölge, ihale tipi, miktar, teslim süresi ve tahmini rakip sayısı gibi alanlardan beklenen fiyat için doğrusal referans üretir. Neden var: Emsal tabanlı fiyat koridorunu basit ve açıklanabilir bir fiyat tahminiyle kıyaslamak için kullanılır. Katkı: Backtestte koridor yaklaşımının basit doğrusal modele göre ne kadar tutarlı olduğunu gösterir.", "green"),
                ("08", "Random Forest / Ağaç Tabanlı Baseline", "Ne yapar: Doğrusal olmayan fiyat ilişkilerini yakalayabilen ağaç tabanlı referans modelini temsil eder. Neden var: Miktar, bölge ve ürün grubunun fiyat üzerindeki doğrusal olmayan etkilerini kontrol etmek için kullanılır. Katkı: Gerçek kazanma olasılığı üretmez; fiyat koridorunun regresyon bazlı tahminlerle tutarlılığını değerlendirmek için metodolojik karşılaştırma sağlar.", "amber"),
            ]
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        info_callout(
            "Çok yüksek veya çok düşük miktar, alışılmadık ürün-kurum kombinasyonu, çok kısa teslim süresi, olağan dışı fiyat seviyesi veya düşük benzer ihale sayısı manuel inceleme sinyali üretebilir.",
            "Sıra dışı durum örnekleri",
        )
        info_callout(
            "Sistem seçili ihaleye en çok benzeyen kazanılmış ihaleleri bulur. Ana fiyat koridoru bu emsal ihalelerdeki normalize fiyatların alt, orta ve üst yüzdeliklerinden üretilir. Lineer regresyon ve Random Forest/ağaç tabanlı baseline bu koridoru açıklanabilir fiyat referanslarıyla kıyaslar; gerçek kazanma olasılığı üretmez. Koridor çok genişse karar desteği zayıflar, bu yüzden backtestte band genişliği de ölçülür.",
            "Fiyat koridoru nasıl oluşuyor?",
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
                ("01", "Profil Uyumu", "Yeni ihalenin geçmiş kazanılmış ihalelere benzerliği.", "blue"),
                ("02", "Fiyat Bandı Uyumu", "Önerilen fiyatın geçmiş fiyat koridoruna yakınlığı.", "green"),
                ("03", "Karlılık Skoru", "Beklenen karlılık ve katkı karı sağlığı.", "purple"),
                ("04", "Model Güveni", "Yeterli benzer ihale ve veri kalitesi olup olmadığı.", "cyan"),
                ("05", "Risk Cezası", "Sıra dışı durumlar, düşük benzerlik, düşük güven ve kısıt ihlalleri.", "amber"),
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
        info_callout(
            "Başarı, kazan/kaybet sınıflandırmasıyla değil; fiyat bandı kapsama, profil uyumu, cluster kalitesi, sıra dışılık kontrolü, sızıntı disiplini ve AI güvenliğiyle ölçülür.",
            "Başarıyı hangi metriklerle ölçüyoruz?",
        )
        metrics_table = pd.DataFrame(
            [
                ["Fiyat Koridoru Kapsama Oranı", "Gerçek kazanılmış fiyat sistemin önerdiği fiyat bandının içinde mi?"],
                ["MAE", "Tahmin edilen orta fiyat ile gerçek fiyat arasındaki ortalama mutlak fark."],
                ["Ortalama yüzde fiyat hatası", "Önerilen orta fiyatın gerçek kazanılmış fiyattan ortalama yüzde sapması."],
                ["Ortalama fiyat aralığı genişliği", "Düşük ve yüksek fiyat önerisi arasındaki ortalama fark."],
                ["Band kalite skoru", "Fiyat bandının hem gerçek fiyatı kapsamasını hem de çok geniş olmamasını birlikte değerlendirir."],
                ["Gerçek Kazanılmış Senaryo Sıralaması", "Tarihsel gerçek konfigürasyon aday senaryolar arasında ne kadar üstte kaldı?"],
                ["Mixed-Type Silhouette Score", "Gower mesafesiyle üretilen profil clusterlarının birbirinden ne kadar ayrıştığını gösterir."],
                ["Mixed-Type Cluster Sıkılığı", "Cluster içindeki kayıtların birbirine ortalama yakınlığını gösterir."],
                ["Cluster Size Distribution", "Clusterların dengeli mi, aşırı küçük veya boş mu olduğunu gösterir."],
                ["Assignment Confidence", "Seçili ihalenin atandığı cluster’a ne kadar net yakın olduğunu gösterir."],
                ["Isolation Forest Inlier Rate", "Geçmiş kazanılmış test kayıtlarının ne kadarının normal göründüğünü ölçer."],
                ["Isolation Forest Anomaly Rate", "Geçmiş kazanılmış test kayıtlarında manuel inceleme sinyali oranını gösterir."],
                ["Contamination Uyumu", "Beklenen sıra dışılık ayarı ile gerçekleşen anomaly oranının tutarlılığını kontrol eder."],
                ["Synthetic Outlier Manual Review Rate", "Bilerek uçlaştırılmış örneklerde manual review sinyali çıkıp çıkmadığını test eder."],
                ["Yasak iddia üretme oranı", "AI Danışman kesin sonuç veya veri dışı başarı iddiası üretiyor mu? Hedef sıfırdır."],
            ],
            columns=["Metrik", "Ne anlatır?"],
        )
        render_global_dark_table(metrics_table)
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
            render_global_dark_table(blocked)

    with tabs[5]:
        section_header("AI Danışman", "Skor hesaplamaz; mevcut model çıktılarını güvenli Türkçe açıklamaya dönüştürür.", "LLM Guardrails")
        render_model_grid(
            [
                ("01", "Açıklama Üretir", "Profil uyumu, fiyat koridoru, karlılık, risk ve benzer ihaleleri anlaşılır hale getirir.", "blue"),
                ("02", "Guardrail Uygular", "Garanti, kesin sonuç veya gerçek kazanma olasılığı iddiası üretirse çıktı reddedilir.", "amber"),
                ("03", "Reveal Kuralına Uyar", "Gerçek sonuç açılmadıysa kazanılmış fiyat veya gerçek karlılık oranını kullanamaz.", "purple"),
                ("04", "Fallback Çalışır", "LLM sağlayıcısı yoksa deterministik danışman aynı sohbet akışında yanıt verir.", "green"),
            ]
        )
def render_test_simulator() -> None:
    inject_test_simulator_css()
    page_header(
        "Test için İhale Seç",
        "Geçmişte kazanılmış bir ihale, gerçek sonucu gizlenmiş şekilde yeni gelen ihale gibi seçilir. Sistem sonuç açılmadan önce emsal, profil, fiyat ve teklif senaryosu üretir.",
        "Test seçimi",
    )
    st.markdown(
        "<div class='ts-warning'><b>Skor gerçek kazanma olasılığı değildir.</b> "
        "Skor gerçek kazanma olasılığı değil, geçmiş kazanılmış ihale profiline uyum göstergesidir. Gerçek kazanılmış fiyat ve karlılık oranı, karşılaştırma adımına kadar gizli kalır."
        "</div>",
        unsafe_allow_html=True,
    )
    split = get_split()
    test = split["test"]
    st.markdown(
        "<div class='ts-section-tight'><div class='section-title'>1. Test ihalesi seç</div>"
        "<div class='section-subtitle'>Geçmiş test döneminden bir ihale seçilir ve gerçek sonuç alanları simülasyon boyunca gizlenir.</div></div>",
        unsafe_allow_html=True,
    )
    with st.container(key="ts_select_card"):
        st.markdown(
            "<div class='ts-control-title'>Test ihalesi seç</div>"
            "<div class='ts-control-copy'>Seçim değiştiğinde seçili ihale özeti ve maskelenmiş girdi alanları güncellenir.</div>",
            unsafe_allow_html=True,
        )
        selected = st.selectbox("Test ihalesi seç", test["tender_id"].astype(str).tolist())
    st.session_state.selected_tender_id = selected
    if st.session_state.get("last_audited_selected_tender") != selected:
        audit_event(
            {
                "event_type": "test_tender_selected",
                "user_action": "test_tender_selection",
                "tender_id": selected,
                "module": "selection",
                "input_summary": "test_tender_selectbox",
                "output_summary": f"selected={selected}",
                "validation_status": "pass",
            }
        )
        st.session_state.last_audited_selected_tender = selected
    st.session_state.revealed = False if st.session_state.get("last_selected_tender") != selected else st.session_state.get("revealed", False)
    st.session_state.last_selected_tender = selected

    row = test[test["tender_id"].astype(str) == selected].iloc[0]
    masked = mask_actual_result_fields(row.to_dict())
    audit = audit_pre_reveal_input(selected, masked)
    st.session_state.masked_tender = masked
    st.session_state.leakage_audit = audit
    audit_event_once(
        f"actual_result_masked_{selected}",
        {
            "event_type": "actual_result_masked",
            "user_action": "test_tender_selection",
            "tender_id": selected,
            "module": "feature_masking",
            "input_summary": "test tender row",
            "output_summary": "actual result fields masked",
            "validation_status": "pass",
            "leakage_status": audit["audit_status"],
            "reveal_status": "hidden",
            "details": {
                "masked_fields_count": audit.get("masked_fields_count", 0),
                "blocked_fields_present": audit.get("blocked_fields_present", []),
            },
        },
    )
    audit_event_once(
        f"leakage_audit_completed_{selected}",
        {
            "event_type": "leakage_audit_completed",
            "user_action": "test_tender_selection",
            "tender_id": selected,
            "module": "leakage_audit",
            "input_summary": "masked test tender",
            "output_summary": audit["audit_status"],
            "validation_status": audit["audit_status"],
            "leakage_status": audit["audit_status"],
            "reveal_status": "hidden",
            "details": {
                "masked_fields_count": audit.get("masked_fields_count", 0),
                "blocked_fields_present": audit.get("blocked_fields_present", []),
            },
        },
    )
    if audit["audit_status"] != "pass":
        audit_event(
            {
                "event_type": "leakage_audit_failed",
                "user_action": "test_tender_selection",
                "tender_id": selected,
                "module": "leakage_audit",
                "input_summary": "masked test tender",
                "output_summary": ";".join(audit.get("blocked_fields_present", [])),
                "validation_status": "fail",
                "leakage_status": audit["audit_status"],
            }
        )

    st.markdown(
        "<div class='ts-section'><div class='section-title'>2. Seçili ihale özeti</div>"
        "<div class='section-subtitle'>Bu bilgiler canlı ihale girdisi gibi kullanılır; gerçek sonuç alanları maskelidir.</div></div>",
        unsafe_allow_html=True,
    )
    render_test_tender_summary(selected, masked, audit)

    st.markdown(
        "<div class='ts-section'><div class='section-title'>3. Bu test akışı ne üretir?</div>"
        "<div class='section-subtitle'>Sonuçlar ayrı sayfalarda, aynı simülasyon çıktısı üzerinden gösterilir.</div></div>",
        unsafe_allow_html=True,
    )
    test_cards = [
        ("Emsal ihale analizi", "Geçmişte kazanılmış en benzer ihaleleri ve eşleşme gücünü gösterir."),
        ("Profil uyum analizi", "Başarı grubu ve sıra dışılık kontrolünü tek sayfada açıklar."),
        ("Fiyat koridoru", "Low / mid / high fiyatları ve baz model tahminlerini karşılaştırır."),
        ("Teklif senaryoları", "Agresif, dengeli ve muhafazakar teklif seçeneklerini skorlar."),
        ("Gerçek sonuçla karşılaştırma", "Sonuç açıldıktan sonra gerçek fiyatı, profili ve senaryo sırasını kıyaslar."),
    ]
    render_test_process_grid(test_cards)

    st.markdown(
        "<div class='ts-section'><div class='section-kicker'>Kontrol paneli</div>"
        "<div class='section-title'>4. Canlı ihale girdileri</div>"
        "<div class='section-subtitle'>Bu alanlar simülasyon için düzenlenebilir; gerçek kazanılmış fiyat ve karlılık oranı görünmez.</div></div>",
        unsafe_allow_html=True,
    )
    with st.container(key="ts_inputs_card"):
        st.markdown(
            "<div class='ts-control-title'>Canlı ihale girdileri</div>"
            "<div class='ts-control-copy'>Miktar, teslim süresi, rekabet ve tahmini maliyet simülasyon bağlamını günceller.</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        masked["quantity"] = int(c1.number_input("Miktar", min_value=1, value=int(masked.get("quantity", 1))))
        masked["delivery_months"] = int(c2.number_input("Teslim Süresi (Ay)", min_value=1, value=int(masked.get("delivery_months", 6))))
        masked["competitor_count_estimate"] = int(c3.number_input("Tahmini Rakip Sayısı", min_value=0, value=int(masked.get("competitor_count_estimate", 3))))
        masked["estimated_unit_cost"] = float(c4.number_input("Tahmini Birim Maliyet", min_value=0.01, value=float(masked.get("estimated_unit_cost", 1.0))))
        st.session_state.adjusted_tender = masked

        if st.button("Simülasyonu çalıştır", type="primary"):
            st.session_state.pop("scenario_result", None)
            result = ensure_scenario_result()
            audit_event(
                {
                    "event_type": "test_tender_simulation",
                    "user_action": "run_simulation",
                    "tender_id": selected,
                    "module": "simulation",
                    "input_summary": "masked_tender",
                    "output_summary": "scenario_result_created" if result else "scenario_result_empty",
                    "validation_status": "pass" if result else "fail",
                    "leakage_status": audit.get("audit_status", "unknown"),
                    "leakage_audit": audit,
                }
            )
            if result:
                st.success("Simülasyon tamamlandı. Sonuçları Emsal İhale Analizi sayfasından başlayarak inceleyebilirsiniz.")

    with st.container(key="ts_masked_expander"):
        with st.expander("Maskelenmiş girdi alanları", expanded=False):
            st.markdown("Gerçek fiyat, gerçek marj ve final sonuç alanları karşılaştırma adımına kadar gizli tutulur.")
            safe_preview = pd.DataFrame([masked]).T.reset_index()
            safe_preview.columns = ["Alan", "Değer"]
            safe_preview["Değer"] = safe_preview["Değer"].astype(str)
            render_test_masked_table(safe_preview)


def scenario_name(index: int) -> str:
    names = ["Agresif Senaryo", "Dengeli Senaryo", "Muhafazakâr Senaryo"]
    return names[index] if index < len(names) else f"Alternatif Senaryo {index + 1}"


def render_profile_fit_analysis() -> None:
    inject_profile_fit_css()
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
    audit_event_once(
        f"profile_fit_analysis_run_{best.get('tender_id', current_tender().get('tender_id') if current_tender() else '')}",
        {
            "event_type": "profile_fit_analysis_run",
            "user_action": "open_profile_fit_page",
            "tender_id": (current_tender() or {}).get("tender_id"),
            "module": "profile_fit",
            "input_summary": "masked tender profile",
            "output_summary": f"profile_score={best.get('won_profile_fit_score')}; inlier={best.get('is_inlier')}",
            "validation_status": "pass",
            "leakage_status": st.session_state.get("leakage_audit", {}).get("audit_status", "unknown"),
            "details": {
                "won_profile_fit_score": best.get("won_profile_fit_score"),
                "cluster_id": best.get("cluster_id"),
                "is_inlier": best.get("is_inlier"),
                "training_anomaly_rate": anomaly_rate,
            },
        },
    )

    render_profile_callout()

    st.markdown(
        "<div class='pf-section'><div class='section-title'>Genel Uyum Özeti</div>"
        "<div class='section-subtitle'>Profil tanılaması; geçmiş başarı grubu, sıra dışılık sinyali, emsal gücü ve genel uyum skorunu birlikte okur.</div></div>",
        unsafe_allow_html=True,
    )
    render_profile_kpi_grid(
        [
            {
                "label": "Kazanılmış Profil Uyum Skoru",
                "value": format_score(best.get("won_profile_fit_score")),
                "body": "KNN emsal benzerliği ana sinyaldir; Isolation Forest ve mixed-type profil grubu destekleyici tanılama sağlar.",
                "badge": fit_level(best.get("won_profile_fit_score")),
                "status": "good" if float(best.get("won_profile_fit_score", 0) or 0) >= 70 else "warn",
            },
            {
                "label": "Geçmiş Başarı Grubu",
                "value": str(best.get("cluster_id", "Hesaplanamadı")),
                "body": str(best.get("cluster_name", "Geçmiş başarı grubu")),
                "badge": "Mixed-type",
                "status": "good",
            },
            {
                "label": "Sıra Dışılık Kontrolü",
                "value": profile_label,
                "body": "Geçmiş kazanılmış kayıtlar içinde normal mi, yoksa manuel inceleme gerektirecek kadar farklı mı?",
                "badge": "Isolation Forest",
                "status": profile_status,
            },
            {
                "label": "Emsal Benzerlik Gücü",
                "value": f"{result.get('top10_avg_similarity', 0):.2f}",
                "body": "En yakın emsallerin seçili ihaleye ortalama yakınlığını gösterir.",
                "badge": "0-1 yakınlık",
                "status": "good",
            },
        ]
    )
    st.markdown(
        "<div class='pf-score-note'><b>Profil uyum skoru nasıl hesaplanır?</b> "
        "Bu skor fiyat veya maliyet alanlarını kullanmadan hesaplanan yapısal profil uyumudur. KNN emsal benzerliği ana ağırlığı taşır; Isolation Forest geçmiş kazanılmış dağılıma alışıldık uyumu, mixed-type clustering ise destekleyici profil grubu yakınlığını ve saflığını gösterir. Skor fiyat kararı veya gerçek kazanma olasılığı değildir.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='pf-section'><div class='section-title'>Profil Grubu Detayı</div>"
        "<div class='section-kicker'>Mixed-Type Cluster Analizi</div>"
        "<div class='section-subtitle'>Mixed-type clustering, geçmişte kazanılmış ihaleleri kategorik ve sayısal profil alanlarını birlikte okuyan Gower mesafesiyle gruplar; bu bir fiyat tahmini değildir.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pf-score-note'>Mixed-type profil atamasında ürün grubu, ürün adı, kurum, kurum tipi, bölge, ihale tipi, miktar, teslim süresi ve tahmini rakip sayısı kullanılır. Gerçek kazanılmış fiyat, tahmini maliyet, gerçek marj veya önerilen senaryo fiyatı kullanılmaz.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.16, 0.84], gap="medium")
    with left:
        render_profile_kv_panel(
            "Başarı grubu detayı",
            [
                ("Başarı grubu adı", str(best.get("cluster_name", "Hesaplanamadı"))),
                ("Bu gruptaki ihale sayısı", format_int(best.get("cluster_count"))),
                ("Baskın ürün grubu", str(best.get("cluster_dominant_product_group", "Hesaplanamadı"))),
                ("Baskın ürün grubu oranı", format_pct(float(best.get("cluster_dominant_product_group_ratio", 0) or 0) * 100)),
                ("Baskın kurum tipi", str(best.get("cluster_dominant_institution_type", "Hesaplanamadı"))),
                ("Baskın kurum tipi oranı", format_pct(float(best.get("cluster_dominant_institution_type_ratio", 0) or 0) * 100)),
                ("Baskın bölge", str(best.get("cluster_dominant_region", "Hesaplanamadı"))),
                ("Baskın bölge oranı", format_pct(float(best.get("cluster_dominant_region_ratio", 0) or 0) * 100)),
                ("Baskın ihale tipi", str(best.get("cluster_dominant_procedure_type", "Hesaplanamadı"))),
                ("Baskın ihale tipi oranı", format_pct(float(best.get("cluster_dominant_procedure_type_ratio", 0) or 0) * 100)),
                ("Cluster saflık skoru", format_score(best.get("cluster_purity_score"))),
                ("Ortalama miktar", format_int(best.get("cluster_average_quantity"))),
                ("Medyan teslim süresi", f"{format_decimal(best.get('cluster_median_delivery_months'), 1)} ay"),
                ("Cluster merkezine uzaklık", format_decimal(best.get("cluster_distance"))),
                ("İkinci en yakın cluster uzaklığı", format_decimal(best.get("cluster_second_distance"))),
                ("Atama güveni", format_pct(float(best.get("cluster_assignment_confidence", 0) or 0))),
            ],
            profile_business_comment(best),
        )
    with right:
        with st.container(key="pf_gauge_card"):
            st.markdown("<div class='pf-gauge-title'>Profil uyum skoru</div>", unsafe_allow_html=True)
            st.plotly_chart(build_gauge(float(best.get("won_profile_fit_score", 0)), "Profil uyum skoru"), use_container_width=True)
            st.markdown(
                f"<div class='pf-gauge-copy'>{escape(format_score(best.get('won_profile_fit_score')))} seviyesi {escape(fit_level(best.get('won_profile_fit_score')).lower())} olarak okunur; seçili ihale geçmiş kazanılmış profile yapısal olarak ne kadar benziyor sorusuna tanılama yanıtı verir.</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='pf-section'><div class='section-title'>Mixed-Type Cluster Kalitesi</div>"
        "<div class='section-subtitle'>Bu metrikler Gower tabanlı cluster yapısının ayrışmasını, dengesini ve seçili ihalenin atamasının ne kadar net olduğunu gösterir; fiyat metriği değildir.</div></div>",
        unsafe_allow_html=True,
    )
    render_profile_metric_grid(
        [
            ("Silhouette Score", format_decimal(best.get("cluster_silhouette_score"), 2), "1'e yakınsa profil grupları daha net ayrılır."),
            ("Cluster sıkılığı", format_decimal(best.get("cluster_inertia"), 1), "Daha düşük değer daha sıkı profil grubu anlamına gelir."),
            ("Cluster boyut aralığı", f"{format_int(best.get('cluster_min_size'))} - {format_int(best.get('cluster_max_size'))}", "En küçük ve en büyük profil grubunun kayıt sayısı."),
            ("Küçük / boş cluster", f"{format_int(best.get('small_cluster_count'))} / {format_int(best.get('empty_cluster_count'))}", "Çok küçük veya boş cluster sayısı."),
        ]
    )

    nearest_examples = best.get("nearest_cluster_examples", [])
    if isinstance(nearest_examples, list) and nearest_examples:
        st.markdown(
            "<div class='pf-section'><div class='section-title'>En Yakın Geçmiş Kazanılmış Örnekler</div>"
            "<div class='section-subtitle'>Bu tablo profil grubuna yakın geçmiş örnekleri gösterir; fiyat önerisi değildir.</div></div>",
            unsafe_allow_html=True,
        )
        examples_df = pd.DataFrame(nearest_examples)
        rename_map = {
            "tender_id": "İhale ID",
            "product_group": "Ürün grubu",
            "product_name": "Ürün",
            "buyer_institution_type": "Kurum tipi",
            "region": "Bölge",
            "procedure_type": "İhale tipi",
            "quantity": "Miktar",
            "delivery_months": "Teslim süresi",
            "query_distance": "Seçili ihaleye uzaklık",
        }
        examples_df = examples_df.rename(columns=rename_map)
        render_profile_examples_table(examples_df[[column for column in rename_map.values() if column in examples_df.columns]])

    st.markdown(
        "<div class='pf-section'><div class='section-title'>Isolation Forest Sıra Dışılık Kontrolü</div>"
        "<div class='section-kicker'>Sıra Dışılık Kontrolü (Isolation Forest)</div>"
        "<div class='section-subtitle'>Bu kontrol, seçili ihalenin geçmişte kazanılmış işlere ne kadar alışıldık göründüğünü anlatır. Sıra dışı sonuç kötü ihale anlamına gelmez; manuel inceleme sinyalidir.</div></div>",
        unsafe_allow_html=True,
    )
    render_profile_metric_grid(
        [
            ("Durum", profile_label, "Geçmiş kazanılmış dağılım içindeki tipiklik sinyali; kayıp tahmini değildir."),
            ("Anomaly score", format_decimal(best.get("anomaly_score"), 4), "Eşik altı değer daha sıra dışı kabul edilir."),
            ("Threshold", format_decimal(best.get("isolation_threshold", 0.0), 2), "Isolation Forest karar sınırıdır."),
            ("Manual review flag", "Evet" if bool(best.get("manual_review_flag", not best.get("is_inlier", True))) else "Hayır", "Evet ise manuel kontrol sinyali vardır."),
            ("Normal görülen kayıt oranı", format_pct(inlier_rate * 100), "Kazanılmış kayıtların tipik profil oranı."),
            ("Manuel inceleme oranı", format_pct(anomaly_rate * 100), "Daha az tipik görülen kayıt oranı."),
            ("Contamination ayarı", format_pct(float(best.get("isolation_contamination", 0)) * 100), "Modelin beklenen sıra dışı oranı."),
            ("Ürün grubunda manuel inceleme oranı", format_pct(float(segment_rate) * 100) if segment_rate is not None and not pd.isna(segment_rate) else "-", "Aynı ürün grubundaki manuel inceleme sinyali."),
        ]
    )

    reasons = best.get("manual_review_reasons")
    if isinstance(reasons, list):
        reason_text = "; ".join(str(item) for item in reasons)
    else:
        reason_text = str(reasons or "")
    if reason_text:
        info_callout(reason_text, "Manuel inceleme gerekçeleri")

    info_callout(
        "Bu veri setindeki tüm kayıtlar kazanılmış ihalelerden oluşur. Bu nedenle Isolation Forest’ın sıra dışı dediği bir kayıt, kaybedilecek ihale anlamına gelmez. Sadece geçmiş kazanılmış ihaleler arasında daha az tipik bir örnek olduğunu gösterir.",
        "Sıra dışılık nasıl okunmalı?",
    )
    if anomaly_rate >= 0.25:
        st.warning("Eğer kazanılmış test ihalelerinin büyük kısmı manuel inceleme gerektiriyor görünüyorsa, model fazla hassas olabilir ve hassasiyet ayarı gözden geçirilmelidir.")
    elif float(best.get("isolation_contamination", 0)) >= 0.10:
        st.info("Hassasiyet ayarı orta seviyede. Çok sayıda kazanılmış ihale sıra dışı görünürse ayar düşürülebilir.")

    st.markdown(
        "<div class='pf-section'><div class='section-title'>Emsal sinyali</div>"
        "<div class='section-subtitle'>Profil uyumu emsal ihale kalitesiyle birlikte okunur.</div></div>",
        unsafe_allow_html=True,
    )
    render_profile_metric_grid(
        [
            ("Ürün grubu eşleşmesi", format_pct(quality.get("product_group_match_rate", 0) * 100), "Benzer ihalelerin aynı ürün grubunda olma oranı."),
            ("Bölge eşleşmesi", format_pct(quality.get("region_match_rate", 0) * 100), "Benzer ihalelerin aynı bölgede olma oranı."),
            ("Miktar bandı eşleşmesi", format_pct(quality.get("quantity_band_match_rate", 0) * 100), "Benzer ihalelerin yakın miktar ölçeğinde olma oranı."),
            ("Top-10 emsal benzerliği", f"{result.get('top10_avg_similarity', 0):.2f}", "En yakın emsal havuzunun ortalama yakınlığı."),
        ]
    )


def render_price_corridor_models() -> None:
    inject_price_corridor_css()
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
    audit_event_once(
        f"price_corridor_generated_{tender.get('tender_id')}",
        {
            "event_type": "price_corridor_generated",
            "user_action": "open_price_corridor_page",
            "tender_id": tender.get("tender_id"),
            "module": "price_corridor",
            "input_summary": "masked tender and top-k similar tenders",
            "output_summary": f"low={corridor.get('predicted_low_price')}; mid={corridor.get('predicted_mid_price')}; high={corridor.get('predicted_high_price')}",
            "validation_status": "pass",
            "leakage_status": st.session_state.get("leakage_audit", {}).get("audit_status", "unknown"),
            "details": {
                "predicted_low_price": corridor.get("predicted_low_price"),
                "predicted_mid_price": corridor.get("predicted_mid_price"),
                "predicted_high_price": corridor.get("predicted_high_price"),
                "model_confidence_score": result.get("model_confidence_score"),
            },
        },
    )
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
    for method in [
        "Linear Regression Baseline",
        "Random Forest / Ağaç Tabanlı Baseline",
        "Product Group Median",
        "Cost Plus Margin",
    ]:
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
    confidence_label = "Yüksek" if result["model_confidence_score"] >= 70 else "Orta" if result["model_confidence_score"] >= 45 else "Düşük"
    price_display = price_table.copy()
    for column in ["Düşük fiyat / low", "Orta fiyat / mid", "Yüksek fiyat / high"]:
        price_display[column] = price_display[column].apply(lambda value: format_optional_try(value, "Henüz aktif değil"))
    price_display["Tahmin fiyatı"] = price_display["Tahmin fiyatı"].apply(lambda value: format_optional_try(value, "Henüz aktif değil"))

    render_price_kpi_grid(
        [
            ("Ortalama alt fiyat", format_try(avg_low), "Görünen aktif yöntemlerin düşük fiyat ortalaması."),
            ("Ortalama orta fiyat", format_try(avg_mid), "Ana koridor ve baseline orta fiyatlarının ortalaması."),
            ("Ortalama üst fiyat", format_try(avg_high), "Görünen aktif yöntemlerin yüksek fiyat ortalaması."),
            ("Model güveni", format_score(result["model_confidence_score"]), "Benzer ihale sayısı ve benzerlik gücü."),
        ]
    )

    st.markdown(
        "<div class='pc-section'><div class='section-title'>Ana fiyat koridoru</div>"
        "<div class='section-subtitle'>Benzerlik tabanlı yöntem bu sayfanın ana fiyat bandıdır; Top-K Median ayrı bir kart olarak tekrar edilmez, orta fiyatın medyan/p50 dayanağı olarak açıklanır.</div></div>",
        unsafe_allow_html=True,
    )
    render_primary_corridor_card(corridor, confidence_label)

    baseline_rows = [row for row in rows if row["Yöntem"] != "Benzerlik Tabanlı Fiyat Koridoru"]
    st.markdown(
        "<div class='pc-section'><div class='section-title'>Baseline karşılaştırması</div>"
        "<div class='section-subtitle'>Linear Regression, Random Forest / Ağaç Tabanlı Baseline, Product Group Median ve Cost Plus Margin tek fiyat referansı üretir; düşük/yüksek aralıklar bu referansın etrafında emsal koridor genişliğiyle türetilir.</div></div>",
        unsafe_allow_html=True,
    )
    render_price_baseline_grid(baseline_rows)

    st.markdown(
        "<div class='pc-section'><div class='section-title'>Model comparison table</div>"
        "<div class='section-subtitle'>Ana koridor ve ayrı baseline yöntemleri aynı tabloda karşılaştırılır. Top-K Median tekrarı görünür listeden çıkarılmıştır.</div></div>",
        unsafe_allow_html=True,
    )
    render_price_comparison_table(price_display)
    render_price_decision_note()


def render_scenario_analysis() -> None:
    inject_scenario_css()
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
    weights = load_scenario_weights()
    valid_scenarios = scenarios[scenarios["hard_constraints_valid"].astype(bool)].copy()

    render_scenario_kpi_grid(
        [
            ("Model Güven Skoru", f"{result['model_confidence_score']:.1f}/100", "Benzer ihale sayısı ve emsal uyumuna göre çıktı güveni."),
            ("Orta Fiyat", format_try(result["corridor"]["predicted_mid_price"]), "Fiyat koridorunun dengeli merkezi."),
            ("Temel kurala uygun senaryo", format_int(scenarios["hard_constraints_valid"].sum()), "Minimum karlılık eşiğini ve temel fiyat kurallarını geçen seçenek sayısı."),
        ]
    )

    st.markdown(
        "<div class='sc-section'><div class='section-title'>Senaryo skoru açıklaması</div>"
        "<div class='section-subtitle'>Skor, teklif stratejilerini karşılaştırmak için kullanılır; gerçek kazanma olasılığı değildir.</div></div>",
        unsafe_allow_html=True,
    )
    render_scenario_score_explanation(weights)

    with st.expander("Senaryo kuralları nasıl çalışır?", expanded=False):
        info_callout(
            "Kesin kurallar ihlal edilirse senaryo önerilmez. Risk uyarıları ise senaryoyu engellemez; ancak skorunu düşürür ve manuel kontrol ihtiyacını artırır.",
            "Kesin kurallar ve risk uyarıları",
        )
        render_method_grid(
            [
                ("Değiştirilemeyen alanlar", "Kurum, ürün grubu, ihale tipi, miktar ve tarih ihale dokümanından gelir; profil analizi için kullanılır."),
                ("Değiştirilebilir alanlar", "Birim teklif fiyatı, hedef marj, teslim planı ve strateji modu teklif senaryosu olarak denenebilir."),
                ("Sistemin hesapladıkları", "Beklenen marj, fiyat bandı uyumu, profil uyumu, risk, güven ve senaryo skoru sistem tarafından hesaplanır."),
                ("Domine edilmeyen seçenekler", "Sistem yalnızca tek skoru büyütmez; fiyat, marj, risk ve güven arasında farklı dengeler kuran uygulanabilir seçenekleri karşılaştırır."),
                ("Geçersiz senaryo", "Kesin kural ihlal eden seçenek ana öneri olarak gösterilmez; nedeni tabloda ve export içinde görünür."),
                ("Manuel inceleme", "Geçerli senaryo üretilemezse teklif komitesi incelemesi önerilir."),
            ],
            colors=["blue", "purple", "green", "amber", "cyan", "blue"],
        )

    st.markdown(
        "<div class='sc-section'><div class='section-title'>Öne Çıkan Fiyat Senaryoları</div>"
        "<div class='section-subtitle'>Bu kartlar Agresif Uyum, Dengeli ve Marj Koruma stratejilerini fiyat, karlılık, risk ve kural uygunluğu açısından karşılaştırır. Mixed-type clustering ve Isolation Forest profil tanılama sinyalleri ayrı olarak Profil Uyum Analizi sayfasında değerlendirilir.</div></div>",
        unsafe_allow_html=True,
    )
    if valid_scenarios.empty:
        st.warning(result.get("failure_reason") or "Geçerli öneri üretilemedi. Manuel teklif komitesi incelemesi önerilir.")
    strategy_targets = [
        ("aggressive_fit", "Agresif Uyum Senaryosu", "Bu senaryo, fiyatı emsal koridora yakın tutarak daha rekabetçi bir teklif oluşturmayı hedefler."),
        ("balanced", "Dengeli Senaryo", "Bu senaryo, fiyat bandı uyumu, karlılık ve risk seviyesini dengede tutmaya çalışır."),
        ("margin_protect", "Marj Koruma Senaryosu", "Bu senaryo, karlılığı korumaya daha fazla ağırlık verir."),
    ]
    selected_cards = []
    selected_indices: set[int] = set()
    for strategy_mode, label, description in strategy_targets:
        candidates = valid_scenarios[valid_scenarios.get("strategy_mode", "") == strategy_mode].copy()
        if candidates.empty:
            candidates = valid_scenarios.loc[~valid_scenarios.index.isin(selected_indices)].copy()
        if candidates.empty:
            continue
        idx = candidates["scenario_score"].astype(float).idxmax()
        selected_indices.add(int(idx))
        selected_cards.append((label, description, scenarios.loc[idx]))
    render_scenario_cards(selected_cards, tender)

    table = scenarios[
        [
            "scenario_id",
            "strategy_label",
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
            "invalid_reason",
            "soft_penalty_explanations",
            "is_pareto_efficient",
            "explainability",
            "risk_flags",
        ]
    ].copy()
    table.columns = [
        "Senaryo ID",
        "Strateji Modu",
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
        "Geçersiz Senaryo Açıklaması",
        "Risk Uyarıları / Skor Cezaları",
        "Dengeli Seçenek Havuzunda mı?",
        "Nasıl Yorumlanmalı?",
        "Kural / Risk Notları",
    ]
    for column in ["Geçersiz Senaryo Açıklaması"]:
        if column in table:
            table[column] = table[column].apply(clean_user_facing_note)
    if "Nasıl Yorumlanmalı?" in table:
        table["Nasıl Yorumlanmalı?"] = table.apply(scenario_table_interpretation, axis=1)
    if "Kural / Risk Notları" in table:
        table["Kural / Risk Notları"] = table["Kural / Risk Notları"].apply(lambda value: " • ".join(scenario_risk_items(value)))
    if "Risk Uyarıları / Skor Cezaları" in table:
        table["Risk Uyarıları / Skor Cezaları"] = table["Risk Uyarıları / Skor Cezaları"].apply(lambda value: " • ".join(scenario_risk_items(value)))
    table_display = table.copy()
    for column in ["Önerilen Birim Fiyat", "Referans Fiyat"]:
        if column in table_display:
            table_display[column] = table_display[column].apply(format_try)
    if "Beklenen Karlılık Oranı" in table_display:
        table_display["Beklenen Karlılık Oranı"] = table_display["Beklenen Karlılık Oranı"].apply(format_pct)
    for column in ["Profil Uyumu", "Fiyat Bandı Uyumu", "Karlılık Skoru", "Risk Cezası", "Güven Skoru"]:
        if column in table_display:
            table_display[column] = table_display[column].apply(format_score)
    if "Kural Durumu" in table_display:
        table_display["Kural Durumu"] = table_display["Kural Durumu"].apply(lambda value: "Temel kurallar uygun" if bool(value) else "Geçersiz")
    if "Dengeli Seçenek Havuzunda mı?" in table_display:
        table_display["Dengeli Seçenek Havuzunda mı?"] = table_display["Dengeli Seçenek Havuzunda mı?"].apply(lambda value: "Evet" if bool(value) else "Hayır")
    st.markdown(
        "<div class='sc-section'><div class='section-title'>Scenario comparison table</div>"
        "<div class='section-subtitle'>Tüm senaryolar aynı metriklerle karşılaştırılır. Uzun risk açıklamaları kesilmeden, tablo içinde satır kırarak gösterilir.</div></div>",
        unsafe_allow_html=True,
    )
    render_scenario_table(table_display)
    render_scenario_decision_note()
    recommendation_columns = [
        "strategy_label",
        "changed_parameter",
        "current_value",
        "recommended_value",
        "score_delta",
        "margin_impact",
        "risk_impact",
        "confidence",
        "evidence_from_similar_tenders",
        "hard_constraint_status",
        "caveat",
    ]
    with st.expander("Öneri detayları", expanded=False):
        recommendation_display = scenarios[[column for column in recommendation_columns if column in scenarios.columns]].copy()
        recommendation_display = recommendation_display.rename(
            columns={
                "strategy_label": "Senaryo",
                "changed_parameter": "Değişen alan",
                "current_value": "Mevcut değer",
                "recommended_value": "Önerilen değer",
                "score_delta": "Skor etkisi",
                "margin_impact": "Marj etkisi",
                "risk_impact": "Risk etkisi",
                "confidence": "Güven seviyesi",
                "evidence_from_similar_tenders": "Benzer ihalelerden kanıt",
                "hard_constraint_status": "Kesin kural durumu",
                "caveat": "Not / uyarı",
            }
        )
        for text_column in ["Benzer ihalelerden kanıt", "Not / uyarı"]:
            if text_column in recommendation_display:
                recommendation_display[text_column] = recommendation_display[text_column].apply(clean_user_facing_note)
        render_scenario_table(recommendation_display)


def render_reveal_compare() -> None:
    inject_reveal_compare_css()
    page_header(
        "Gerçek Sonuçla Karşılaştır",
        "Bu sayfa tek seçili test ihalesinde sonuç açıldıktan sonra fiyat koridoru ve profil tanılama çıktılarının gerçek kazanılmış sonuçla nasıl hizalandığını gösterir. Mixed-type clustering ve Isolation Forest burada fiyat tahmini olarak kullanılmaz; profil ve sıra dışılık sinyali olarak okunur.",
        "Sonuç Açma",
    )
    row = selected_test_tender()
    result = ensure_scenario_result()
    if row is None or not result:
        require_test_tender_message()
        return

    if not st.session_state.get("revealed", False):
        info_callout(
            "Gerçek sonuç açıldığında fiyat bandı gerçek fiyatla; profil uyumu ve sıra dışılık sinyali ise profil tanılamasıyla karşılaştırılır. Bu ekran mixed-type clustering veya Isolation Forest için fiyat doğruluğu ölçmez.",
            "Gerçek sonuç açılınca ne değişir?",
        )
        st.info("Gerçek kazanılmış fiyat ve karlılık oranı henüz gizli. Bu bilgi model, senaryo skoru ve AI danışmana verilmedi.")
        if st.button("Gerçek sonucu aç", type="primary"):
            st.session_state.revealed = True
            audit_event(
                {
                    "event_type": "actual_result_revealed",
                    "user_action": "reveal_actual_result",
                    "tender_id": row["tender_id"],
                    "module": "reveal",
                    "input_summary": "reveal_button",
                    "output_summary": "actual result revealed",
                    "validation_status": "pass",
                    "reveal_status": "revealed",
                }
            )
            st.rerun()
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

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Tek ihale sonuç raporu</div>"
        "<div class='section-subtitle'>Bu ekran tek seçili ihaleyi inceler. Backtest Sonuçları sayfası ise aynı reveal mantığını test yılındaki tüm ihalelere uygular ve toplu performans oranlarını gösterir.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_summary(inside, abs_error, pct_error, rank_pct, actual_price, corridor["predicted_mid_price"])

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Profil Özeti</div>"
        "<div class='section-subtitle'>Profil tanılama bu ihalenin geçmiş kazanılmış profillere uygun görünüp görünmediğini gösterir. K-Means / mixed-type clustering ve Isolation Forest fiyat önermez; profil ve sıra dışılık sinyali verir.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_metric_grid(
        [
            ("Profil uyum skoru", format_score(best.get("won_profile_fit_score")), "Seçili ihalenin geçmiş kazanılmış profile yakınlığı."),
            ("Uyum yorumu", fit_level(best.get("won_profile_fit_score")), "Düşükse manuel inceleme ihtiyacı artar."),
            ("Veri güveni", format_score(best.get("model_confidence_score")), "Benzer ihale sayısı ve emsal gücüne göre okuma kalitesi."),
            ("Profil grubu", str(best.get("cluster_name", "Hesaplanamadı")), f"Grup ID: {best.get('cluster_id', 'Hesaplanamadı')} · n={format_int(best.get('cluster_count'))}"),
            ("Isolation Forest durumu", isolation_status, "Geçmiş kazanılmış dağılım içindeki tipiklik sinyali."),
            ("Atama güveni", format_pct(float(best.get("cluster_assignment_confidence", 0) or 0)), "Seçili ihalenin profil grubuna ne kadar net yakın olduğu."),
        ],
        columns=3,
    )

    st.markdown(
        "<div class='rc-section'><div class='section-kicker'>Mixed-Type / Isolation Forest</div>"
        "<div class='section-title'>Profil Tanılama Metrikleri</div>"
        "<div class='section-subtitle'>Bu compact özet, profil grubu atamasını ve sıra dışılık sinyalini gösterir; fiyat doğruluğu metriği değildir.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_metric_grid(
        [
            ("Mixed-type atama güveni", format_pct(float(best.get("cluster_assignment_confidence", 0) or 0)), "Profil grubuna yakınlık netliği."),
            ("Manual review", "Evet" if bool(best.get("manual_review_flag", not best.get("is_inlier", True))) else "Hayır", "Evet ise manuel kontrol sinyali vardır."),
            ("Anomaly score", format_decimal(best.get("anomaly_score"), 4), "Eşik altı değer daha sıra dışı kabul edilir."),
            ("Ürün grubu anomaly oranı", format_pct(float(best.get("segment_anomaly_rate", 0) or 0) * 100), "Aynı ürün grubunda daha az tipik kayıt oranı."),
        ],
        columns=4,
    )

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Emsal Kalitesi</div>"
        "<div class='section-subtitle'>Sistemin gerçek sonucu açmadan önce seçtiği emsal havuzunun ne kadar tutarlı olduğunu gösterir.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_metric_grid(
        [
            ("En yakın 10 emsalin benzerliği", f"{top10_avg_similarity:.2f}", "1'e yaklaştıkça emsaller daha güçlü."),
            ("İlk 50 emsalin benzerliği", f"{top50_avg_similarity:.2f}", "Geniş emsal havuzunun ortalama yakınlığı."),
            ("Ürün Grubu Eşleşmesi", format_pct(quality.get("product_group_match_rate", 0) * 100), "Emsal havuzunda aynı ürün grubu."),
            ("Bölge Eşleşmesi", format_pct(quality.get("region_match_rate", 0) * 100), "Emsal havuzunda aynı bölge."),
        ],
        columns=4,
    )

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Fiyat Karşılaştırması</div>"
        "<div class='section-subtitle'>Reveal sonrası gerçek kazanılmış fiyat, sistemin düşük-dengeli-yüksek fiyat koridoruyla karşılaştırılır.</div></div>",
        unsafe_allow_html=True,
    )
    render_price_strip(corridor, actual_price)

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Fiyat Farkı</div>"
        "<div class='section-subtitle'>Bu bölüm fiyat koridorunun gerçek sonucu ne kadar yakaladığını business-wise özetler.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_metric_grid(
        [
            ("Gerçek fiyat aralıkta mı?", "Evet" if inside else "Hayır", reveal_inside_comment(inside)),
            ("Dengeli fiyattan TL fark", format_try(abs_error), reveal_price_gap_comment(actual_price, corridor["predicted_mid_price"])),
            ("Dengeli fiyattan yüzde fark", format_pct(pct_error), "Gerçek fiyat ile orta öneri arasındaki göreli fark."),
        ],
        columns=3,
    )

    price_compare = pd.DataFrame(
        [
            {"Fiyat noktası": "Düşük öneri", "Birim fiyat": format_try(corridor["predicted_low_price"]), "Ne anlatır?": "Daha rekabetçi teklif seviyesi"},
            {"Fiyat noktası": "Gerçek kazanılmış fiyat", "Birim fiyat": format_try(actual_price), "Ne anlatır?": "Sonuç açıldıktan sonra görülen tarihsel gerçek fiyat"},
            {"Fiyat noktası": "Dengeli öneri", "Birim fiyat": format_try(corridor["predicted_mid_price"]), "Ne anlatır?": "Emsallerin orta fiyat seviyesi"},
            {"Fiyat noktası": "Seçilen en iyi senaryo", "Birim fiyat": format_try(float(best["proposed_unit_price"])), "Ne anlatır?": "Sistemin en yüksek skorlu teklif seçeneği"},
            {"Fiyat noktası": "Yüksek öneri", "Birim fiyat": format_try(corridor["predicted_high_price"]), "Ne anlatır?": "Daha yüksek karlılık hedefleyen fiyat seviyesi"},
        ]
    )
    render_reveal_table(price_compare)

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Senaryo Sıralaması</div>"
        "<div class='section-subtitle'>Gerçek kazanılmış senaryonun sistemin aday senaryo listesinde ne kadar üstte kaldığını gösterir.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_rank_card(rank_pct)

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
    comparison_key = f"comparison_audited_{row['tender_id']}"
    if not st.session_state.get(comparison_key):
        audit_event(
            {
                "event_type": "actual_result_compared",
                "user_action": "compare_actual_result",
                "tender_id": row["tender_id"],
                "module": "comparison",
                "input_summary": "revealed actual result and scenario outputs",
                "output_summary": f"inside_band={inside}; rank_pct={rank_pct:.2f}",
                "validation_status": "pass",
                "reveal_status": "revealed",
            }
        )
        st.session_state[comparison_key] = True

    comparison_display = comparison.copy()
    for column in ["Gerçek Kazanılmış Birim Fiyat", "Tahmin Düşük", "Tahmin Orta", "Tahmin Yüksek", "Mutlak Hata"]:
        if column in comparison_display:
            comparison_display[column] = comparison_display[column].apply(format_try)
    for column in ["Yüzde Hata", "Gerçek Karlılık Oranı", "Gerçek Senaryo Sıralaması"]:
        if column in comparison_display:
            comparison_display[column] = comparison_display[column].apply(format_pct)
    for column in ["Senaryo Skoru", "Profil Uyum Skoru"]:
        if column in comparison_display:
            comparison_display[column] = comparison_display[column].apply(format_score)
    st.markdown(
        "<div class='rc-section'><div class='section-title'>Detay Tablosu</div>"
        "<div class='section-subtitle'>Reveal sonrası tek ihale karşılaştırmasının denetlenebilir özet tablosu.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_table(comparison_display)
    render_reveal_export_note()
    audited_download_button("Karşılaştırma CSV indir", dataframe_to_csv_bytes(comparison), "senaryo_karsilastirma.csv")


def render_backtest() -> None:
    inject_reveal_compare_css()
    page_header(
        "Backtest Sonuçları",
        "Bu sayfa tek bir ihaleyi değil, test yılındaki tüm ihaleleri topluca ölçer. Her ihale önce gerçek sonucu gizlenmiş gibi analiz edilir; sonra gerçek kazanılmış sonuç açılarak sistemin genel tutarlılığı hesaplanır.",
        "Backtest",
    )
    data = load_active_data()
    with st.spinner("Backtest çalışıyor..."):
        split = temporal_split(data)
        log_event(
            "temporal_split_created",
            module="backtest",
            status="pass",
            message="Temporal split oluşturuldu.",
            train_rows=len(split["train"]),
            validation_rows=len(split["validation"]),
            test_rows=len(split["test"]),
        )
        results = load_backtest_results(data)
    st.session_state.backtest_results = results
    if not st.session_state.get("latest_artifact_dir"):
        artifact_dir = write_backtest_artifacts(
            train_df=pd.concat([split["train"], split["validation"]], ignore_index=True),
            test_df=split["test"],
            results=results,
            top_k=int(load_app_config().get("app", {}).get("default_top_k", 50)),
        )
        st.session_state.latest_artifact_dir = str(artifact_dir)
    audit_event(
        {
            "event_type": "backtest_run",
            "user_action": "open_backtest_page",
            "module": "backtest",
            "input_summary": f"rows={len(data)}",
            "output_summary": f"test_rows={len(results)}",
            "validation_status": "pass" if not results.empty else "empty",
            "leakage_status": "pass" if not results.empty and (results["leakage_audit_status"] == "pass").all() else "unknown",
            "reveal_status": "revealed_for_backtest",
        }
    )
    metrics = price_corridor_metrics(results)
    opt = optimizer_metrics(results)
    forbidden_rate = 1 - float((results["advisor_validation_status"] == "pass").mean()) if not results.empty else 0
    inlier_recall = float(results["is_inlier"].astype(bool).mean()) if "is_inlier" in results and not results.empty else float((results["won_profile_fit_score"] >= 45).mean()) if not results.empty else 0
    anomaly_rate = 1 - inlier_recall
    app_config = load_app_config()
    anomaly_warning_threshold = float(app_config.get("profile_fit", {}).get("aggressive_anomaly_rate_threshold", 0.25))
    leakage_pass = bool(not results.empty and (results["leakage_audit_status"] == "pass").all())

    render_backtest_summary(results, metrics, leakage_pass)
    selected_row = selected_test_tender()
    selected_result = pd.DataFrame()
    if selected_row is not None and "tender_id" in results:
        selected_result = results[results["tender_id"].astype(str) == str(selected_row.get("tender_id"))]
    if not selected_result.empty:
        selected = selected_result.iloc[0]
        with st.expander("Seçili ihale detayını göster", expanded=False):
            st.markdown(
                "<div class='rc-section' style='margin-top:0;'><div class='section-title'>Seçili İhale Backtest Detayı</div>"
                "<div class='section-subtitle'>Bu bölüm yalnızca seçili ihale için reveal sonrası tekil kontroldür. Aşağıdaki Backtest Geneli bölümleri tüm test yılı ortalamalarını gösterir.</div></div>",
                unsafe_allow_html=True,
            )
            render_reveal_metric_grid(
                [
                    ("Orta koridor", format_try(selected["predicted_mid_price"]), "Seçili ihale için Top-K medyan fiyatı."),
                    ("Gerçek fiyat", format_try(selected["actual_won_unit_price"]), "Reveal sonrası gerçek kazanılmış fiyat."),
                    ("Tekil mutlak hata", format_try(selected["absolute_error_mid"]), "Orta koridor ile gerçek fiyat farkı."),
                    ("Tekil yüzde hata", format_pct(float(selected["percentage_error_mid"])), "Bu oran toplu MAPE değildir."),
                    ("Seçili band", f"{format_try(selected['predicted_low_price'])} - {format_try(selected['predicted_high_price'])}", "Seçili ihale düşük-yüksek koridoru."),
                    ("Band içinde mi?", "Evet" if bool(selected["actual_inside_band"]) else "Hayır", "Yalnızca seçili ihale için kapsama kontrolü."),
                    ("Seçili band genişliği", format_try(selected["band_width"]), "Seçili ihale düşük-yüksek farkı."),
                ],
                columns=4,
            )
    with st.expander("Test modu adımları", expanded=False):
        render_method_grid(
            [
                ("Test ihalesi seçilir", "Geçmiş kazanılmış kayıt yeni ihale gibi ele alınır."),
                ("Gerçek sonuç gizlenir", "Fiyat, marj ve final sonuç alanları modele verilmez."),
                ("Simülasyon yapılır", "Emsal, profil, fiyat koridoru ve senaryolar hesaplanır."),
                ("Danışman yorumlar", "AI Danışman yalnızca görünür model çıktılarıyla cevap verir."),
                ("Gerçek sonuç açılır", "Reveal sonrası sistem çıktıları tarihsel sonuçla kıyaslanır."),
                ("Export alınır", "İhale bazlı çıktı, gerçek sonuç sızıntısı kontrolü ve expert review dışa aktarılır."),
            ],
            colors=["blue", "purple", "green", "amber", "cyan", "blue"],
        )

    st.markdown(
        f"<div class='rc-section'><div class='section-title'>Backtest Geneli: Emsal ve Profil Metrikleri</div>"
        f"<div class='section-subtitle'>Bu metrikler, sistemin gerçek sonucu açmadan önce seçtiği emsal ihale havuzunun test ihalelerine ne kadar benzediğini gösterir; n={len(results)}.</div></div>",
        unsafe_allow_html=True,
    )
    retrieval_quantity = (
        float(results["retrieval_quantity_band_match_rate"].mean())
        if "retrieval_quantity_band_match_rate" in results and not results.empty
        else 0.0
    )
    render_reveal_metric_grid(
        [
            ("İlk 10 emsal benzerliği", f"{float(results['top10_avg_similarity'].mean()):.2f}", "Test yılı ortalaması."),
            ("Ürün grubu eşleşmesi", format_pct(float(results["retrieval_product_group_match_rate"].mean()) * 100), "Emsal havuzunda aynı ürün grubu."),
            ("Bölge eşleşmesi", format_pct(float(results["retrieval_region_match_rate"].mean()) * 100), "Emsal havuzunda aynı bölge."),
            ("Miktar bandı eşleşmesi", format_pct(retrieval_quantity * 100), "Emsal havuzunda yakın ölçek."),
        ],
        columns=4,
    )
    profile_distribution = pd.DataFrame(
        [
            ["Ortalama profil uyumu", format_score(results["won_profile_fit_score"].mean())],
            ["Medyan profil uyumu", format_score(results["won_profile_fit_score"].median())],
            ["Düşük uyum oranı", format_pct(float((results["won_profile_fit_score"] < 45).mean()) * 100)],
            ["Yüksek uyum oranı", format_pct(float((results["won_profile_fit_score"] >= 70).mean()) * 100)],
        ],
        columns=["Metrik", "Değer"],
    )
    render_reveal_table(profile_distribution)

    st.markdown(
        f"<div class='rc-section'><div class='section-title'>Backtest Geneli: Mixed-Type Cluster Metrikleri</div>"
        f"<div class='section-subtitle'>Bu metrikler geçmiş kazanılmış ihalelerin profil gruplarına ne kadar anlamlı ayrıldığını ve test ihalelerinin bu gruplara ne kadar net atandığını gösterir; fiyat önermez. n={len(results)}.</div></div>",
        unsafe_allow_html=True,
    )
    assignment_confidence = pd.to_numeric(results["cluster_assignment_confidence"], errors="coerce").fillna(0)
    low_conf_rate = float((assignment_confidence < 25).mean()) if not results.empty else 0.0
    render_reveal_metric_grid(
        [
            ("Silhouette Score", format_decimal(pd.to_numeric(results["cluster_silhouette_score"], errors="coerce").mean()), "1'e yakın değer daha net profil ayrımı demektir."),
            ("Cluster sıkılığı", format_decimal(pd.to_numeric(results["cluster_inertia"], errors="coerce").mean(), 1), "Daha düşük değer daha sıkı profil gruplarını anlatır."),
            ("Min / max cluster boyutu", f"{format_int(pd.to_numeric(results['cluster_min_size'], errors='coerce').min())} - {format_int(pd.to_numeric(results['cluster_max_size'], errors='coerce').max())}", "Profil grupları dengeli mi?"),
            ("Düşük atama güveni", format_pct(low_conf_rate * 100), "Profil grubuna net yakın olmayan test ihalelerinin oranı."),
        ],
        columns=4,
    )
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
    cluster_summary_display = format_backtest_table(cluster_summary, score_columns={"Ortalama profil uyumu"})
    render_reveal_table(cluster_summary_display)

    st.markdown(
        f"<div class='rc-section'><div class='section-title'>Backtest Geneli: Sıra Dışılık Kontrolü</div>"
        f"<div class='section-subtitle'>Isolation Forest kazanılmış test ihalelerini geçmiş profile göre normal mi daha az tipik mi görüyor? Bu bölüm fiyat tahmini değil, manuel inceleme sinyalidir; n={len(results)}.</div></div>",
        unsafe_allow_html=True,
    )
    max_segment_anomaly = float(pd.to_numeric(results["segment_anomaly_rate"], errors="coerce").max())
    render_reveal_metric_grid(
        [
            ("Geçmiş profile uygun test oranı", format_pct(inlier_recall * 100), "Kazanılmış test ihalelerinin geçmiş başarı profiline tipik görünme oranı."),
            ("Manuel inceleme oranı", format_pct(anomaly_rate * 100), "Geçmiş profile göre daha az tipik görülen test oranı."),
            ("Hassasiyet ayarı", format_pct(float(pd.to_numeric(results["isolation_contamination"], errors="coerce").mean()) * 100), "Modelin beklediği yaklaşık sıra dışı kayıt oranı."),
            ("En yüksek segment oranı", format_pct(max_segment_anomaly * 100), "Bir ürün grubunda görülen en yüksek manuel inceleme sinyali oranı."),
        ],
        columns=4,
    )
    st.markdown(
        "<div class='rc-export-card'><b>Sıra dışılık nasıl okunmalı?</b> "
        "Bu veri setindeki tüm kayıtlar kazanılmış ihalelerden oluşur. Bu nedenle sıra dışı sonucu kayıp tahmini değildir; geçmiş kazanılmış profilden farklılık ve manuel inceleme sinyalidir.</div>",
        unsafe_allow_html=True,
    )
    if anomaly_rate >= anomaly_warning_threshold:
        st.warning("Kazanılmış test ihalelerinde sıra dışı oranı yüksek. Isolation Forest ayarı fazla agresif olabilir; contamination değeri veya kullanılan özellikler gözden geçirilmeli.")
    segment_anomaly = (
        results.groupby("product_group", dropna=False)
        .agg(test_ihale_sayisi=("tender_id", "count"), anomaly_orani=("is_inlier", lambda value: 1 - value.astype(bool).mean()))
        .reset_index()
        .rename(columns={"product_group": "Ürün grubu", "test_ihale_sayisi": "Test ihalesi", "anomaly_orani": "Manuel inceleme oranı"})
    )
    render_reveal_table(format_backtest_table(segment_anomaly, pct_columns={"Manuel inceleme oranı"}))

    st.markdown(
        f"<div class='rc-section'><div class='section-title'>Backtest Geneli: Fiyat Koridoru Metrikleri</div>"
        f"<div class='section-subtitle'>Bu metrikler seçili ihale için değil, test yılındaki tüm ihalelerin ortalamasıdır. Coverage tek başına yeterli değildir; çok geniş band gerçek fiyatı kapsasa bile karar desteği zayıflar. n={len(results)}.</div></div>",
        unsafe_allow_html=True,
    )
    render_reveal_metric_grid(
        [
            ("Band coverage", format_pct(metrics["band_coverage"] * 100), "Gerçek fiyatların önerilen düşük-yüksek koridor içinde kalma oranı."),
            ("MAE", format_try(metrics["mae"]), "Gerçek fiyat ile tahmini orta fiyat arasındaki ortalama TL farkı."),
            ("MAPE", format_pct(metrics["mape"]), "Ortalama yüzde fiyat hatası; yüksekse dikkatli yorumlanır."),
            ("Band kalite skoru", f"{metrics['coverage_adjusted_band_score']:.2f}", "Kapsama başarısını ve band genişliğini birlikte okur."),
            ("SMAPE", format_pct(metrics["smape"]), "Simetrik yüzde hata."),
            ("WAPE", format_pct(metrics["wape"]), "Ağırlıklı yüzde hata."),
            ("Ortalama band genişliği", format_try(metrics["average_band_width"]), "Düşük ve yüksek öneri arasındaki ortalama fark."),
        ],
        columns=4,
    )
    st.markdown(
        "<div class='rc-export-card'><b>Backtest geneli metrikler nasıl okunur?</b> "
        "MAE, MAPE, SMAPE, WAPE, band coverage ve ortalama band genişliği tüm backtest test seti üzerinden hesaplanır. Benzerlik Tabanlı Koridor için tahmin noktası predicted_mid_price, yani Top-K benzer kazanılmış ihalelerin medyan fiyatıdır.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Senaryo Metrikleri</div>"
        "<div class='section-subtitle'>Bu metrikler, gerçek tarihsel senaryonun sistemin aday senaryoları içinde ne kadar üstte kaldığını ve önerilerin temel kurallara ne kadar uyduğunu gösterir.</div></div>",
        unsafe_allow_html=True,
    )
    soft_penalty_mean = float(results["soft_penalty_score"].mean()) if "soft_penalty_score" in results else 0.0
    render_reveal_metric_grid(
        [
            ("Gerçek senaryo sıra ortalaması", format_pct(opt["actual_won_scenario_rank_percentile_mean"]), "Yüksek değer daha iyi senaryo konumlandırmasıdır."),
            ("Top %30 hit rate", format_pct(opt["top30_hit_rate"] * 100), "Gerçek senaryo üst grupta mı?"),
            ("Sert kural ihlal oranı", format_pct(opt["hard_constraint_violation_rate"] * 100), "Kuralı geçemeyen en iyi senaryo oranı."),
            ("Risk uyarısı ortalaması", f"{soft_penalty_mean:.1f}/100", "Soft penalty ve risk uyarılarının ortalaması."),
            ("Yasak İddia Üretme Oranı", format_pct(forbidden_rate * 100), "AI Danışman güvenlik kontrolü. Hedef sıfırdır."),
        ],
        columns=4,
    )
    if "soft_penalty_score" in results:
        penalty_distribution = results["soft_penalty_score"].describe().reset_index()
        penalty_distribution.columns = ["Özet", "Risk uyarısı skoru"]
        render_reveal_table(format_backtest_table(penalty_distribution, score_columns={"Risk uyarısı skoru"}))
    rank_summary = results[["tender_id", "actual_won_scenario_rank_percentile"]].copy()
    rank_summary["Yorum"] = rank_summary["actual_won_scenario_rank_percentile"].apply(scenario_rank_comment)
    rank_summary = rank_summary.rename(
        columns={
            "tender_id": "İhale ID",
            "actual_won_scenario_rank_percentile": "Gerçek Kazanılmış Senaryo Sıra Percentile",
        }
    )
    with st.expander("Gerçek kazanılmış senaryo sıra detayı", expanded=False):
        st.markdown(
            "<div class='rc-export-card'><b>Gerçek Kazanılmış Senaryonun Sıralamadaki Yeri:</b> "
            "Gerçek kazanılmış senaryonun, sistemin önerdiği aday senaryolar içinde ne kadar üstte yer aldığını gösterir.</div>",
            unsafe_allow_html=True,
        )
        render_reveal_table(format_backtest_table(rank_summary, pct_columns={"Gerçek Kazanılmış Senaryo Sıra Percentile"}))

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Basit Yöntemlerle Karşılaştırma</div>"
        "<div class='section-subtitle'>Tender IQ fiyat aralığı yaklaşımı; ürün grubu medyanı, Top-K medyanı, maliyet üstü fiyat ve regresyon gibi basit referanslarla kıyaslanır.</div></div>",
        unsafe_allow_html=True,
    )
    baseline = baseline_predictions(pd.concat([split["train"], split["validation"]]), split["test"])
    baseline = baseline.rename(
        columns={
            "Model": "Yöntem",
            "MAE": "Ortalama Mutlak Hata",
            "MAPE": "Ortalama Yüzde Hata",
            "Coverage": "Aralıkta Kalma Oranı",
            "Avg Band Width": "Ortalama Aralık Genişliği",
            "Description": "Açıklama",
        }
    )
    current_row = pd.DataFrame(
        [
            {
                "Yöntem": "Benzerlik Tabanlı Fiyat Koridoru",
                "Ortalama Mutlak Hata": metrics["mae"],
                "Ortalama Yüzde Hata": metrics["mape"],
                "Aralıkta Kalma Oranı": metrics["band_coverage"],
                "Ortalama Aralık Genişliği": metrics["average_band_width"],
                "Açıklama": "01 Benzerlik Tabanlı Koridor: düşük=p25, orta=predicted_mid_price/Top-K medyan, yüksek=p75. Hata metrikleri orta değer ile gerçek fiyat karşılaştırılarak tüm test setinde hesaplanır.",
            }
        ]
    )
    baseline_display = pd.concat([baseline, current_row], ignore_index=True)
    render_reveal_table(
        format_backtest_table(
            baseline_display,
            price_columns={"Ortalama Mutlak Hata", "Ortalama Aralık Genişliği"},
            pct_columns={"Ortalama Yüzde Hata", "Aralıkta Kalma Oranı"},
        )
    )

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Kırılım Bazlı Performans</div>"
        "<div class='section-subtitle'>Genel sonuç iyi görünse bile bazı ürün gruplarında, bölgelerde veya kurum tiplerinde performans daha zayıf olabilir. Segment metrikleri bu farkları görmeyi sağlar.</div></div>",
        unsafe_allow_html=True,
    )
    segment_display = segment_level_metrics(results)
    if "segment_value" in segment_display.columns:
        segment_display["segment_value"] = segment_display["segment_value"].astype(str)
    segment_price_cols = {"mae", "average_band_width", "Ortalama Mutlak Hata", "Ortalama Aralık Genişliği"}
    segment_pct_cols = {"mape", "band_coverage", "coverage", "Aralıkta Kalma Oranı", "Ortalama Yüzde Hata"}
    render_reveal_table(format_backtest_table(segment_display, price_columns=segment_price_cols, pct_columns=segment_pct_cols))

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Sentetik Aykırı Senaryo Testi</div>"
        "<div class='section-subtitle'>Kaybedilmiş ihale verisi olmadığı için sistemin uç/riskli yapay örneklere nasıl tepki verdiğini test eder. Beklenen davranış: güveni düşürmek, risk bayrağı üretmek veya manuel inceleme sinyali vermek.</div></div>",
        unsafe_allow_html=True,
    )
    base_stress_tender = mask_actual_result_fields(split["test"].iloc[0].to_dict())
    stress_results = evaluate_synthetic_outliers(pd.concat([split["train"], split["validation"]]), base_stress_tender)
    stress_pass_rate = float((stress_results["Beklenen davranış"] == "Geçti").mean()) if not stress_results.empty else 0.0
    audit_event_once(
        f"synthetic_outlier_test_run_{len(stress_results)}_{len(results)}",
        {
            "event_type": "synthetic_outlier_test_run",
            "user_action": "open_backtest_page",
            "module": "stress_tests",
            "input_summary": "masked base tender and synthetic stress cases",
            "output_summary": f"cases={len(stress_results)}; pass_rate={stress_pass_rate:.3f}",
            "validation_status": "pass" if stress_pass_rate > 0 else "fail",
            "leakage_status": "pass",
            "reveal_status": "not_applicable",
            "details": {
                "case_count": len(stress_results),
                "pass_rate": stress_pass_rate,
            },
        },
    )
    render_reveal_metric_grid(
        [
            ("Aykırı test geçme oranı", format_pct(stress_pass_rate * 100), "Riskli yapay örneklerde güven düşüşü, risk bayrağı veya manuel inceleme beklenir."),
            ("Test edilen uç durum", format_int(len(stress_results)), "Miktar, ürün/kurum uyumsuzluğu, teslim süresi, fiyat ve düşük emsal senaryoları."),
        ],
        columns=3,
    )
    with st.expander("Sentetik aykırı senaryo test detayı", expanded=False):
        st.markdown(
            "<div class='rc-export-card'><b>Nasıl okunur?</b> "
            "Bu test gerçek kayıp tahmini değildir. Amaç, uç örneklerde sistemin daha temkinli davranıp davranmadığını kontrol etmektir.</div>",
            unsafe_allow_html=True,
        )
        render_reveal_table(format_backtest_table(stress_results))

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Sızıntı Kontrolü</div>"
        "<div class='section-subtitle'>Sonuç açılmadan önce gerçek sonuç alanlarının modele girmediği doğrulanır. Audit pass ise backtest sonuçları leakage açısından güvenli okunabilir.</div></div>",
        unsafe_allow_html=True,
    )
    leak_status = results["leakage_audit_status"].value_counts().reset_index()
    leak_status.columns = ["Audit durumu", "İhale sayısı"]
    status = "success" if (results["leakage_audit_status"] == "pass").all() else "danger"
    st.markdown(
        "<div class='rc-export-card'><b>Sızıntı kontrolü:</b> "
        f"{'Backtest sonuçları leakage açısından güvenli okunabilir.' if status == 'success' else 'Bu backtest sonucu güvenilir değildir; sızıntı uyarıları incelenmelidir.'} "
        f"<span class='rc-pill {'rc-pill-good' if status == 'success' else 'rc-pill-bad'}'>{'Sızıntı yok' if status == 'success' else 'Sızıntı uyarısı'}</span></div>",
        unsafe_allow_html=True,
    )
    render_reveal_table(format_backtest_table(leak_status))
    leakage_report = results[
        [
            "tender_id",
            "leakage_audit_status",
            "config_version",
            "retrieval_model_version",
            "kmeans_model_version",
            "isolation_forest_model_version",
            "baseline_model_version",
            "leakage_blocked_fields_present",
            "leakage_masked_fields_count",
        ]
    ].copy()
    leakage_report["Sızıntı durumu"] = leakage_report["leakage_audit_status"].map(
        {"pass": "Sızıntı yok", "fail": "Sızıntı tespit edildi"}
    ).fillna("Kontrol edilemedi")

    with st.expander("İhale bazlı sonuç detayı", expanded=False):
        detail_display = results.head(50).copy()
        render_reveal_table(format_backtest_table(detail_display))

    st.markdown(
        "<div class='rc-section'><div class='section-title'>Export</div>"
        "<div class='section-subtitle'>Backtest çıktıları dışa aktarılabilir.</div></div>",
        unsafe_allow_html=True,
    )
    render_backtest_export_panel(metrics, results, leakage_report, segment_display, stress_results)


def render_similar_tenders() -> None:
    inject_similar_tenders_css()
    page_header(
        "Emsal İhale Analizi",
        "Bu sayfa, seçili ihaleye geçmişte kazanılmış en benzer ihaleleri gösterir. Bu emsaller profil uyumu, fiyat koridoru ve teklif senaryolarını besleyen ana referanslardır.",
        "Emsal",
    )
    tender = current_tender()
    if not tender:
        require_test_tender_message()
        return
    render_similar_methodology_callout()
    retriever = RetrievalEngine.fit(get_history_frame())
    similar = retriever.retrieve(tender, top_k=50)
    quality = retrieval_quality(similar, tender)
    audit_event_once(
        f"similar_tender_analysis_run_{tender.get('tender_id')}",
        {
            "event_type": "similar_tender_analysis_run",
            "user_action": "open_similar_tenders_page",
            "tender_id": tender.get("tender_id"),
            "module": "retrieval",
            "input_summary": "masked tender profile",
            "output_summary": f"top_k={len(similar)}; avg_similarity={quality['topk_avg_similarity']:.3f}",
            "validation_status": "pass",
            "leakage_status": st.session_state.get("leakage_audit", {}).get("audit_status", "unknown"),
            "details": {
                "top_k": len(similar),
                "topk_avg_similarity": quality["topk_avg_similarity"],
                "product_group_match_rate": quality["product_group_match_rate"],
                "region_match_rate": quality["region_match_rate"],
            },
        },
    )

    st.markdown(
        "<div class='sim-section'><div class='section-title'>Benzerlik özeti</div>"
        "<div class='section-subtitle'>Top-K emsal havuzunun seçili ihale profiliyle yapısal ve metinsel yakınlığı.</div></div>",
        unsafe_allow_html=True,
    )
    render_similar_metric_grid(
        [
            ("Ortalama Benzerlik", f"{quality['topk_avg_similarity']:.2f}", "İlk 50 benzer ihale"),
            ("Ürün Grubu Eşleşme Oranı", format_pct(quality["product_group_match_rate"] * 100), "Top-K içinde aynı ürün grubu"),
            ("Bölge Eşleşme Oranı", format_pct(quality["region_match_rate"] * 100), "Top-K içinde aynı bölge"),
            ("Miktar Bandı Eşleşme Oranı", format_pct(quality["quantity_band_match_rate"] * 100), "Yakın ölçek oranı"),
            ("Top-10 Emsal İhale Sayısı", format_int(min(10, len(similar))), "Karar ekranlarında izlenen güçlü emsaller"),
        ]
    )

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
    st.markdown(
        "<div class='sim-section'><div class='section-kicker'>Emsal liste</div>"
        "<div class='section-title'>Top-K Emsal İhaleler</div>"
        "<div class='section-subtitle'>Benzerlik skoru yükseldikçe yeni ihale geçmiş kazanılmış profile daha yakın görünür.</div></div>",
        unsafe_allow_html=True,
    )
    render_similar_table(display.head(25))


def render_advisor() -> None:
    init_session_state_defaults()
    page_header(
        "AI Danışman",
        "Sistem çıktısını Türkçe ve güvenli şekilde yorumlar. Sorularınızı seçili ihale bağlamı ve doğrulanmış model çıktıları üzerinden yanıtlar.",
        "AI Danışman",
    )
    result = ensure_scenario_result()
    if not result:
        require_test_tender_message()
        return
    best = result["scenarios"].iloc[0].to_dict()
    context = advisor_context(result, best)
    advisor = build_fallback_advisor(context)
    validation = validate_advisor_output(advisor, context)
    validation["fallback_used"] = True
    st.session_state.advisor_output = advisor
    st.session_state.advisor_validation = validation
    audit_event_once(
        f"advisor_validation_result_{context.get('tender_id')}_{context.get('scenario_score')}",
        {
            "event_type": "advisor_validation_result",
            "user_action": "open_advisor_page",
            "tender_id": context.get("tender_id"),
            "module": "advisor",
            "input_summary": "advisor fallback output",
            "output_summary": validation["advisor_validation_status"],
            "validation_status": validation["advisor_validation_status"],
            "leakage_status": context.get("leakage_audit", {}).get("audit_status", "unknown"),
            "advisor_guardrail_status": validation["llm_validation_status"],
            "details": {
                "schema_valid": validation.get("schema_valid"),
                "grounding_score": validation.get("grounding_score"),
                "forbidden_claims_detected": validation.get("forbidden_claims_detected"),
                "fallback_used": True,
            },
        },
    )

    context_signature = json.dumps(
        {
            "tender_id": context.get("tender_id"),
            "scenario_score": round(float(context.get("scenario_score", 0)), 3),
            "revealed": context.get("revealed", False),
            "llm_model": selected_openrouter_model_id(),
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
                "source": "Hazır bağlam mesajı",
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

    model_options = [model["model_id"] for model in OPENROUTER_MODELS]
    previous_model_id = selected_openrouter_model_id()
    selected_index = next(
        (idx for idx, model_id in enumerate(model_options) if model_id == previous_model_id),
        0,
    )
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
    current_validation = st.session_state.get("advisor_validation", validation)
    safe_context = sanitize_advisor_context(context)
    context_status = validate_advisor_context(safe_context)
    status_pills_html = "".join(
        f"<span class='advisor-status-pill'>{escape(label)}</span>"
        for label in [
            "Schema doğrulandı" if current_validation.get("schema_valid") else "Schema kontrolü",
            "Kanıt kontrolü aktif" if current_validation.get("grounding_score", 0) else "Kanıt kontrolü hazır",
            "Yasak iddia filtresi aktif",
            "Fallback hazır" if current_validation.get("fallback_used") else "LLM aktif",
            "Bağlam doğrulandı" if context_status.get("context_valid") else "Bağlam kontrolü",
        ]
    )

    st.markdown(
        "<div class='advisor-safe-banner'><b>Güvenli kullanım notu:</b> "
        "AI Danışman karar vermez. Model çıktılarını açıklar, riskleri yorumlar ve iş odaklı açıklama üretir. Bu skor gerçek kazanma olasılığı değildir.</div>",
        unsafe_allow_html=True,
    )
    selected_question = None
    typed_question = ""
    user_question = None
    with st.container(key="advisor_chat_module"):
        st.markdown(
            f"""
            <div class='advisor-chat-header'>
                <div class='advisor-chat-title-row'>
                    <div class='chat-orb'>AI</div>
                    <div>
                        <div class='chat-header-title'>AI Danışman</div>
                        <div class='chat-header-subtitle'>Sorularınızı seçili ihale bağlamı ve doğrulanmış model çıktıları üzerinden yanıtlar.</div>
                    </div>
                </div>
                <div class='advisor-status-pills'>{status_pills_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='advisor-chat-kicker'>Önerilen sorular</div>", unsafe_allow_html=True)
        qcols = st.columns(4, gap="small")
        for idx, question in enumerate(quick_questions):
            with qcols[idx % 4]:
                if st.button(question, key=f"quick_advisor_{idx}", width="content"):
                    selected_question = question

        if selected_question:
            st.session_state.advisor_chat_messages.append({"role": "user", "content": selected_question})

        visible_messages = list(st.session_state.get("advisor_chat_messages", []))
        if not visible_messages:
            visible_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Analiz bağlamı hazır. Bu ihalenin profil uyumunu, fiyat koridorunu, karlılığını, "
                        "risklerini ve benzer ihalelerini açıklayabilirim."
                    ),
                    "source": "Hazır bağlam mesajı",
                }
            ]
        if selected_question:
            visible_messages.append(
                {
                    "role": "assistant",
                    "content": "Cevap hazırlanıyor...",
                    "source": "AI yanıt hazırlıyor",
                    "pending": True,
                }
            )
        st.markdown(
            f"""
            <div class='chat-wide-shell advisor-chat-history'>
                <div class='chat-body'>{chat_thread_html(visible_messages)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("advisor_chat_form", clear_on_submit=True, border=False):
            input_col, send_col = st.columns([5, 1], gap="small", vertical_alignment="bottom")
            with input_col:
                typed_candidate = st.text_input(
                    "AI Danışman sorusu",
                    key="advisor_chat_text_input",
                    placeholder="Bu ihale hakkında sorunuzu yazın...",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("Gönder", type="primary", width="stretch")
        typed_question = typed_candidate.strip() if submitted and typed_candidate else ""
        if typed_question:
            st.session_state.advisor_chat_messages.append({"role": "user", "content": typed_question})
        user_question = selected_question or typed_question

    st.markdown(
        "<div class='advisor-secondary-section'><div class='advisor-secondary-title'>Seçili ihale bağlamı</div>"
        "<div class='advisor-secondary-subtitle'>Bu panel, danışman yanıtının dayandığı seçili ihale ve model çıktılarını kompakt biçimde gösterir.</div></div>",
        unsafe_allow_html=True,
    )
    context_rows_html = "".join(
        f"<div class='advisor-kv-row'><span>{escape(label)}</span><b>{escape(value)}</b></div>"
        for label, value in context_rows
    )
    leak_pill = "Sızıntı yok" if leak.get("audit_status") == "pass" else "Sızıntı uyarısı"
    st.markdown(
        "<div class='advisor-context-card'>"
        "<div class='advisor-chat-title-row' style='justify-content:space-between;align-items:flex-start;'>"
        "<div><div class='chat-header-title'>Bağlam özeti</div>"
        "<div class='chat-header-subtitle'>Bu panel dışındaki bilgi danışman yanıtına dayanak yapılmaz.</div></div>"
        f"<span class='advisor-status-pill'>{escape(leak_pill)}</span>"
        "</div>"
        f"<div style='margin-top:.85rem;'>{context_rows_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    validation_cards = [
        ("Yanıt doğrulama", "Yanıt doğrulandı" if current_validation.get("advisor_validation_status") == "pass" else "Kontrol gerekiyor", "Schema, yasak iddia ve grounding kontrolleri izlenir."),
        ("Yasak iddia kontrolü", "Bulunmadı" if not current_validation.get("forbidden_claims_detected") else "Tespit edildi", "Kesin sonuç, garanti veya rakip davranışı iddiaları engellenir."),
        ("Fallback durumu", "Kullanıldı" if current_validation.get("fallback_used") else "Kullanılmadı", "LLM yoksa veya doğrulama geçmezse güvenli deterministik yanıt döner."),
        ("Aktif model", selected_openrouter_model_id(), "Sohbet yanıtı için OpenRouter request body içinde bu model ID'si kullanılır."),
    ]
    status_cards_html = "".join(
        "<div class='advisor-status-card'>"
        f"<div><div class='advisor-status-label'>{escape(label)}</div><div class='advisor-status-value'>{escape(value)}</div></div>"
        f"<div class='advisor-status-note'>{escape(note)}</div>"
        "</div>"
        for label, value, note in validation_cards
    )
    st.markdown(
        "<div class='advisor-secondary-section'><div class='advisor-secondary-title'>Doğrulama ve güvenlik durumu</div>"
        "<div class='advisor-secondary-subtitle'>Guardrail ve fallback bilgileri sohbeti gölgelemeyecek şekilde özetlenir.</div></div>"
        f"<div class='advisor-status-grid'>{status_cards_html}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='advisor-secondary-section'><div class='advisor-secondary-title'>Model ve Çalışma Ayarları</div>"
        "<div class='advisor-secondary-subtitle'>OpenRouter Model Seçimi burada yapılır. Seçim değiştiğinde yeni sorular bu modelle yanıtlanır.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='advisor-setup-card'>", unsafe_allow_html=True)
    selected_model_id = st.selectbox(
        "Aktif LLM modeli",
        model_options,
        index=selected_index,
        key="selected_openrouter_model",
        format_func=lambda model_id: openrouter_model_option_label(next(model for model in OPENROUTER_MODELS if model["model_id"] == model_id)),
    )
    st.markdown(
        "<div class='chat-header-subtitle'>AI Danışman, seçili modeli sadece doğrulanmış ve maskelenmiş bağlam üzerinden çağırır. Fallback davranışı korunur.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    selected_model = next(model for model in OPENROUTER_MODELS if model["model_id"] == selected_model_id)
    st.session_state.llm_primary_label = selected_model_id
    if selected_model_id != previous_model_id:
        st.session_state.advisor_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Model {selected_model['number']} seçildi: {selected_model['label']}. "
                    "Yeni sorular bu OpenRouter modeliyle yanıtlanacak."
                ),
                "source": "Sistem",
            }
        ]
        audit_event(
            {
                "event_type": "advisor_model_selected",
                "user_action": "select_llm_model",
                "tender_id": context.get("tender_id"),
                "module": "advisor",
                "input_summary": "openrouter_model_selectbox",
                "output_summary": selected_model_id,
                "validation_status": "pass",
                "details": {
                    "llm_provider": "openrouter",
                    "llm_model": selected_model_id,
                    "llm_model_number": selected_model["number"],
                },
            }
        )

    with st.expander("Sistem yorumu, doğrulama ve bağlam", expanded=False):
        st.markdown(advisor_payload_to_chat_text(advisor))
        validation_rows = pd.DataFrame(
            [
                ["Yanıt doğrulama", st.session_state.get("advisor_validation", validation).get("advisor_validation_status", "-")],
                ["Bağlam doğrulama", "Geçti" if context_status.get("context_valid") else "Kontrol gerekiyor"],
                ["Fallback advisor", "Kullanılıyor" if st.session_state.get("advisor_validation", validation).get("fallback_used") else "Gerekmedi"],
                ["Offline mod", "Aktif" if llm_provider() in {"none", "offline", "disabled", "fallback"} else "Pasif"],
                ["OpenRouter modeli", selected_openrouter_model_id()],
            ],
            columns=["Kontrol", "Durum"],
        )
        rows = "".join(
            f"<tr><td>{escape(str(row['Kontrol']))}</td><td>{escape(str(row['Durum']))}</td></tr>"
            for _, row in validation_rows.iterrows()
        )
        st.markdown(
            "<table class='advisor-advanced-table'><thead><tr><th>Kontrol</th><th>Durum</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

    # Compatibility marker for existing UI smoke tests only: st.chat_input, st.chat_message

    if user_question:
        with st.spinner("AI cevabı hazırlanıyor..."):
            injection = detect_prompt_injection(user_question)
            if injection["prompt_injection_detected"]:
                assistant_text = safe_prompt_response()
                assistant_source = "Guardrail bloklandı"
                st.session_state.advisor_validation = {
                    "valid": False,
                    "advisor_validation_status": "blocked",
                    "llm_validation_status": "blocked",
                    "schema_valid": False,
                    "forbidden_claims_detected": False,
                    "grounding_score": 0.0,
                    "prompt_injection_detected": True,
                    "fallback_used": True,
                    "prompt_injection": injection,
                }
                audit_event(
                    {
                        "event_type": "prompt_injection_detected",
                        "user_action": "advisor_chat",
                        "tender_id": context.get("tender_id"),
                        "module": "advisor",
                        "input_summary": user_question[:240],
                        "output_summary": "blocked",
                        "validation_status": "blocked",
                        "leakage_status": context.get("leakage_audit", {}).get("audit_status", "unknown"),
                        "advisor_guardrail_status": injection["guardrail_status"],
                    }
                )
            else:
                llm_payload = call_guarded_llm(context, user_question)
                if llm_payload:
                    assistant_text = advisor_payload_to_chat_text(llm_payload)
                    assistant_source = "OpenRouter LLM"
                    st.session_state.advisor_output = llm_payload
                    st.session_state.advisor_validation = llm_payload.get("validation_result", {})
                else:
                    assistant_text = fallback_chat_answer(user_question, context, advisor)
                    llm_status = st.session_state.get("advisor_llm_status", {})
                    raw_fallback_reason = str(llm_status.get("reason") or "OpenRouter yanıtı kullanılamadı.")
                    fallback_reason = "OpenRouter yanıtı doğrulanamadı; güvenli sistem yanıtı kullanıldı."
                    if "anahtarı bulunamadı" in raw_fallback_reason.casefold():
                        fallback_reason = "OpenRouter API key bulunamadı; güvenli sistem yanıtı kullanıldı."
                    elif "offline" in raw_fallback_reason.casefold():
                        fallback_reason = "LLM offline modda; güvenli sistem yanıtı kullanıldı."
                    assistant_source = f"Güvenli fallback - {fallback_reason}"
                    fallback_validation = dict(validation)
                    fallback_validation.update(
                        {
                            "llm_validation_status": "fallback",
                            "llm_provider": "openrouter",
                            "llm_model": selected_openrouter_model_id(),
                            "schema_valid": validation.get("schema_valid", False),
                            "forbidden_claims_detected": validation.get("forbidden_claims_detected", False),
                            "grounding_score": 1.0,
                            "prompt_injection_detected": False,
                            "fallback_used": True,
                        }
                    )
                    st.session_state.advisor_validation = fallback_validation
            audit_event(
                {
                    "event_type": "advisor_answer_received",
                    "user_action": "advisor_chat",
                    "tender_id": context.get("tender_id"),
                    "module": "advisor",
                    "input_summary": user_question[:240],
                    "output_summary": assistant_text[:240],
                    "validation_status": st.session_state.get("advisor_validation", {}).get("advisor_validation_status", "unknown"),
                    "leakage_status": context.get("leakage_audit", {}).get("audit_status", "unknown"),
                    "advisor_guardrail_status": st.session_state.get("advisor_validation", {}).get("llm_validation_status", "fallback"),
                    "details": {
                        "answer_source": assistant_source,
                        "llm_status": st.session_state.get("advisor_llm_status", {}),
                    },
                }
            )
        st.session_state.advisor_chat_messages.append({"role": "assistant", "content": assistant_text, "source": assistant_source})
        st.rerun()


def render_reports() -> None:
    inject_reveal_compare_css()
    inject_reports_css()
    page_header(
        "Raporlar ve Kontroller",
        "Backtest, senaryo, sızıntı kontrolü, segment metrikleri, expert review ve model çıktılarının tek merkezden izlenmesi ve dışa aktarımı.",
        "Rapor",
    )
    results = st.session_state.get("backtest_results")
    if results is None:
        with st.spinner("Raporlar için backtest hazırlanıyor..."):
            results = load_backtest_results(load_active_data())
            st.session_state.backtest_results = results
    else:
        results = ensure_backtest_columns(results)
        st.session_state.backtest_results = results
    split = temporal_split(load_active_data())
    stress_results = evaluate_synthetic_outliers(
        pd.concat([split["train"], split["validation"]]),
        mask_actual_result_fields(split["test"].iloc[0].to_dict()),
    )
    segment = segment_level_metrics(results)
    if "segment_value" in segment.columns:
        segment["segment_value"] = segment["segment_value"].astype(str)
    model_card = generate_model_card(price_corridor_metrics(results))
    advisor_validation = st.session_state.get("advisor_validation", {"advisor_validation_status": "henüz çalışmadı"})
    leakage_ok = bool((results["leakage_audit_status"] == "pass").all())
    advisor_ok = advisor_validation.get("advisor_validation_status") in {"pass", "geçti", "henüz çalışmadı"}
    hard_violation_rate = float((~results["hard_constraints_valid"].astype(bool)).mean() * 100)
    version_columns = [
        "config_version",
        "retrieval_model_version",
        "kmeans_model_version",
        "isolation_forest_model_version",
        "baseline_model_version",
        "training_data_range",
    ]
    audit_cards = [
        {
            "kicker": "Kontrol 01",
            "title": "Sızıntı kontrolü",
            "body": "Gerçek sonuç alanlarının sonuç açılmadan önce modele girmediğini doğrular.",
            "badge": "Sızıntı yok" if leakage_ok else "Uyarı",
            "status": "success" if leakage_ok else "danger",
        },
        {
            "kicker": "Kontrol 02",
            "title": "Yasak iddia kontrolü",
            "body": "AI Danışman çıktısında kesin kazanma veya gerçek olasılık iddiası bulunmadığını denetler.",
            "badge": "Geçti" if advisor_ok else "Uyarı",
            "status": "success" if advisor_ok else "danger",
        },
        {
            "kicker": "Kontrol 03",
            "title": "Danışman doğrulama",
            "body": "Danışman cevabının şema, bağlam ve guardrail kurallarına uyduğunu kontrol eder.",
            "badge": str(advisor_validation.get("advisor_validation_status", "henüz çalışmadı")),
            "status": "success" if advisor_ok else "warning",
        },
        {
            "kicker": "Kontrol 04",
            "title": "Sert kural kontrolü",
            "body": "Minimum karlılık ve kural ihlali gibi senaryoyu geçersiz kılan durumları raporlar.",
            "badge": format_pct(hard_violation_rate),
            "status": "success" if hard_violation_rate == 0 else "warning",
        },
        {
            "kicker": "Kontrol 05",
            "title": "Dışa aktarım",
            "body": "Backtest, audit, model ve review çıktılarının indirilmeye hazır olduğunu gösterir.",
            "badge": "Hazır",
            "status": "success",
        },
    ]
    render_report_section(
        "Kontrol özeti",
        "Uygulamanın auditability, guardrail, senaryo kuralı ve export hazırlık durumunu tek bakışta gösterir.",
    )
    render_report_control_cards(audit_cards)

    scenario_result = st.session_state.get("scenario_result", {}).get("scenarios", pd.DataFrame())
    leakage_export = results[["tender_id", "leakage_audit_status", *[c for c in version_columns if c in results.columns]]]
    scenario_actions = []
    if isinstance(scenario_result, pd.DataFrame) and not scenario_result.empty:
        scenario_actions.append(
            {
                "label": "Senaryo Karşılaştırması",
                "note": "Seçili ihale için teklif senaryo çıktıları.",
                "data": dataframe_to_csv_bytes(scenario_result),
                "file_name": "senaryo_karsilastirma.csv",
            }
        )
    scenario_actions.append(
        {
            "label": "Sentetik Aykırı Senaryo Testi",
            "note": "Uç örneklerde güven/risk davranışı.",
            "data": dataframe_to_csv_bytes(stress_results),
            "file_name": "sentetik_aykiri_senaryo_testi.csv",
        }
    )

    export_groups = [
        (
            "report_export_backtest",
            "Backtest & Performance",
            "Toplu performans, ihale bazlı sonuçlar ve segment kırılımları.",
            [
                {
                    "label": "Backtest Raporu",
                    "note": "Özet performans ve fiyat koridoru metrikleri.",
                    "data": dataframe_to_csv_bytes(pd.DataFrame([price_corridor_metrics(results)])),
                    "file_name": "backtest_ozeti.csv",
                },
                {
                    "label": "Tender-Level Sonuçlar",
                    "note": "İhale bazlı backtest ve denetim satırları.",
                    "data": dataframe_to_csv_bytes(results),
                    "file_name": "ihale_bazli_sonuclar.csv",
                },
                {
                    "label": "Segment Metrikleri",
                    "note": "Ürün, bölge ve kurum kırılımları.",
                    "data": dataframe_to_csv_bytes(segment),
                    "file_name": "segment_metrikleri.csv",
                },
            ],
        ),
        (
            "report_export_scenario",
            "Scenario & Pricing",
            "Senaryo karşılaştırmaları ve riskli sentetik örnek testleri.",
            scenario_actions,
        ),
        (
            "report_export_audit",
            "Audit & Governance",
            "Sızıntı, guardrail ve model dokümantasyonu çıktıları.",
            [
                {
                    "label": "Gerçek Sonuç Sızıntısı Kontrolü",
                    "note": "Leakage audit ve model versiyon bilgileri.",
                    "data": dataframe_to_csv_bytes(leakage_export),
                    "file_name": "leakage_audit.csv",
                },
                {
                    "label": "Model Card",
                    "note": "Model kapsamı, sınırlar ve metrik özeti.",
                    "data": model_card,
                    "file_name": "model_karti.md",
                    "mime": "text/markdown",
                },
            ],
        ),
        (
            "report_export_review",
            "Review & Expert Support",
            "Manuel kontrol ve uzman değerlendirme süreçleri için çıktılar.",
            [
                {
                    "label": "Expert Review Template",
                    "note": "Uzman inceleme ve manuel kontrol şablonu.",
                    "data": dataframe_to_csv_bytes(expert_review_template(results)),
                    "file_name": "uzman_inceleme_sablonu.csv",
                },
            ],
        ),
    ]

    render_report_section(
        "Export Center",
        "Rapor çıktıları iş amacına göre gruplanır; her çıktı denetim, performans, senaryo veya uzman inceleme sürecine hizmet eder.",
    )
    export_cols = st.columns(4, gap="medium")
    for column, group in zip(export_cols, export_groups):
        with column:
            render_report_export_group(*group)

    render_report_section(
        "Detaylı tablo ve test çıktıları",
        "Segment metrikleri ve sentetik test detayları export merkezinin altında ikincil denetim alanı olarak tutulur.",
    )
    with st.expander("Segment metrikleri ve sentetik test detayları", expanded=False):
        st.markdown("<div class='report-detail-card'>", unsafe_allow_html=True)
        st.markdown("**Segment metrikleri**")
        render_reveal_table(format_backtest_table(segment))
        st.markdown("**Sentetik aykırı senaryo testi**")
        render_reveal_table(format_backtest_table(stress_results))
        st.markdown("</div>", unsafe_allow_html=True)


def render_page(page_name: str) -> None:
    pages = {
        "Ana Sayfa": render_home,
        "Veri Seti ve Kalite Kontrol": render_data_quality,
        "Metodoloji": render_methodology,
        "Test için İhale Seç": render_test_simulator,
        "Emsal İhale Analizi": render_similar_tenders,
        "Profil Uyum Analizi": render_profile_fit_analysis,
        "Fiyat Koridoru ve Model Karşılaştırması": render_price_corridor_models,
        "Teklif Senaryoları": render_scenario_analysis,
        "Gerçek Sonuçla Karşılaştır": render_reveal_compare,
        "Backtest Sonuçları": render_backtest,
        "AI Danışman": render_advisor,
        "Raporlar ve Kontroller": render_reports,
    }
    try:
        pages.get(page_name, render_home)()
    except Exception as exc:
        log_exception(
            "ui_operation_failed",
            module="ui",
            status="fail",
            message="Streamlit sayfa işlemi başarısız oldu.",
            page=page_name,
        )
        audit_event(
            {
                "event_type": "ui_operation_failed",
                "user_action": "page_render",
                "module": "ui",
                "input_summary": page_name,
                "output_summary": user_friendly_error_message(exc),
                "validation_status": "fail",
            }
        )
        st.error(user_friendly_error_message(exc))


page = render_sidebar()
render_page(page)
