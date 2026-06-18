"""Tender decision helper - Streamlit MVP."""

from __future__ import annotations

import re
import json
import os
import unicodedata
import warnings
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "x_ilac_synthetic_tenders_2021_2025.csv"
PRIMARY_PRICE_FIELD = "inflation_adjusted_unit_price_2026_try"
REFERENCE_PRICE_FIELD = "winning_unit_price_try"
SIMILAR_TENDER_COUNT = 50
DISPLAY_TENDER_COUNT = 10
PRICE_MODEL_VERSION = "2026-price-model-x-company-clean-v2"
SUCCESS_PROFILE_VERSION = "success-profile-x-company-clean-v4-segment-clusters"
SUCCESS_PROFILE_COUNT = 4
MODEL_FEATURES = [
    "product_name",
    "product_group",
    "region",
    "procedure_type",
    "buyer_institution",
    "quantity",
    "delivery_months",
    "competitor_count_estimate",
]
CATEGORICAL_FEATURES = [
    "product_name",
    "product_group",
    "region",
    "procedure_type",
    "buyer_institution",
]
NUMERIC_FEATURES = ["quantity", "delivery_months", "competitor_count_estimate"]
PROFILE_PRICE_FIELD = "profile_unit_price_2026_try"
PROFILE_MARGIN_FIELD = "profile_margin_pct"
PROFILE_FEATURES = [*MODEL_FEATURES, PROFILE_PRICE_FIELD, PROFILE_MARGIN_FIELD]
PROFILE_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
PROFILE_NUMERIC_FEATURES = [
    "quantity",
    "delivery_months",
    "competitor_count_estimate",
    PROFILE_PRICE_FIELD,
    PROFILE_MARGIN_FIELD,
]
SUCCESS_CLUSTER_FEATURES = ["product_group", "quantity", PROFILE_PRICE_FIELD]
SUCCESS_CLUSTER_CATEGORICAL_FEATURES = ["product_group"]
SUCCESS_CLUSTER_NUMERIC_FEATURES = ["quantity", PROFILE_PRICE_FIELD]

HISTORICAL_TEXT_FIELDS = [
    "tender_title",
    "product_name",
    "product_group",
    "buyer_institution",
    "region",
    "procedure_type",
]
QUERY_TEXT_FIELDS = ["product_name", "product_group", "region", "procedure_type"]

SIMILARITY_WEIGHTS = {
    "text_embedding": 0.45,
    "product_group": 0.12,
    "product_name": 0.10,
    "region": 0.08,
    "procedure_type": 0.05,
    "quantity": 0.10,
    "delivery_months": 0.05,
    "competitor_count": 0.05,
}
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = {
    "Google: Gemma 4 31B IT (free)": "google/gemma-4-31b-it:free",
    "NVIDIA: Nemotron 3 Super 120B A12B (free)": "nvidia/nemotron-3-super-120b-a12b:free",
    "OpenRouter: Owl Alpha": "openrouter/owl-alpha",
}
DEFAULT_OPENROUTER_MODEL_LABELS = list(OPENROUTER_MODELS.keys())


st.set_page_config(
    page_title="İhale Karar Yardımcısı",
    page_icon="TI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --panel-2: #eef4fb;
            --border: rgba(15, 23, 42, 0.12);
            --text: #172033;
            --muted: #64748b;
            --blue: #2563eb;
            --cyan: #0891b2;
            --amber: #d97706;
            --green: #16a34a;
            --red: #dc2626;
        }

        .stApp {
            background:
                radial-gradient(circle at 80% -10%, rgba(37, 99, 235, 0.10), transparent 30%),
                linear-gradient(180deg, #f8fafc 0%, #eef4fb 100%);
            color: var(--text);
        }

        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1500px;
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        .brand-mark {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--blue), var(--cyan));
            color: white;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        .sidebar-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .sidebar-note {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.45;
            margin-top: 0.45rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .eyebrow {
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .page-title {
            color: var(--text);
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.02;
            font-weight: 790;
            margin: 0;
        }

        .page-subtitle {
            color: var(--muted);
            max-width: 840px;
            font-size: 0.98rem;
            margin-top: 0.7rem;
            line-height: 1.55;
        }

        .scope-pill {
            display: inline-flex;
            gap: 0.42rem;
            align-items: center;
            float: right;
            margin-top: 0.5rem;
            padding: 0.42rem 0.7rem;
            border: 1px solid rgba(22, 163, 74, 0.22);
            border-radius: 999px;
            color: #166534;
            background: rgba(22, 163, 74, 0.08);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .scope-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 14px rgba(34, 197, 94, 0.35);
        }

        .section-kicker {
            color: var(--blue);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 760;
            margin-bottom: 0.9rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            height: 100%;
        }

        [data-testid="column"] {
            align-self: stretch;
        }

        [data-testid="column"] > div {
            height: 100%;
        }

        [data-testid="column"] [data-testid="stVerticalBlock"] {
            height: 100%;
        }

        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label {
            color: #334155 !important;
            font-size: 0.78rem !important;
            font-weight: 650 !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div > div {
            background: #ffffff;
            color: var(--text);
            border-color: var(--border);
            border-radius: 8px;
        }

        [data-testid="stButton"] button {
            width: 100%;
            min-height: 42px;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, #2563eb, #0891b2);
            color: #ffffff;
            font-weight: 850;
            margin-top: 1.6rem;
        }

        [data-testid="stDownloadButton"] button {
            width: 100%;
            min-height: 42px;
            border-radius: 8px;
            border: 1px solid rgba(37, 99, 235, 0.22);
            background: rgba(37, 99, 235, 0.08);
            color: #1d4ed8;
            font-weight: 850;
        }

        .metric-card {
            min-height: 136px;
            height: 100%;
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 820;
            margin-top: 0.45rem;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.72rem;
            margin-top: 0.3rem;
        }

        .score-card {
            padding: 1.2rem;
            border: 1px solid rgba(22, 163, 74, 0.24);
            border-radius: 8px;
            background: linear-gradient(145deg, rgba(22, 163, 74, 0.09), rgba(37, 99, 235, 0.07));
            min-height: 220px;
        }

        .score-value {
            color: var(--text);
            font-size: 3.7rem;
            line-height: 0.95;
            font-weight: 850;
            letter-spacing: 0;
        }

        .score-label {
            color: #166534;
            font-size: 1.1rem;
            font-weight: 800;
            margin-top: 0.6rem;
        }

        .confidence {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.58rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.08);
            border: 1px solid rgba(37, 99, 235, 0.20);
            color: #1d4ed8;
            font-size: 0.72rem;
            font-weight: 800;
        }

        .method-badge {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            padding: 0.34rem 0.58rem;
            margin: 0.35rem 0 0.45rem 0;
            border-radius: 999px;
            border: 1px solid rgba(8, 145, 178, 0.24);
            background: rgba(8, 145, 178, 0.09);
            color: #155e75;
            font-size: 0.72rem;
            font-weight: 850;
        }

        .method-badge.green {
            border-color: rgba(22, 163, 74, 0.24);
            background: rgba(22, 163, 74, 0.09);
            color: #166534;
        }

        .profile-summary {
            min-height: 116px;
            margin: 1rem 0 0.45rem 0;
            padding: 0.95rem 1rem;
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.82);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.55rem;
        }

        .profile-status {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .profile-score {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 850;
            line-height: 1.05;
            overflow-wrap: anywhere;
        }

        .profile-note {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }

        .profile-card-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.9rem 0 1rem 0;
            align-items: stretch;
        }

        .profile-card-panel {
            min-height: 385px;
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.45);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .profile-card-panel .method-badge {
            margin-top: 0.35rem;
        }

        .profile-card-panel p {
            color: var(--muted);
            font-size: 0.83rem;
            line-height: 1.55;
            margin: 0.65rem 0 0 0;
        }

        .mini-summary {
            min-height: 116px;
            padding: 0.9rem 0.95rem;
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.82);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .mini-value {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 840;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .mini-note {
            color: var(--muted);
            font-size: 0.74rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }

        .explain-box {
            padding: 1rem;
            border: 1px solid rgba(37, 99, 235, 0.16);
            border-radius: 8px;
            background: #ffffff;
            color: #334155;
            line-height: 1.55;
            font-size: 0.92rem;
        }

        .scope-note {
            padding: 0.85rem 0.95rem;
            border: 1px solid rgba(217, 119, 6, 0.22);
            border-radius: 8px;
            background: rgba(217, 119, 6, 0.07);
            color: #7c2d12;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .chat-screen-header {
            padding: 0.95rem 1rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(37, 99, 235, 0.34);
            border-left: 5px solid var(--blue);
            border-radius: 8px;
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.14), rgba(8, 145, 178, 0.09));
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.60);
        }

        .chat-screen-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 0.22rem;
        }

        .chat-screen-meta {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
        }

        [data-testid="stChatMessage"] {
            padding: 0.72rem 0.8rem;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 8px;
            background: #ffffff;
            margin-bottom: 0.7rem;
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(37, 99, 235, 0.30);
            border-radius: 8px;
            background: #ffffff;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.chat-screen-header) {
            border: 1px solid rgba(37, 99, 235, 0.32);
            background:
                linear-gradient(180deg, rgba(239, 246, 255, 0.92), rgba(236, 254, 255, 0.44));
            box-shadow: 0 16px 38px rgba(37, 99, 235, 0.12);
        }

        @media (max-width: 900px) {
            .scope-pill { float: none; margin-top: 1rem; }
            .score-value { font-size: 3rem; }
            .profile-card-grid { grid-template-columns: 1fr; }
            .profile-card-panel { min-height: auto; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> set[str]:
    text = normalize_text(value)
    return set(re.findall(r"[\w.%]+", text, flags=re.UNICODE))


def combine_text(record: pd.Series | dict[str, Any], fields: list[str]) -> str:
    return " | ".join(normalize_text(record.get(field, "")) for field in fields)


def exact_match_score(left: Any, right: Any) -> float:
    return 1.0 if normalize_text(left) == normalize_text(right) else 0.0


def product_name_score(query_name: str, candidate_name: str) -> float:
    query_normalized = normalize_text(query_name)
    candidate_normalized = normalize_text(candidate_name)
    if not query_normalized or not candidate_normalized:
        return 0.0
    if query_normalized == candidate_normalized:
        return 1.0
    if query_normalized in candidate_normalized or candidate_normalized in query_normalized:
        return 1.0

    query_tokens = tokenize(query_name)
    candidate_tokens = tokenize(candidate_name)
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens))


def numeric_similarity_score(left: Any, right: Any) -> float:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return 0.0
    denominator = max(abs(left_value), abs(right_value), 1.0)
    return max(0.0, 1.0 - abs(left_value - right_value) / denominator)


def hybrid_similarity_score(
    text_embedding_score: float,
    product_group: float,
    product_name: float,
    region: float,
    procedure_type: float,
    quantity: float,
    delivery_months: float,
    competitor_count: float,
) -> float:
    return (
        SIMILARITY_WEIGHTS["text_embedding"] * text_embedding_score
        + SIMILARITY_WEIGHTS["product_group"] * product_group
        + SIMILARITY_WEIGHTS["product_name"] * product_name
        + SIMILARITY_WEIGHTS["region"] * region
        + SIMILARITY_WEIGHTS["procedure_type"] * procedure_type
        + SIMILARITY_WEIGHTS["quantity"] * quantity
        + SIMILARITY_WEIGHTS["delivery_months"] * delivery_months
        + SIMILARITY_WEIGHTS["competitor_count"] * competitor_count
    )


@st.cache_data
def load_dataset(data_mtime_ns: int) -> pd.DataFrame:
    _ = data_mtime_ns
    df = pd.read_csv(DATA_PATH)
    required = [
        *HISTORICAL_TEXT_FIELDS,
        "tender_id",
        "year",
        "quantity",
        "delivery_months",
        "competitor_count_estimate",
        "winning_unit_price_try",
        "cpi_factor_to_2026",
        "inflation_adjusted_unit_price_2025_try",
        "inflation_adjusted_unit_price_2026_try",
        "inflation_adjusted_contract_value_2026_try",
        "gross_margin_pct",
        "discount_to_estimated_cost_pct",
        "strategic_fit_score",
    ]
    missing = [field for field in required if field not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


@st.cache_data
def load_dataset_download_bytes(data_mtime_ns: int) -> bytes:
    _ = data_mtime_ns
    return DATA_PATH.read_bytes()


@st.cache_resource
def build_text_embedding_index(search_texts: tuple[str, ...]) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    sparse_matrix = vectorizer.fit_transform(search_texts)
    max_components = min(128, sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1)

    if max_components >= 2:
        reducer = TruncatedSVD(n_components=max_components, algorithm="arpack", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dense_matrix = reducer.fit_transform(sparse_matrix)
        embedding_matrix = normalize(np.nan_to_num(dense_matrix, nan=0.0, posinf=0.0, neginf=0.0))
        method = f"Yerel metin embedding'i: TF-IDF + {max_components} boyutlu SVD"
    else:
        reducer = None
        embedding_matrix = normalize(sparse_matrix)
        method = "TF-IDF fallback"

    return {
        "vectorizer": vectorizer,
        "reducer": reducer,
        "matrix": embedding_matrix,
        "method": method,
    }


def get_similarity_index(df: pd.DataFrame) -> dict[str, Any]:
    search_texts = tuple(df.apply(lambda row: combine_text(row, HISTORICAL_TEXT_FIELDS), axis=1))
    return build_text_embedding_index(search_texts)


def build_model_pipeline(model: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("number", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_linear_pipeline() -> Pipeline:
    return build_model_pipeline(LinearRegression())


def build_xgboost_pipeline() -> Pipeline:
    return build_model_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=1,
        )
    )


def build_profile_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                PROFILE_CATEGORICAL_FEATURES,
            ),
            ("number", StandardScaler(), PROFILE_NUMERIC_FEATURES),
        ]
    )


def build_success_cluster_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                SUCCESS_CLUSTER_CATEGORICAL_FEATURES,
            ),
            ("number", StandardScaler(), SUCCESS_CLUSTER_NUMERIC_FEATURES),
        ]
    )


def build_profile_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[MODEL_FEATURES].copy()
    frame[PROFILE_PRICE_FIELD] = df[PRIMARY_PRICE_FIELD].astype(float)
    frame[PROFILE_MARGIN_FIELD] = df["gross_margin_pct"].astype(float)
    return frame[PROFILE_FEATURES]


def build_profile_query_frame(
    query: dict[str, Any],
    proposed_price: float,
    expected_margin_pct: float,
) -> pd.DataFrame:
    row = {field: query[field] for field in MODEL_FEATURES}
    row[PROFILE_PRICE_FIELD] = proposed_price
    row[PROFILE_MARGIN_FIELD] = expected_margin_pct
    return pd.DataFrame([row])[PROFILE_FEATURES]


def build_success_cluster_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[["product_group", "quantity"]].copy()
    frame[PROFILE_PRICE_FIELD] = df[PRIMARY_PRICE_FIELD].astype(float)
    return frame[SUCCESS_CLUSTER_FEATURES]


def volume_label(value: float, low: float, high: float) -> str:
    if value >= high:
        return "yüksek hacimli"
    if value <= low:
        return "düşük hacimli"
    return "orta hacimli"


def margin_label(value: float) -> str:
    if value >= 25:
        return "yüksek kazançlı"
    if value >= 12:
        return "orta kazançlı"
    return "düşük kazançlı"


def price_level_label(value: float, low: float, high: float) -> str:
    if value >= high:
        return "yüksek fiyatlı"
    if value <= low:
        return "düşük fiyatlı"
    return "orta fiyatlı"


def profile_mode(series: pd.Series) -> str:
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else "Çeşitli"


def build_cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> dict[int, dict[str, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    quantity_low = float(df["quantity"].quantile(0.33))
    quantity_high = float(df["quantity"].quantile(0.67))
    price_low = float(df[PRIMARY_PRICE_FIELD].quantile(0.33))
    price_high = float(df[PRIMARY_PRICE_FIELD].quantile(0.67))

    working = df.copy()
    working["success_profile_cluster"] = labels
    for cluster_id, group in working.groupby("success_profile_cluster"):
        median_quantity = float(group["quantity"].median())
        median_price = float(group[PRIMARY_PRICE_FIELD].median())
        median_margin = float(group["gross_margin_pct"].median())
        top_group = profile_mode(group["product_group"])
        top_region = profile_mode(group["region"])
        top_procedure = profile_mode(group["procedure_type"])
        name = (
            f"{top_group} - "
            f"{volume_label(median_quantity, quantity_low, quantity_high)} / "
            f"{price_level_label(median_price, price_low, price_high)} / "
            f"{margin_label(median_margin)} profil"
        )
        profiles[int(cluster_id)] = {
            "name": name,
            "count": int(len(group)),
            "top_product_group": top_group,
            "top_region": top_region,
            "top_procedure": top_procedure,
            "median_quantity": median_quantity,
            "median_price": median_price,
            "median_margin": median_margin,
            "average_strategic_fit": float(group["strategic_fit_score"].mean()),
        }
    return profiles


@st.cache_resource
def train_success_profile_models(
    df: pd.DataFrame,
    profile_version: str,
    data_mtime_ns: int,
) -> dict[str, Any]:
    _ = profile_version
    _ = data_mtime_ns
    profile_frame = build_profile_training_frame(df)
    one_class_preprocessor = build_profile_preprocessor()
    one_class_encoded = one_class_preprocessor.fit_transform(profile_frame)

    one_class_model = IsolationForest(
        n_estimators=300,
        contamination=0.12,
        random_state=42,
    )
    one_class_model.fit(one_class_encoded)
    one_class_scores = one_class_model.decision_function(one_class_encoded)

    cluster_frame = build_success_cluster_training_frame(df)
    cluster_preprocessor = build_success_cluster_preprocessor()
    cluster_encoded = cluster_preprocessor.fit_transform(cluster_frame[SUCCESS_CLUSTER_FEATURES])

    cluster_model = KMeans(
        n_clusters=SUCCESS_PROFILE_COUNT,
        n_init=20,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cluster_labels = cluster_model.fit_predict(cluster_encoded)
        cluster_distances = cluster_model.transform(cluster_encoded)
    assigned_distances = cluster_distances[np.arange(len(cluster_labels)), cluster_labels]
    distance_by_cluster = {
        int(cluster_id): np.sort(assigned_distances[cluster_labels == cluster_id])
        for cluster_id in range(SUCCESS_PROFILE_COUNT)
    }

    return {
        "one_class_preprocessor": one_class_preprocessor,
        "one_class_model": one_class_model,
        "one_class_scores": np.sort(one_class_scores),
        "cluster_preprocessor": cluster_preprocessor,
        "cluster_model": cluster_model,
        "distance_by_cluster": distance_by_cluster,
        "cluster_profiles": build_cluster_profiles(df, cluster_labels),
    }


def residual_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    residuals = actual.to_numpy(dtype=float) - predicted.astype(float)
    return {
        "p25": float(np.quantile(residuals, 0.25)),
        "median": float(np.quantile(residuals, 0.50)),
        "p75": float(np.quantile(residuals, 0.75)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(mean_absolute_percentage_error(actual, predicted) * 100),
    }


@st.cache_resource
def train_price_models(
    df: pd.DataFrame,
    model_version: str,
    data_mtime_ns: int,
) -> dict[str, Any]:
    _ = model_version
    _ = data_mtime_ns
    x = df[MODEL_FEATURES].copy()
    y = df[PRIMARY_PRICE_FIELD].astype(float)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    linear_cv = build_linear_pipeline()
    xgboost_cv = build_xgboost_pipeline()
    linear_predictions = cross_val_predict(linear_cv, x, y, cv=folds)
    xgboost_predictions = cross_val_predict(xgboost_cv, x, y, cv=folds)

    linear_residuals = residual_metrics(y, linear_predictions)
    xgboost_residuals = residual_metrics(y, xgboost_predictions)
    linear_residuals["coverage"] = corridor_coverage(
        y,
        linear_residuals["p25"],
        linear_residuals["p75"],
        linear_predictions,
    )
    xgboost_residuals["coverage"] = corridor_coverage(
        y,
        xgboost_residuals["p25"],
        xgboost_residuals["p75"],
        xgboost_predictions,
    )

    linear_model = build_linear_pipeline()
    xgboost_model = build_xgboost_pipeline()
    linear_model.fit(x, y)
    xgboost_model.fit(x, y)

    return {
        "linear_model": linear_model,
        "xgboost_model": xgboost_model,
        "linear_residuals": linear_residuals,
        "xgboost_residuals": xgboost_residuals,
    }


def retrieve_similar_tenders(
    df: pd.DataFrame,
    query: dict[str, Any],
    top_k: int = SIMILAR_TENDER_COUNT,
) -> pd.DataFrame:
    similarity_index = get_similarity_index(df)
    query_text = combine_text(query, QUERY_TEXT_FIELDS)
    query_vector = similarity_index["vectorizer"].transform([query_text])
    if similarity_index["reducer"] is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            query_embedding = similarity_index["reducer"].transform(query_vector)
    else:
        query_embedding = query_vector
    query_embedding = normalize(np.nan_to_num(query_embedding, nan=0.0, posinf=0.0, neginf=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        text_embedding_scores = cosine_similarity(query_embedding, similarity_index["matrix"])[0]
    text_embedding_scores = np.nan_to_num(text_embedding_scores, nan=0.0, posinf=0.0, neginf=0.0)

    candidates = []
    for initial_rank, idx in enumerate(text_embedding_scores.argsort()[::-1], start=1):
        row = df.iloc[int(idx)]
        text_embedding_score = float(text_embedding_scores[idx])
        group_score = exact_match_score(query["product_group"], row["product_group"])
        name_score = product_name_score(query["product_name"], row["product_name"])
        region_score = exact_match_score(query["region"], row["region"])
        procedure_score = exact_match_score(query["procedure_type"], row["procedure_type"])
        quantity_score = numeric_similarity_score(query["quantity"], row["quantity"])
        delivery_score_value = numeric_similarity_score(query["delivery_months"], row["delivery_months"])
        competitor_score = numeric_similarity_score(
            query["competitor_count_estimate"],
            row["competitor_count_estimate"],
        )
        final_score = hybrid_similarity_score(
            text_embedding_score,
            group_score,
            name_score,
            region_score,
            procedure_score,
            quantity_score,
            delivery_score_value,
            competitor_score,
        )

        item = row.to_dict()
        item.update(
            {
                "initial_text_embedding_rank": initial_rank,
                "text_embedding_score": text_embedding_score,
                "text_similarity_method": similarity_index["method"],
                "product_group_score": group_score,
                "product_name_score": name_score,
                "region_score": region_score,
                "procedure_type_score": procedure_score,
                "quantity_score": quantity_score,
                "delivery_months_score": delivery_score_value,
                "competitor_count_score": competitor_score,
                "overall_similarity_score": final_score,
            }
        )
        candidates.append(item)

    return (
        pd.DataFrame(candidates)
        .sort_values(
            [
                "overall_similarity_score",
                "text_embedding_score",
                "product_name_score",
                "region_score",
                "procedure_type_score",
                "quantity_score",
            ],
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def percentile_metrics(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "average": float(series.mean()),
        "std": float(series.std(ddof=0)),
    }


def predict_model_prices(models: dict[str, Any], query: dict[str, Any]) -> dict[str, float]:
    query_frame = pd.DataFrame([{field: query[field] for field in MODEL_FEATURES}])
    linear_prediction = float(models["linear_model"].predict(query_frame)[0])
    xgboost_prediction = float(models["xgboost_model"].predict(query_frame)[0])
    return {
        "linear": max(0.01, linear_prediction),
        "xgboost": max(0.01, xgboost_prediction),
    }


def build_model_supported_corridor(
    topk_corridor: dict[str, float],
    predictions: dict[str, float],
    models: dict[str, Any],
) -> dict[str, float]:
    linear_residuals = models["linear_residuals"]
    xgboost_residuals = models["xgboost_residuals"]
    linear_low = predictions["linear"] + linear_residuals["p25"]
    linear_high = predictions["linear"] + linear_residuals["p75"]
    xgboost_low = predictions["xgboost"] + xgboost_residuals["p25"]
    xgboost_high = predictions["xgboost"] + xgboost_residuals["p75"]

    return {
        "low": max(0.01, float(np.median([topk_corridor["p25"], linear_low, xgboost_low]))),
        "middle": max(
            0.01,
            float(
                np.median(
                    [
                        topk_corridor["median"],
                        predictions["linear"],
                        predictions["xgboost"],
                    ]
                )
            ),
        ),
        "high": max(0.01, float(np.median([topk_corridor["p75"], linear_high, xgboost_high]))),
        "linear_low": max(0.01, float(linear_low)),
        "linear_high": max(0.01, float(linear_high)),
        "xgboost_low": max(0.01, float(xgboost_low)),
        "xgboost_high": max(0.01, float(xgboost_high)),
    }


def corridor_coverage(
    actual_prices: pd.Series,
    low_residual: float,
    high_residual: float,
    predictions: np.ndarray,
) -> float:
    low_bounds = predictions + low_residual
    high_bounds = predictions + high_residual
    covered = (actual_prices.to_numpy(dtype=float) >= low_bounds) & (
        actual_prices.to_numpy(dtype=float) <= high_bounds
    )
    return float(covered.mean() * 100)


def model_confidence_level(
    average_similarity: float,
    topk_median: float,
    linear_prediction: float,
    xgboost_prediction: float,
) -> str:
    center_values = np.array([topk_median, linear_prediction, xgboost_prediction], dtype=float)
    center_average = float(center_values.mean())
    disagreement = float((center_values.max() - center_values.min()) / max(center_average, 1.0))
    if average_similarity >= 0.72 and disagreement <= 0.18:
        return "Yüksek"
    if average_similarity < 0.55 or disagreement >= 0.35:
        return "Düşük"
    return "Orta"


def confidence_level(count: int, average_similarity: float) -> str:
    if count < 7 or average_similarity < 0.55:
        return "Düşük"
    if count >= 10 and average_similarity >= 0.75:
        return "Yüksek"
    return "Orta"


def historical_price_fit(proposed_price: float, corridor: dict[str, float], margin_pct: float) -> dict[str, Any]:
    p25 = corridor["p25"]
    p75 = corridor["p75"]
    p90 = corridor["p90"]

    if proposed_price < p25:
        score = 75 if margin_pct >= 15 else 60 if margin_pct >= 8 else 40
        return {
            "label": "Agresif fiyat",
            "level": "Orta",
            "score": score,
            "explanation": "Önerilen fiyat, benzer kazanılmış ihalelerin alt bandında. Rekabetçi olabilir; birim kazanç kontrol edilmelidir.",
        }
    if proposed_price <= p75:
        return {
            "label": "Emsal kazanım bandında",
            "level": "Yüksek",
            "score": 90,
            "explanation": "Önerilen fiyat, benzer kazanılmış ihalelerin ana fiyat bandında kalıyor.",
        }
    if proposed_price <= p90:
        return {
            "label": "Üst emsal bandında",
            "level": "Orta",
            "score": 70,
            "explanation": "Önerilen fiyat, benzer kazanılmış ihalelerin üst fiyat bandında.",
        }
    return {
        "label": "Emsal bandının üstünde",
        "level": "Düşük",
        "score": 35,
        "explanation": "Önerilen fiyat, benzer kazanılmış ihalelerin tarihsel üst eşiğinin üzerinde.",
    }


def empirical_pwin_score(
    average_similarity: float,
    high_similarity_share: float,
    price_fit_score: float,
    one_class_score: float,
    cluster_score: float,
) -> float:
    return float(
        np.clip(
            0.30 * average_similarity * 100
            + 0.15 * high_similarity_share
            + 0.25 * price_fit_score
            + 0.15 * one_class_score
            + 0.15 * cluster_score,
            0,
            100,
        )
    )


def score_success_profiles(
    profile_models: dict[str, Any],
    query_frame: pd.DataFrame,
) -> dict[str, Any]:
    encoded_query = profile_models["one_class_preprocessor"].transform(query_frame)

    query_one_class_score = float(profile_models["one_class_model"].decision_function(encoded_query)[0])
    historical_scores = profile_models["one_class_scores"]
    one_class_percentile = float(
        np.searchsorted(historical_scores, query_one_class_score, side="right")
        / len(historical_scores)
        * 100
    )
    one_class_score = float(np.clip(one_class_percentile, 0, 100))
    if one_class_score >= 70:
        one_class_label = "Geçmiş profile uygun"
    elif one_class_score >= 45:
        one_class_label = "Sınırda"
    else:
        one_class_label = "Geçmiş profile uzak"

    cluster_query = query_frame[SUCCESS_CLUSTER_FEATURES].copy()
    encoded_cluster_query = profile_models["cluster_preprocessor"].transform(cluster_query)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cluster_distances = profile_models["cluster_model"].transform(encoded_cluster_query)[0]
    cluster_id = int(np.argmin(cluster_distances))
    cluster_distance = float(cluster_distances[cluster_id])
    cluster_reference_distances = profile_models["distance_by_cluster"][cluster_id]
    cluster_distance_percentile = float(
        np.searchsorted(cluster_reference_distances, cluster_distance, side="right")
        / len(cluster_reference_distances)
    )
    cluster_score = float(np.clip((1 - cluster_distance_percentile) * 100, 0, 100))
    if cluster_score >= 70:
        cluster_label = "Güçlü eşleşme"
    elif cluster_score >= 40:
        cluster_label = "Orta eşleşme"
    else:
        cluster_label = "Zayıf eşleşme"

    profile = profile_models["cluster_profiles"][cluster_id]
    return {
        "one_class_score": one_class_score,
        "one_class_label": one_class_label,
        "cluster_id": cluster_id,
        "cluster_score": cluster_score,
        "cluster_label": cluster_label,
        "profile": profile,
    }


def scenario_margins(corridor: dict[str, float], cost: float) -> dict[str, float]:
    scenarios = {
        "Düşük fiyat": corridor.get("low", corridor.get("p25", 0.0)),
        "Orta fiyat": corridor.get("middle", corridor.get("median", 0.0)),
        "Yüksek fiyat": corridor.get("high", corridor.get("p75", 0.0)),
    }
    return {
        name: ((price - cost) / price * 100) if price > 0 else 0
        for name, price in scenarios.items()
    }


def margin_score(margin_pct: float) -> int:
    if margin_pct >= 25:
        return 100
    if margin_pct >= 15:
        return 75
    if margin_pct >= 8:
        return 50
    if margin_pct >= 0:
        return 25
    return 0


def competition_score(competitor_count: int) -> int:
    if competitor_count <= 2:
        return 90
    if competitor_count <= 4:
        return 70
    if competitor_count <= 6:
        return 45
    return 25


def delivery_score(delivery_months: int) -> int:
    return {3: 60, 6: 75, 9: 85, 12: 90}.get(delivery_months, 75)


def attractiveness_label(score: float) -> str:
    if score >= 75:
        return "Güçlü fırsat"
    if score >= 60:
        return "Orta fiyatla ilerlenebilir"
    if score >= 45:
        return "Kazanç korunarak ilerlenebilir"
    return "Manuel inceleme gerekir"


def format_try(value: float) -> str:
    return f"{value:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    return f"%{value:.1f}"


def build_gauge(score: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 46, "color": "#172033"}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": "#16a34a", "thickness": 0.26},
                "bgcolor": "rgba(15,23,42,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(239,68,68,0.18)"},
                    {"range": [45, 60], "color": "rgba(245,158,11,0.18)"},
                    {"range": [60, 75], "color": "rgba(59,130,246,0.18)"},
                    {"range": [75, 100], "color": "rgba(34,197,94,0.18)"},
                ],
            },
        )
    )
    figure.update_layout(
        height=195,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#172033"},
    )
    return figure


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_summary(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="mini-summary">
            <div class="mini-label">{escape(label)}</div>
            <div class="mini-value">{escape(value)}</div>
            <div class="mini-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def deduplicate_model_ids(model_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    for model_id in model_ids:
        if model_id and model_id not in ordered:
            ordered.append(model_id)
    return ordered


def selected_openrouter_model_ids(primary_label: str, fallback_labels: list[str]) -> list[str]:
    return deduplicate_model_ids(
        [
            OPENROUTER_MODELS[primary_label],
            *[OPENROUTER_MODELS[label] for label in fallback_labels],
        ]
    )


def get_server_openrouter_api_key() -> str:
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        secret_key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        if secret_key:
            return secret_key

        openrouter_section = st.secrets.get("openrouter", {})
        if hasattr(openrouter_section, "get"):
            return str(openrouter_section.get("api_key", "")).strip()
    except Exception:
        return ""

    return ""


def compact_similar_tenders(similar: pd.DataFrame, limit: int = SIMILAR_TENDER_COUNT) -> list[dict[str, Any]]:
    columns = [
        "tender_id",
        "year",
        "buyer_institution",
        "product_name",
        "product_group",
        "region",
        "procedure_type",
        "quantity",
        "delivery_months",
        "competitor_count_estimate",
        "winning_unit_price_try",
        PRIMARY_PRICE_FIELD,
        "inflation_adjusted_contract_value_2026_try",
        "gross_margin_pct",
        "discount_to_estimated_cost_pct",
        "strategic_fit_score",
        "overall_similarity_score",
    ]
    records = []
    for row in similar.head(limit)[columns].to_dict(orient="records"):
        records.append(
            {
                "tender_id": row["tender_id"],
                "year": int(row["year"]),
                "buyer_institution": row["buyer_institution"],
                "product_name": row["product_name"],
                "product_group": row["product_group"],
                "region": row["region"],
                "procedure_type": row["procedure_type"],
                "quantity": int(row["quantity"]),
                "delivery_months": int(row["delivery_months"]),
                "competitor_count_estimate": int(row["competitor_count_estimate"]),
                "winning_unit_price_try": round(float(row["winning_unit_price_try"]), 2),
                "unit_price_2026_try": round(float(row[PRIMARY_PRICE_FIELD]), 2),
                "contract_value_2026_try": round(float(row["inflation_adjusted_contract_value_2026_try"]), 2),
                "gross_margin_pct": round(float(row["gross_margin_pct"]), 2),
                "discount_to_estimated_cost_pct": round(float(row["discount_to_estimated_cost_pct"]), 2),
                "strategic_fit_score": round(float(row["strategic_fit_score"]), 2),
                "overall_similarity_score": round(float(row["overall_similarity_score"]), 4),
            }
        )
    return records


def build_llm_context(
    query: dict[str, Any],
    estimated_unit_cost_try: float,
    similar: pd.DataFrame,
    price_corridor: dict[str, float],
    nominal_reference: dict[str, float],
    margin_benchmark: dict[str, float],
    discount_benchmark: dict[str, float],
    model_corridor: dict[str, float],
    scenario_prices: dict[str, float],
    margins: dict[str, float],
    model_predictions: dict[str, float],
    models: dict[str, Any],
    model_confidence: str,
    model_disagreement_pct: float,
    price_fit: dict[str, Any],
    success_profile_scores: dict[str, Any],
    empirical_pwin: float,
    pwin_contributions: dict[str, float],
    final_score_components: dict[str, float],
    final_attractiveness_score: float,
    final_label: str,
    strong_similar_count: int,
) -> dict[str, Any]:
    profile = success_profile_scores["profile"]
    return {
        "goal": (
            "p(win)'i artıran, ancak sağlıksız/negatif marja düşmeyen teklif "
            "stratejisini yorumla. p(win) gerçek kazan/kaybet olasılığı değil; "
            "sadece kazanılmış emsal hafızasına dayalı karar destek göstergesidir."
        ),
        "data_scope": {
            "dataset": "X İlaç Şirketi sentetik demo verisi",
            "record_type": "Sadece geçmişte kazanılmış ihaleler",
            "only_won_tenders": True,
            "lost_tenders_available": False,
            "classification_model_available": False,
            "interpretation_limit": (
                "Kaybedilmiş ihale olmadığı için gerçek kazan/kaybet olasılığı, "
                "rakip davranışı veya sınıflandırma modeli üretilemez."
            ),
            "similar_tender_count": int(len(similar)),
            "strong_similar_count": int(strong_similar_count),
            "high_similarity_threshold": 0.70,
            "price_basis": "Mayıs 2026 TL seviyesine normalize birim fiyat",
        },
        "methodology": {
            "retrieval": (
                "Geçmiş kazanılmış ihaleler TF-IDF + SVD tabanlı yerel embedding ve "
                "ürün, bölge, usul, miktar, teslim süresi, rakip sayısı ağırlıklarıyla sıralanır."
            ),
            "pricing": (
                "Top-k fiyat dağılımı, Linear Regression ve XGBoost tahminleri median mantığıyla "
                "birleştirilerek düşük/orta/yüksek fiyat koridoru oluşturulur."
            ),
            "pwin": (
                "Emsal p(win), kazanılmış emsallere yakınlık göstergesidir; gerçek kazanma ihtimali değildir."
            ),
            "profile_models": (
                "Isolation Forest yeni ihalenin kazanılmış iş profiline normal/uzaklığını; "
                "KMeans ise hangi geçmiş başarı kümesine benzediğini gösterir."
            ),
        },
        "new_tender": {
            **query,
            "estimated_unit_cost_try": round(float(estimated_unit_cost_try), 2),
        },
        "pwin": {
            "empirical_pwin_pct": round(float(empirical_pwin), 2),
            "formula": (
                "0.30*ortalama_benzerlik + 0.15*güçlü_emsal_oranı + "
                "0.25*fiyat_bandı_uyumu + 0.15*IsolationForest + 0.15*KMeans"
            ),
            "contributions": {key: round(float(value), 2) for key, value in pwin_contributions.items()},
        },
        "pricing": {
            "topk_price_corridor_2026_try": {
                "min": round(float(price_corridor["min"]), 2),
                "p25": round(float(price_corridor["p25"]), 2),
                "median": round(float(price_corridor["median"]), 2),
                "p75": round(float(price_corridor["p75"]), 2),
                "p90": round(float(price_corridor["p90"]), 2),
                "max": round(float(price_corridor["max"]), 2),
                "average": round(float(price_corridor["average"]), 2),
                "std": round(float(price_corridor["std"]), 2),
            },
            "nominal_winning_price_reference_try": {
                key: round(float(nominal_reference[key]), 2)
                for key in ["min", "p25", "median", "p75", "p90", "max", "average", "std"]
            },
            "model_supported_prices_2026_try": {
                "low": round(float(scenario_prices["Düşük fiyat"]), 2),
                "middle": round(float(scenario_prices["Orta fiyat"]), 2),
                "high": round(float(scenario_prices["Yüksek fiyat"]), 2),
            },
            "model_corridor_internal_values_2026_try": {
                key: round(float(model_corridor[key]), 2)
                for key in ["linear_low", "linear_high", "xgboost_low", "xgboost_high"]
            },
            "margin_by_price_option_pct": {
                key: round(float(value), 2) for key, value in margins.items()
            },
            "historical_margin_benchmark_pct": {
                key: round(float(value), 2) for key, value in margin_benchmark.items()
            },
            "historical_discount_to_estimated_cost_benchmark_pct": {
                key: round(float(value), 2) for key, value in discount_benchmark.items()
            },
            "linear_prediction_2026_try": round(float(model_predictions["linear"]), 2),
            "xgboost_prediction_2026_try": round(float(model_predictions["xgboost"]), 2),
            "model_confidence": model_confidence,
            "model_disagreement_pct": round(float(model_disagreement_pct), 2),
            "price_fit_label": price_fit["label"],
            "price_fit_level": price_fit["level"],
            "price_fit_explanation": price_fit["explanation"],
        },
        "learner_outputs": {
            "isolation_forest": {
                "label": success_profile_scores["one_class_label"],
                "score_0_100": round(float(success_profile_scores["one_class_score"]), 2),
            },
            "kmeans": {
                "cluster_id": int(success_profile_scores["cluster_id"]),
                "label": success_profile_scores["cluster_label"],
                "score_0_100": round(float(success_profile_scores["cluster_score"]), 2),
                "profile_name": profile["name"],
                "profile_count": int(profile["count"]),
                "profile_median_price_2026_try": round(float(profile["median_price"]), 2),
                "profile_median_margin_pct": round(float(profile["median_margin"]), 2),
            },
            "linear_regression": {
                "prediction_2026_try": round(float(model_predictions["linear"]), 2),
                "mae_try": round(float(models["linear_residuals"]["mae"]), 2),
                "mape_pct": round(float(models["linear_residuals"]["mape"]), 2),
                "coverage_pct": round(float(models["linear_residuals"]["coverage"]), 2),
            },
            "xgboost": {
                "prediction_2026_try": round(float(model_predictions["xgboost"]), 2),
                "mae_try": round(float(models["xgboost_residuals"]["mae"]), 2),
                "mape_pct": round(float(models["xgboost_residuals"]["mape"]), 2),
                "coverage_pct": round(float(models["xgboost_residuals"]["coverage"]), 2),
            },
        },
        "opportunity_priority": {
            "score_0_100": round(float(final_attractiveness_score), 2),
            "label": final_label,
            "components": {key: round(float(value), 2) for key, value in final_score_components.items()},
        },
        "retrieved_evidence": compact_similar_tenders(similar),
    }


def build_llm_prompt(context: dict[str, Any], user_question: str | None = None) -> str:
    question = user_question.strip() if user_question else ""
    return f"""
Aşağıdaki JSON, bir ihale karar destek uygulamasının hesapladığı tüm ana çıktılarıdır.
Görevin hesap yapmak değil, bu çıktıları yöneticiye doğru bağlamla yorumlamaktır.

Kritik bağlam:
- Veri sadece geçmişte kazanılmış ihalelerden oluşur; kaybedilen ihale yoktur.
- Bu nedenle gerçek kazan/kaybet olasılığı, supervised classification veya rakip bazlı kazanma tahmini yapılamaz.
- p(win), "bu yeni ihale geçmişte kazandığımız işlere, fiyat bandımıza ve başarı profillerimize ne kadar benziyor?" sorusunun emsal bazlı karar destek göstergesidir.
- Kazanılmış verilerden şunlar yapılabildi: benzer ihale retrieval, Mayıs 2026'ya normalize fiyat koridoru, Linear/XGBoost fiyat tahmini, IsolationForest kazanım profili yakınlığı, KMeans başarı profili eşleşmesi ve fırsat öncelik puanı.
- Verilmeyen bilgiyi uydurma, sayısal değerleri değiştirme, sadece MODEL_CONTEXT_JSON içeriğine dayan.

Yanıtı Türkçe, doğal sohbet diliyle ve kısa Markdown başlıklarıyla ver.
JSON döndürme. Uzun ham veri tekrarlama. Business kullanıcısına anlatır gibi temelden başla,
sonra teknik sinyalleri sade ama doğru açıkla. Gerektiğinde p(win), fiyat koridoru, marj,
IsolationForest, KMeans, Linear Regression ve XGBoost çıktılarını birlikte yorumla.
Sorunun kapsamına göre şu düzeni kullan:
- Kısa cevap
- İş açısından anlamı
- Teknik dayanak
- Fiyat / marj yorumu
- Riskler
- Sonraki aksiyon

Kullanıcı sorusu varsa özellikle onu cevapla: {question or "Genel yönetici yorumu üret."}

MODEL_CONTEXT_JSON:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def parse_llm_json_response(content: str) -> dict[str, Any] | None:
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


def flatten_llm_response(result: dict[str, Any]) -> str:
    parsed = result.get("parsed")
    if not parsed:
        return result.get("raw", "")

    lines: list[str] = []
    for key, label in [
        ("decision_summary", "Yönetici Özeti"),
        ("data_situation", "Veri Durumu"),
        ("recommended_action", "Önerilen Aksiyon"),
        ("pwin_interpretation", "p(win) Yorumu"),
        ("pricing_interpretation", "Fiyat Yorumu"),
        ("margin_risk", "Marj Riski"),
    ]:
        if parsed.get(key):
            lines.append(f"**{label}**\n{parsed[key]}")

    learner_signals = parsed.get("learner_signals")
    if isinstance(learner_signals, dict) and learner_signals:
        signal_lines = [f"- {label}: {value}" for label, value in learner_signals.items()]
        lines.append("**Learner Sinyalleri**\n" + "\n".join(signal_lines))

    for key, label in [
        ("supporting_evidence", "Kanıtlar"),
        ("risks", "Riskler"),
        ("next_actions", "Sonraki Adımlar"),
    ]:
        values = parsed.get(key)
        if isinstance(values, list) and values:
            lines.append(f"**{label}**\n" + "\n".join(f"- {value}" for value in values))

    model = result.get("model")
    if model:
        lines.append(f"`Model: {model}`")
    return "\n\n".join(lines)


def call_openrouter_interpretation(
    api_key: str,
    model_ids: list[str],
    context: dict[str, Any],
    user_question: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Tender IQ",
    }
    referer = os.getenv("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Sen kamu ihalesi fiyatlandırma ve karar destek model çıktılarını "
                "yorumlayan kıdemli bir analistsin. Sadece verilen structured JSON'a "
                "dayan. Hesap yapma, sayı uydurma, veri kapsamını abartma. Veride "
                "kaybedilmiş ihaleler olmadığı için p(win)'i gerçek kazanma olasılığı "
                "gibi sunma; kazanılmış emsallere dayalı uygunluk göstergesi olarak "
                "açıkla. Önce business anlamı sade biçimde ver, sonra teknik dayanağı "
                "temelden ama doğru şekilde anlat. Yöneticiye karar, risk, fiyat ve "
                "sonraki aksiyonları net, temiz ve yapılandırılmış biçimde aktar."
            ),
        },
    ]
    if conversation_history:
        messages.extend(conversation_history[-8:])
    messages.append({"role": "user", "content": build_llm_prompt(context, user_question)})
    errors: list[str] = []
    for model_id in model_ids:
        body = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2600,
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in {401, 403}:
                raise
            response_text = exc.response.text if exc.response is not None else str(exc)
            errors.append(f"{model_id}: {response_text}")
            continue
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            errors.append(f"{model_id}: beklenmeyen yanıt formatı ({exc})")
            continue

        return {
            "model": data.get("model", model_id),
            "attempted_models": model_ids,
            "raw": content,
            "parsed": parse_llm_json_response(content),
        }

    raise RuntimeError("Tüm OpenRouter modelleri başarısız oldu: " + " | ".join(errors))


def render_llm_structured_response(result: dict[str, Any]) -> None:
    parsed = result.get("parsed")
    if not parsed:
        st.markdown(result.get("raw", ""))
        return

    st.markdown(f"**Kullanılan model:** `{result.get('model', 'bilinmiyor')}`")
    for key, label in [
        ("decision_summary", "Yönetici Özeti"),
        ("data_situation", "Veri Durumu"),
        ("recommended_action", "Önerilen Aksiyon"),
        ("pwin_interpretation", "p(win) Yorumu"),
        ("pricing_interpretation", "Fiyat Yorumu"),
        ("margin_risk", "Marj Riski"),
    ]:
        if parsed.get(key):
            st.markdown(f"**{label}**")
            st.write(parsed[key])

    learner_signals = parsed.get("learner_signals")
    if isinstance(learner_signals, dict):
        st.markdown("**Learner Sinyalleri**")
        st.json(learner_signals, expanded=False)

    for key, label in [
        ("supporting_evidence", "Kanıtlar"),
        ("risks", "Riskler"),
        ("next_actions", "Sonraki Adımlar"),
    ]:
        values = parsed.get(key)
        if isinstance(values, list) and values:
            st.markdown(f"**{label}**")
            for value in values:
                st.write(f"- {value}")


def render_llm_chatbot(llm_context: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="section-kicker">LLM yorum katmanı</div>
        <div class="section-title">OpenRouter Analist Chatbotu</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Bu bölüm yeni hesap yapmaz. Analizdeki p(win), fiyat koridoru, benzer ihaleler, "
        "IsolationForest, KMeans, Linear Regression ve XGBoost çıktılarını LLM'e structured context "
        "olarak gönderir ve yönetici yorumu üretir."
    )

    openrouter_api_key = get_server_openrouter_api_key()
    if (
        st.session_state.get("llm_model_defaults_version") != "google-primary-v1"
        or st.session_state.get("llm_primary_label") not in DEFAULT_OPENROUTER_MODEL_LABELS
    ):
        st.session_state.llm_primary_label = DEFAULT_OPENROUTER_MODEL_LABELS[0]
        st.session_state.llm_fallback_labels = DEFAULT_OPENROUTER_MODEL_LABELS[1:3]
        st.session_state.llm_model_defaults_version = "google-primary-v1"
    if "llm_fallback_labels" not in st.session_state:
        st.session_state.llm_fallback_labels = DEFAULT_OPENROUTER_MODEL_LABELS[1:3]

    context_signature = json.dumps(
        {
            "new_tender": llm_context.get("new_tender", {}),
            "pwin": llm_context.get("pwin", {}),
            "pricing": llm_context.get("pricing", {}).get("model_supported_prices_2026_try", {}),
            "profile": llm_context.get("learner_outputs", {}).get("kmeans", {}).get("profile_name", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if st.session_state.get("llm_chat_context_signature") != context_signature:
        st.session_state.llm_chat_context_signature = context_signature
        st.session_state.llm_pending_user_message = ""
        st.session_state.llm_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Analiz bağlamı hazır. Bu ihaleyi p(win), fiyat koridoru, marj, "
                    "IsolationForest, KMeans, Linear Regression ve XGBoost sinyalleriyle birlikte "
                    "yorumlayabilirim. Sorunu yazabilirsin."
                ),
            }
        ]

    with st.container(border=True):
        active_model_ids = selected_openrouter_model_ids(
            st.session_state.llm_primary_label,
            st.session_state.llm_fallback_labels,
        )
        st.markdown(
            f"""
            <div class="chat-screen-header">
                <div class="chat-screen-title">Sohbet ekranı</div>
                <div class="chat-screen-meta">
                    Aktif primary model: {escape(OPENROUTER_MODELS[st.session_state.llm_primary_label])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if openrouter_api_key:
            st.caption("LLM aktif. Sorular güncel analiz context'iyle yanıtlanır.")
        else:
            st.warning("LLM chatbot için server API key tanımlı değil.")

        with st.container(height=430, border=False):
            for message in st.session_state.get("llm_chat_messages", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            pending_user_message = st.session_state.get("llm_pending_user_message")
            if pending_user_message:
                if not openrouter_api_key:
                    assistant_message = (
                        "LLM chatbot şu anda kullanılamıyor. Server tarafında OPENROUTER_API_KEY "
                        "veya Streamlit secrets tanımlanmalı."
                    )
                    with st.chat_message("assistant"):
                        st.warning(assistant_message)
                else:
                    history = [
                        {"role": item["role"], "content": item["content"]}
                        for item in st.session_state.llm_chat_messages[:-1]
                        if item["role"] in {"user", "assistant"}
                    ]
                    with st.chat_message("assistant"):
                        with st.spinner("Cevap hazırlanıyor..."):
                            try:
                                llm_result = call_openrouter_interpretation(
                                    api_key=openrouter_api_key,
                                    model_ids=active_model_ids,
                                    context=llm_context,
                                    user_question=pending_user_message,
                                    conversation_history=history,
                                )
                            except requests.HTTPError as exc:
                                response_text = exc.response.text if exc.response is not None else str(exc)
                                assistant_message = f"OpenRouter isteği başarısız oldu: {response_text}"
                                st.error(assistant_message)
                            except requests.RequestException as exc:
                                assistant_message = f"OpenRouter bağlantı hatası: {exc}"
                                st.error(assistant_message)
                            except (KeyError, IndexError, ValueError, TypeError, RuntimeError) as exc:
                                assistant_message = f"LLM yorumu alınamadı veya yanıt beklenen formatta değil: {exc}"
                                st.error(assistant_message)
                            else:
                                assistant_message = flatten_llm_response(llm_result)
                                st.markdown(assistant_message)

                st.session_state.llm_chat_messages.append(
                    {"role": "assistant", "content": assistant_message}
                )
                st.session_state.llm_pending_user_message = ""
                st.rerun()

        user_message = st.chat_input("Bu ihale hakkında sorunuzu yazın...")
        if user_message:
            st.session_state.llm_chat_messages.append({"role": "user", "content": user_message})
            st.session_state.llm_pending_user_message = user_message
            st.rerun()

    with st.container(border=True):
        st.markdown("##### Chatbot ayarları")
        llm_primary_label = st.selectbox(
            "Chatbot modeli",
            DEFAULT_OPENROUTER_MODEL_LABELS,
            key="llm_primary_label",
        )
        fallback_default = [
            label for label in DEFAULT_OPENROUTER_MODEL_LABELS if label != llm_primary_label
        ][:2]
        if not st.session_state.get("llm_fallback_labels"):
            st.session_state.llm_fallback_labels = fallback_default
        else:
            st.session_state.llm_fallback_labels = [
                label for label in st.session_state.llm_fallback_labels if label != llm_primary_label
            ]
        st.multiselect(
            "Fallback modeller",
            [label for label in DEFAULT_OPENROUTER_MODEL_LABELS if label != llm_primary_label],
            key="llm_fallback_labels",
            help="Primary model cevap veremezse uygulama bu modelleri sırayla dener.",
        )
        current_model_ids = selected_openrouter_model_ids(
            st.session_state.llm_primary_label,
            st.session_state.llm_fallback_labels,
        )
        model_chain = " -> ".join(f"`{model_id}`" for model_id in current_model_ids)
        st.markdown(f"**Model sırası:** {model_chain}")
        st.caption(
            f"Seçim kaydedildi. Şu anda primary model `{OPENROUTER_MODELS[st.session_state.llm_primary_label]}`."
        )

        control_cols = st.columns([1.1, 1.1, 2.4], gap="small")
        with control_cols[0]:
            clear_chat = st.button("Konuşmayı temizle", width="stretch")
        with control_cols[1]:
            show_context = st.checkbox("Structured context", value=False)
        if clear_chat:
            st.session_state.llm_pending_user_message = ""
            st.session_state.llm_chat_messages = [
                {
                    "role": "assistant",
                    "content": "Konuşmayı sıfırladım. Bu analiz için yeni sorunuzu yazabilirsiniz.",
                }
            ]
            st.rerun()
        if show_context:
            st.json(llm_context, expanded=False)


def render_how_it_works_tab() -> None:
    st.markdown(
        """
        <div class="section-kicker">Şeffaf metodoloji</div>
        <div class="section-title">Sistem Nasıl Çalışır?</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Bu ekran bir kazanma ihtimali modeli değildir. Veri sadece geçmişte
        kazanılmış ihaleleri içerdiği için sistemin amacı, yeni bir ihale için
        geçmiş kazanım hafızasına dayalı fiyat bandı, birim kazanç ve karar destek
        puanı üretmektir.

        **1. Veri kullanımı**

        Her geçmiş ihale satırı ürün, ürün grubu, alıcı kurum, bölge, ihale
        usulü, miktar, teslim süresi, tahmini rakip sayısı, kazanılmış fiyat,
        birim kazanç ve stratejik uyum bilgilerini taşır. Ana fiyat alanı
        `inflation_adjusted_unit_price_2026_try` kolonudur; eski yıllardaki
        fiyatlar Mayıs 2026 TL seviyesine taşınmış haliyle karşılaştırılır.

        **2. Enflasyon hesabı ve fiyat normalizasyonu**

        Geçmiş ihalelerdeki TL fiyatları doğrudan karşılaştırmak doğru değildir.
        Örneğin 2021 yılında 10 TL olan bir birim fiyat ile 2025 yılında 10 TL
        olan bir birim fiyat aynı ekonomik anlama gelmez. Bu yüzden sistem,
        geçmiş fiyatları tek bir ortak seviyeye taşır:

        ```text
        Mayıs 2026 TL seviyesi
        ```

        Veri dosyasında her ihale yılı için `cpi_factor_to_2026` adında bir
        katsayı bulunur. Bu katsayı, o yılın fiyatını Mayıs 2026 seviyesine
        taşımak için kullanılır.

        Temel formül:

        ```text
        Mayıs 2026'ya taşınmış birim fiyat =
        o günkü kazanan birim fiyat x cpi_factor_to_2026
        ```

        Örnek:

        ```text
        2021 kazanan birim fiyat: 10 TL
        2021 için CPI katsayısı: 5.9647

        Mayıs 2026 seviyesindeki fiyat:
        10 x 5.9647 = 59.65 TL
        ```

        Bu yüzden fiyat karşılaştırmalarında ana kolon şudur:

        ```text
        inflation_adjusted_unit_price_2026_try
        ```

        Toplam tutar için de aynı mantık kullanılır:

        ```text
        Mayıs 2026'ya taşınmış sözleşme tutarı =
        o günkü sözleşme tutarı x cpi_factor_to_2026
        ```

        Önemli ayrım: TL fiyatlar enflasyonla Mayıs 2026 seviyesine taşınır,
        ama yüzde oranları ayrıca enflasyona taşınmaz. Örneğin birim kazanç
        oranı veya yaklaşık maliyet altı oranı zaten yüzde olduğu için kendi
        dönemindeki fiyat/maliyet ilişkisini gösterir.

        **3. Yeni ihale girdisi**

        Kullanıcı ürün adı, ürün grubu, bölge, ihale usulü, alıcı kurum,
        tahmini birim maliyet, miktar, teslim süresi ve tahmini rakip sayısını
        girer. Bu bilgiler yeni ihale sorgusu olarak kullanılır.

        **4. Benzerlik skoru**

        Sistem her geçmiş ihale için `overall_similarity_score` hesaplar. Metin
        alanları önce sayısal vektöre çevrilir, sonra SVD ile daha yoğun bir
        yerel metin embedding'i oluşturulur. Yeni ihale de aynı embedding
        uzayına taşınır ve cosine similarity ile metin yakınlığı ölçülür. Ürün
        grubu, bölge ve ihale usulü birebir eşleşme skoru alır. Ürün adı token
        ortaklığına göre daha esnek değerlendirilir. Miktar, teslim süresi ve
        rakip sayısı için yakınlık skoru şu mantıkla hesaplanır:

        ```text
        yakınlık = 1 - mutlak_fark / büyük_değer
        ```

        Nihai benzerlik formülü:

        ```text
        overall_similarity_score =
        0.45 * text_embedding_score (ihale metni, ürün, kurum, bölge ve usul metinlerinin vektör benzerliği)
        + 0.12 * product_group_score (ürün grubu aynı mı; örn. IV Solution)
        + 0.10 * product_name_score (ürün adı ne kadar benziyor; token ve kısmi eşleşme)
        + 0.08 * region_score (bölge aynı mı; örn. Marmara)
        + 0.05 * procedure_type_score (ihale usulü aynı mı; örn. Açık İhale)
        + 0.10 * quantity_score (miktar ölçeği ne kadar yakın)
        + 0.05 * delivery_months_score (teslim süresi ne kadar yakın)
        + 0.05 * competitor_count_score (tahmini rakip sayısı ne kadar yakın)
        ```

        Sistem tüm geçmiş ihaleleri bu skora göre sıralar. Hesaplamalarda en
        benzer 50 ihale kullanılır; ekranda açıklanabilirlik için ilk 10 ihale
        gösterilir. Bu MVP harici API kullanmadan yerel embedding üretir. Daha
        güçlü bir üretim versiyonunda aynı yapı SentenceTransformer veya OpenAI
        embedding gibi transformer tabanlı embedding servisiyle değiştirilebilir.

        **5. Top-k fiyat benchmark'ı**

        En benzer 50 ihalenin Mayıs 2026 fiyatlarından `p25`, `median`, `p75`,
        ortalama ve dağılım hesaplanır. Bu bölüm geçmiş kazanılmış ihalelerden
        gelen açıklanabilir fiyat hafızasıdır.

        **6. Linear Regression ve XGBoost ne için kullanılır?**

        Benzer ihaleler bize açıklanabilir bir tarihsel referans verir; ancak
        sadece en benzer ihalelerin medyanına bakmak bazen yeterli olmaz. Bu
        yüzden iki ayrı fiyat tahmin modeli de eğitilir. İki modelin amacı
        aynıdır:

        ```text
        Yeni ihale özelliklerine göre beklenen Mayıs 2026 birim fiyatını tahmin etmek.
        ```

        İki model de tüm geçmiş veriyle eğitilir. Hedef değişken:

        ```text
        inflation_adjusted_unit_price_2026_try
        ```

        Model girdileri:

        ```text
        product_name, product_group, region, procedure_type, buyer_institution,
        quantity, delivery_months, competitor_count_estimate
        ```

        `year` model girdisi olarak kullanılmaz. Çünkü hedef fiyat zaten Mayıs
        2026 seviyesine normalize edilmiştir. Yılı tekrar modele vermek,
        özellikle yeni ihale 2026 gibi eğitim verisinde olmayan bir yıl olarak
        girildiğinde Linear Regression tarafında hatalı extrapolation
        yaratabilir.

        Kategorik alanlar One-Hot Encoding ile sayısallaştırılır; sayısal
        alanlar StandardScaler ile ölçeklenir.

        Linear Regression daha basit ve açıklanabilir bir modeldir. Şunu
        öğrenmeye çalışır: ürün, bölge, ihale usulü, miktar, teslim süresi ve
        rakip sayısı fiyatı ortalama olarak hangi yönde etkiliyor? Bu model
        sistemde temel ve daha stabil bir fiyat tahmini üretir.

        XGBoost daha esnek bir modeldir. Değişkenler arasındaki doğrusal olmayan
        ilişkileri ve kombinasyon etkilerini yakalamaya çalışır. Örneğin aynı
        miktar artışı bazı ürünlerde fiyatı ciddi düşürürken, bazı ürünlerde
        daha az etkileyebilir; XGBoost bu tip kırılımları yakalamak için
        kullanılır.

        İki model de kazanma ihtimali tahmini yapmaz. İkisi de sadece fiyat
        tahmini üretir. Sistem, bu tahminleri top-k geçmiş fiyat dağılımıyla
        birlikte değerlendirir.

        Model güveni düşükse bu her zaman teknik hata anlamına gelmez. Genelde
        şu anlama gelir: benzer geçmiş ihalelerin fiyat hafızası, Linear
        Regression tahmini ve XGBoost tahmini aynı fiyat seviyesinde buluşmuyor.
        Bu durumda sistem fiyatı yine üretir ama kullanıcıya manuel inceleme
        sinyali verir.

        **7. Backtest ve hata payı**

        Sistem 5-fold cross validation ile modelleri geçmiş veri üzerinde test
        eder. Her kayıt için:

        ```text
        residual = gerçek_fiyat - tahmin_fiyatı
        ```

        Residual dağılımından `p25`, `median`, `p75`, MAE, MAPE ve coverage
        hesaplanır. Bu hata payları yeni ihaledeki model tahminlerine koridor
        oluşturmak için eklenir.

        **8. Nihai fiyat koridoru**

        Üç kaynak birleştirilir: top-k fiyat dağılımı, Linear Regression tahmini
        ve XGBoost tahmini. Buradaki amaç tek bir kaynağa güvenmemektir. Top-k
        geçmiş fiyatlar açıklanabilir referanstır; Linear Regression daha basit
        model tahminidir; XGBoost daha esnek model tahminidir. Tek bir modelin
        uçuk tahmini sonucu sürüklememesi için median kullanılır.

        ```text
        düşük fiyat =
        median(
          topk_p25,      yani en benzer 50 ihalenin alt fiyat seviyesi,
          linear_low,    yani Linear tahmin + Linear modelin düşük hata payı,
          xgboost_low    yani XGBoost tahmin + XGBoost modelin düşük hata payı
        )

        orta fiyat =
        median(
          topk_median,          yani en benzer 50 ihalenin orta fiyatı,
          linear_prediction,    yani Linear Regression fiyat tahmini,
          xgboost_prediction    yani XGBoost fiyat tahmini
        )

        yüksek fiyat =
        median(
          topk_p75,       yani en benzer 50 ihalenin üst fiyat seviyesi,
          linear_high,    yani Linear tahmin + Linear modelin yüksek hata payı,
          xgboost_high    yani XGBoost tahmin + XGBoost modelin yüksek hata payı
        )
        ```

        **9. Birim kazanç hesabı**

        Girilen tahmini birim maliyet ile model destekli fiyatlar
        karşılaştırılır:

        ```text
        birim_kazanç_oranı = (fiyat - tahmini_birim_maliyet) / fiyat * 100
        ```

        **10. İhale öncelik puanı**

        İhale öncelik puanı 0-100 arasıdır ve şu bileşenlerden oluşur:

        ```text
        ihale_puanı =
        0.25 * benzerlik_puanı (seçilen top-k geçmiş ihaleler yeni ihaleye ne kadar benziyor)
        + 0.30 * birim_kazanç_puanı (model destekli orta fiyatla beklenen kazanç ne kadar sağlıklı)
        + 0.20 * stratejik_uyum (benzer geçmiş ihalelerdeki stratejik fit ortalaması)
        + 0.15 * rekabet_puanı (tahmini rakip sayısı azaldıkça puan artar)
        + 0.10 * teslim_süresi_puanı (teslim süresi iş kuralına göre puanlanır)
        ```

        Sonuç etiketi puana göre verilir: `Güçlü fırsat`, `Orta fiyatla
        ilerlenebilir`, `Kazanç korunarak ilerlenebilir` veya `Manuel inceleme
        gerekir`.

        **11. One-Class Classification: Kazanım profiline yakınlık**

        Normal kazan/kaybet modelleri iki sınıf ister. Bu veri setinde sadece
        kazanılmış ihaleler olduğu için sistem ayrıca One-Class yaklaşımı
        kullanır. Buradaki soru şudur:

        ```text
        Yeni ihale, geçmişte kazandığımız ihalelerin genel profiline normal görünüyor mu?
        ```

        Bunun için Isolation Forest modeli eğitilir. Model sadece kazanılmış
        ihalelerin ürün, kurum, bölge, usul, miktar, teslim süresi, rakip sayısı,
        fiyat ve birim kazanç profilini görür. Yeni ihale bu profilin içinde kalıyorsa
        `Kazanım Profili Yakınlığı` yüksek çıkar. Profilin dışına düşüyorsa skor
        düşer ve manuel inceleme sinyali verir.

        **12. Cluster-Based Success Profiles: Hangi başarı tipine benziyor?**

        Sistem geçmiş kazanılmış ihaleleri KMeans ile 4 başarı profiline ayırır.
        Bu profiller şuna benzer iş kümelerini temsil eder:

        ```text
        yüksek hacimli / orta kazançlı standart ürün işleri
        düşük hacimli / yüksek kazançlı özel ürün işleri
        belirli bölgelerde sık kazanılan kurum işleri
        ```

        Yeni ihale geldiğinde sistem hangi başarı profiline en yakın olduğunu
        bulur. Böylece sadece skor verilmez; aynı zamanda “bu ihale geçmişte
        kazandığımız hangi tip işe benziyor?” sorusu cevaplanır.

        **13. Emsal p(win) nasıl hesaplanır?**

        `Emsal p(win)` gerçek bir kazan/kaybet olasılık modeli değildir. Müşteri
        tarafında oran gibi okunabilmesi için 0-100 arası verilen emsal kazanım
        emsal bazlı karar destek oranıdır. Beş sinyali birleştirir:

        ```text
        Emsal p(win) =
        %30 ortalama benzerlik
        + %15 güçlü emsal oranı
        + %25 tarihsel fiyat bandı uyumu
        + %15 One-Class kazanım profili yakınlığı
        + %15 başarı profili yakınlığı
        ```

        Bu oran şunu anlatır: yeni ihale geçmişte kazanılmış emsallere ne kadar
        benziyor, önerilen fiyat geçmiş kazanım bandında mı, ihale genel kazanım
        profilimizin içinde mi ve hangi başarı segmentine yakın?
        """
    )


data_mtime_ns = DATA_PATH.stat().st_mtime_ns
df = load_dataset(data_mtime_ns)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">TI</div>
        <div class="sidebar-title">TENDER IQ</div>
        <div class="sidebar-note">
            Geçmişte kazanılmış benzer ihalelere bakarak fiyat aralığı,
            birim kazanç ve ihale puanı gösterir. Kazanma ihtimali hesaplamaz.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Veri")
    st.write(f"{len(df):,} geçmiş kazanılmış ihale kaydı".replace(",", "."))
    st.write(f"{df['year'].min()}-{df['year'].max()} dönemini kapsar")
    st.write("Ürün, kurum, bölge, miktar, fiyat, birim kazanç ve teslim bilgileri içerir")
    st.write("Gerçek şirket verisi değil; X İlaç Şirketi için hazırlanmış sentetik demo verisidir")
    st.download_button(
        label="Veriyi indir",
        data=load_dataset_download_bytes(data_mtime_ns),
        file_name=DATA_PATH.name,
        mime="text/csv",
        width="stretch",
    )

header_left, header_right = st.columns([5, 1.4])
with header_left:
    st.markdown(
        """
        <div class="eyebrow">İhale Karar Yardımcısı</div>
        <h1 class="page-title">Geçmiş İhaleye Göre Fiyat Aralığı</h1>
        <p class="page-subtitle">
            Yeni bir ihale için benzer geçmiş kazanımları bulur, fiyatları
            Mayıs 2026 seviyesine taşır, olası fiyat aralığını ve birim kazancı gösterir.
        </p>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        """
        <div class="scope-pill"><span class="scope-dot"></span> KAZANILAN İHALE HAFIZASI</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="scope-note">
        Bu demo sentetik veriye dayanır. Veri sadece geçmişte kazanılmış
        ihaleleri içerir. Bu ekran kazanma ihtimali vermez; sadece benzer
        kazanılmış ihalelerden fiyat ve birim kazanç kıyaslaması yapar.
    </div>
    """,
    unsafe_allow_html=True,
)

analysis_tab, how_it_works_tab, llm_tab = st.tabs(["Analiz", "Nasıl çalışır?", "LLM yorum katmanı"])

with how_it_works_tab:
    render_how_it_works_tab()

with analysis_tab:
    product_names = sorted(df["product_name"].dropna().unique())
    product_group_by_name = (
        df.dropna(subset=["product_name", "product_group"])
        .drop_duplicates("product_name")
        .set_index("product_name")["product_group"]
        .to_dict()
    )
    regions = sorted(df["region"].dropna().unique())
    procedure_types = sorted(df["procedure_type"].dropna().unique())
    buyer_institutions = sorted(df["buyer_institution"].dropna().unique())

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Yeni ihale</div>
            <div class="section-title">İhale bilgilerini girin</div>
            """,
            unsafe_allow_html=True,
        )
        row_1 = st.columns([1.5, 1.0, 1.0, 1.1, 1.4], gap="small")
        with row_1[0]:
            product_name = st.selectbox(
                "Ürün adı",
                product_names,
                index=product_names.index("%0.9 NaCl 500 ml")
                if "%0.9 NaCl 500 ml" in product_names
                else 0,
            )
        with row_1[1]:
            product_group = product_group_by_name.get(product_name, "")
            st.text_input(
                "Ürün grubu",
                value=product_group,
                disabled=True,
            )
        with row_1[2]:
            region = st.selectbox(
                "Bölge",
                regions,
                index=regions.index("Marmara") if "Marmara" in regions else 0,
            )
        with row_1[3]:
            procedure_type = st.selectbox(
                "İhale usulü",
                procedure_types,
                index=procedure_types.index("Açık İhale") if "Açık İhale" in procedure_types else 0,
            )
        with row_1[4]:
            buyer_institution = st.selectbox(
                "Alıcı kurum",
                buyer_institutions,
                index=buyer_institutions.index("İstanbul Kamu Hastaneleri Birliği")
                if "İstanbul Kamu Hastaneleri Birliği" in buyer_institutions
                else 0,
            )

        row_2 = st.columns([1.0, 1.0, 1.0, 1.0, 0.9], gap="small")
        with row_2[0]:
            estimated_unit_cost_try = st.number_input(
                "Tahmini birim maliyet",
                min_value=0.01,
                max_value=10_000.0,
                value=18.00,
                step=0.50,
                format="%.2f",
            )
        with row_2[1]:
            quantity = st.number_input(
                "Miktar",
                min_value=1,
                max_value=5_000_000,
                value=100_000,
                step=1_000,
                format="%d",
            )
        with row_2[2]:
            delivery_months = st.selectbox("Teslim süresi (ay)", [3, 6, 9, 12], index=1)
        with row_2[3]:
            competitor_count = st.number_input(
                "Tahmini rakip sayısı",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
            )
        with row_2[4]:
            analyze_clicked = st.button("Analiz Et", type="primary")

    if "analysis_ready" not in st.session_state:
        st.session_state.analysis_ready = False

    if analyze_clicked:
        st.session_state.analysis_ready = True

    if not st.session_state.analysis_ready:
        st.info("İhale bilgilerini girin ve geçmiş kazanımlara göre fiyat aralığını görmek için Analiz Et'e tıklayın.")
    else:
        query = {
            "product_name": product_name,
            "product_group": product_group,
            "region": region,
            "procedure_type": procedure_type,
            "buyer_institution": buyer_institution,
            "quantity": int(quantity),
            "delivery_months": int(delivery_months),
            "competitor_count_estimate": int(competitor_count),
        }

        models = train_price_models(df, PRICE_MODEL_VERSION, data_mtime_ns)
        profile_models = train_success_profile_models(df, SUCCESS_PROFILE_VERSION, data_mtime_ns)
        similar = retrieve_similar_tenders(df, query, top_k=SIMILAR_TENDER_COUNT)
        similar_display = similar.head(DISPLAY_TENDER_COUNT)
        price_corridor = percentile_metrics(similar[PRIMARY_PRICE_FIELD])
        nominal_reference = percentile_metrics(similar[REFERENCE_PRICE_FIELD])
        margin_benchmark = {
            "average": float(similar["gross_margin_pct"].mean()),
            "median": float(similar["gross_margin_pct"].median()),
            "p25": float(similar["gross_margin_pct"].quantile(0.25)),
            "p75": float(similar["gross_margin_pct"].quantile(0.75)),
        }
        discount_benchmark = {
            "average": float(similar["discount_to_estimated_cost_pct"].mean()),
            "median": float(similar["discount_to_estimated_cost_pct"].median()),
        }

        model_predictions = predict_model_prices(models, query)
        model_corridor = build_model_supported_corridor(price_corridor, model_predictions, models)
        scenario_prices = {
            "Düşük fiyat": model_corridor["low"],
            "Orta fiyat": model_corridor["middle"],
            "Yüksek fiyat": model_corridor["high"],
        }
        model_confidence = model_confidence_level(
            float(similar["overall_similarity_score"].mean()),
            price_corridor["median"],
            model_predictions["linear"],
            model_predictions["xgboost"],
        )
        model_center_values = np.array(
            [price_corridor["median"], model_predictions["linear"], model_predictions["xgboost"]],
            dtype=float,
        )
        model_disagreement_pct = (
            (model_center_values.max() - model_center_values.min())
            / max(float(model_center_values.mean()), 1.0)
            * 100
        )
        margins = scenario_margins(model_corridor, estimated_unit_cost_try)
        average_similarity = float(similar["overall_similarity_score"].mean())
        confidence = confidence_level(len(similar), average_similarity)
        balanced_margin = margins["Orta fiyat"]
        price_fit = historical_price_fit(
            scenario_prices["Orta fiyat"],
            price_corridor,
            balanced_margin,
        )
        profile_query_frame = build_profile_query_frame(
            query,
            scenario_prices["Orta fiyat"],
            balanced_margin,
        )
        success_profile_scores = score_success_profiles(profile_models, profile_query_frame)
        high_similarity_share = float((similar["overall_similarity_score"] >= 0.70).mean() * 100)
        strong_similar_count = int((similar["overall_similarity_score"] >= 0.70).sum())
        empirical_pwin = empirical_pwin_score(
            average_similarity,
            high_similarity_share,
            float(price_fit["score"]),
            float(success_profile_scores["one_class_score"]),
            float(success_profile_scores["cluster_score"]),
        )
        pwin_similarity_contribution = 0.30 * average_similarity * 100
        pwin_strong_case_contribution = 0.15 * high_similarity_share
        pwin_price_fit_contribution = 0.25 * float(price_fit["score"])
        pwin_one_class_contribution = 0.15 * float(success_profile_scores["one_class_score"])
        pwin_cluster_contribution = 0.15 * float(success_profile_scores["cluster_score"])
        pwin_contributions = {
            "average_similarity": pwin_similarity_contribution,
            "strong_similar_share": pwin_strong_case_contribution,
            "historical_price_fit": pwin_price_fit_contribution,
            "isolation_forest_profile_fit": pwin_one_class_contribution,
            "kmeans_success_profile_fit": pwin_cluster_contribution,
        }

        similarity_component = average_similarity * 100
        margin_component = margin_score(balanced_margin)
        strategic_fit_component = float(similar["strategic_fit_score"].mean())
        competition_component = competition_score(int(competitor_count))
        delivery_component = delivery_score(int(delivery_months))
        similarity_contribution = 0.25 * similarity_component
        margin_contribution = 0.30 * margin_component
        strategic_fit_contribution = 0.20 * strategic_fit_component
        competition_contribution = 0.15 * competition_component
        delivery_contribution = 0.10 * delivery_component

        final_attractiveness_score = (
            similarity_contribution
            + margin_contribution
            + strategic_fit_contribution
            + competition_contribution
            + delivery_contribution
        )
        final_label = attractiveness_label(final_attractiveness_score)
        llm_context = build_llm_context(
            query=query,
            estimated_unit_cost_try=float(estimated_unit_cost_try),
            similar=similar,
            price_corridor=price_corridor,
            nominal_reference=nominal_reference,
            margin_benchmark=margin_benchmark,
            discount_benchmark=discount_benchmark,
            model_corridor=model_corridor,
            scenario_prices=scenario_prices,
            margins=margins,
            model_predictions=model_predictions,
            models=models,
            model_confidence=model_confidence,
            model_disagreement_pct=model_disagreement_pct,
            price_fit=price_fit,
            success_profile_scores=success_profile_scores,
            empirical_pwin=empirical_pwin,
            pwin_contributions=pwin_contributions,
            final_score_components={
                "similarity_component_raw": similarity_component,
                "similarity_contribution": similarity_contribution,
                "margin_component_raw": margin_component,
                "margin_contribution": margin_contribution,
                "strategic_fit_component_raw": strategic_fit_component,
                "strategic_fit_contribution": strategic_fit_contribution,
                "competition_component_raw": competition_component,
                "competition_contribution": competition_contribution,
                "delivery_component_raw": delivery_component,
                "delivery_contribution": delivery_contribution,
            },
            final_attractiveness_score=final_attractiveness_score,
            final_label=final_label,
            strong_similar_count=strong_similar_count,
        )
        st.session_state.latest_llm_context = llm_context

        st.markdown("---")
        st.markdown(
            """
            <div class="section-kicker">Bulguların özeti</div>
            <div class="section-title">Yönetici Özeti</div>
            """,
            unsafe_allow_html=True,
        )

        top_metrics = st.columns(3, gap="medium")
        with top_metrics[0]:
            metric_card("Emsal p(win)", format_pct(empirical_pwin), "Emsal bazlı oran")
        with top_metrics[1]:
            metric_card("Model orta fiyat", format_try(scenario_prices["Orta fiyat"]), "Mayıs 2026 seviyesinde")
        with top_metrics[2]:
            metric_card("Orta fiyat kazancı", format_pct(balanced_margin), "Birim fiyat ve maliyete göre")

        support_metrics = st.columns(2, gap="medium")
        with support_metrics[0]:
            metric_card("Emsal havuzu", f"{len(similar)} ihale", f"Güçlü emsal: {strong_similar_count}")
        with support_metrics[1]:
            metric_card("Model güveni", model_confidence, f"Ortalama benzerlik: {average_similarity:.2f}")
        st.caption(
            "Bu bölüm bulguların kısa özetidir. Emsal havuzu, fiyat hesabında kullanılan en benzer 50 kazanılmış ihaleyi gösterir. "
            f"Güçlü emsal sayısı, bu 50 ihale içinde benzerlik skoru 0.70 ve üzeri olan kayıt adedidir; bu analizde {strong_similar_count}. "
            "Emsal p(win), geçmişte kazanılmış emsallere dayanarak üretilen oran formatında bir karar destek göstergesidir. "
            "Ortalama benzerlik, güçlü emsal sayısı, fiyatın tarihsel kazanım bandına uyumu, One-Class profil yakınlığı "
            "ve başarı profili eşleşmesi birlikte değerlendirilir."
        )
        st.caption(
            "Ortalama benzerlik şöyle bulunur: sistem önce veri tabanındaki her geçmiş ihaleye 0-1 arası benzerlik skoru verir, "
            "sonra en yüksek skorlu 50 ihalenin ortalamasını alır. Metin tarafında bu sürüm yerel embedding kullanır: "
            "ihale metinleri dense vektöre çevrilir ve cosine similarity ile karşılaştırılır. Ürün grubu, ürün adı, bölge, "
            "usul, miktar, teslim süresi ve rakip sayısı da ayrıca ağırlıklandırılır."
        )
        st.caption(
            f"Model güveni {model_confidence.lower()} çünkü benzer ihale medyanı, Linear tahmin ve "
            f"XGBoost tahmin arasındaki ayrışma yaklaşık %{model_disagreement_pct:.1f}. "
            "Ayrışma yükseldikçe sistem sonucu daha dikkatli yorumlamak gerekir."
        )

        section_1 = st.container()

        with section_1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="section-kicker">1. Emsal bulma</div>
                    <div class="section-title">En Benzer 10 Kazanılmış İhale</div>
                    """,
                    unsafe_allow_html=True,
                )
                table = similar_display[
                    [
                        "tender_id",
                        "year",
                        "product_name",
                        "product_group",
                        "region",
                        "procedure_type",
                        "winning_unit_price_try",
                        PRIMARY_PRICE_FIELD,
                        "gross_margin_pct",
                        "discount_to_estimated_cost_pct",
                        "overall_similarity_score",
                    ]
                ].copy()
                table.columns = [
                    "İhale ID",
                    "Yıl",
                    "Ürün",
                    "Grup",
                    "Bölge",
                    "Usul",
                    "O günkü fiyat",
                    "Mayıs 2026 fiyatı",
                    "Birim kazanç %",
                    "Yaklaşık maliyet altı %",
                    "Benzerlik",
                ]
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "O günkü fiyat": st.column_config.NumberColumn(format="%.2f TL"),
                        "Mayıs 2026 fiyatı": st.column_config.NumberColumn(format="%.2f TL"),
                        "Birim kazanç %": st.column_config.NumberColumn(format="%.2f"),
                        "Yaklaşık maliyet altı %": st.column_config.NumberColumn(format="%.2f"),
                        "Benzerlik": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                st.caption(
                    "Benzerlik skoru; metin benzerliği, ürün grubu, ürün adı, bölge, ihale usulü, "
                    "miktar, teslim süresi ve tahmini rakip sayısını birlikte ölçer. "
                    "Bu sürümde metin tarafı yerel embedding ile hesaplanır: ihale metinleri vektöre çevrilir ve "
                    "cosine similarity ile karşılaştırılır. Üretim versiyonunda bu katman transformer tabanlı embedding "
                    "modeliyle daha da güçlendirilebilir. Mevcut ağırlıklar: 0.45 metin embedding benzerliği, "
                    "0.12 ürün grubu, 0.10 ürün adı, 0.08 bölge, 0.05 ihale usulü, 0.10 miktar, "
                    "0.05 teslim süresi, 0.05 rakip sayısı."
                )

        section_3 = st.container()
        section_4 = st.container()

        with section_3:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="section-title">2. Fiyat Aralığı - Model Destekli Fiyat Aralığı</div>
                    """,
                    unsafe_allow_html=True,
                )
                corridor_rows = pd.DataFrame(
                    [
                        [
                            "Benzer ihale medyanı",
                            price_corridor["median"],
                            "En benzer 50 kazanılmış ihalenin Mayıs 2026 seviyesindeki orta fiyatı.",
                        ],
                        [
                            "Linear Regression tahmini",
                            model_predictions["linear"],
                            "Basit ve açıklanabilir fiyat modeli tahmini.",
                        ],
                        [
                            "XGBoost tahmini",
                            model_predictions["xgboost"],
                            "Ürün, bölge, miktar ve usul kombinasyonlarını yakalayan model tahmini.",
                        ],
                        [
                            "Model destekli düşük",
                            scenario_prices["Düşük fiyat"],
                            "Geçmiş fiyat alt bandı ve model düşük sınırları birlikte değerlendirilir.",
                        ],
                        [
                            "Model destekli orta",
                            scenario_prices["Orta fiyat"],
                            "Ana referans fiyat; birim kazanç ve toplam tutar hesaplarında kullanılır.",
                        ],
                        [
                            "Model destekli yüksek",
                            scenario_prices["Yüksek fiyat"],
                            "Geçmiş fiyat üst bandı ve model yüksek sınırları birlikte değerlendirilir.",
                        ],
                        [
                            "Top-k p90 üst eşiği",
                            price_corridor["p90"],
                            "Benzer kazanılmış ihalelerde fiyatların %90'ının altında kaldığı eşik.",
                        ],
                        [
                            "Top-k dağılım",
                            price_corridor["std"],
                            "Benzer ihalelerde fiyatların birbirinden ne kadar ayrıştığını gösterir.",
                        ],
                    ],
                    columns=["Gösterge", "Mayıs 2026 değeri", "Açıklama"],
                )
                st.dataframe(
                    corridor_rows,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Mayıs 2026 değeri": st.column_config.NumberColumn(format="%.2f TL"),
                    },
                )
                st.caption(
                    "Buradaki üç öneri tek bir modelden gelmez. Düşük fiyat; en benzer 50 ihalenin "
                    "p25 fiyatı, Linear modelin düşük sınırı ve XGBoost modelin düşük sınırı arasından "
                    "ortadaki değer seçilerek hesaplanır. Orta fiyat; top-k medyan, Linear tahmin ve "
                    "XGBoost tahmin arasından ortadaki değerdir. Yüksek fiyat; top-k p75, Linear yüksek "
                    "sınır ve XGBoost yüksek sınır arasından ortadaki değerdir."
                )
                st.markdown(
                    f'<span class="confidence">Model güveni: {model_confidence}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Bu analizde model güvenini belirleyen üç merkez değer: benzer ihale medyanı "
                    f"{format_try(price_corridor['median'])}, Linear tahmin "
                    f"{format_try(model_predictions['linear'])}, XGBoost tahmin "
                    f"{format_try(model_predictions['xgboost'])}. Bu değerler birbirinden uzaklaştığında "
                    "güven düşer; yakınlaştığında güven artar."
                )
                st.caption(
                    f"Tarihsel kazanım fiyat uyumu: {price_fit['label']} ({price_fit['level']}). "
                    f"Model destekli orta fiyat {format_try(scenario_prices['Orta fiyat'])}; "
                    f"top-k p25 {format_try(price_corridor['p25'])}, p75 {format_try(price_corridor['p75'])}, "
                    f"p90 {format_try(price_corridor['p90'])}. {price_fit['explanation']}"
                )
                st.markdown("##### Önerilen fiyat seçenekleri")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ["Düşük fiyat", scenario_prices["Düşük fiyat"]],
                            ["Orta fiyat", scenario_prices["Orta fiyat"]],
                            ["Yüksek fiyat", scenario_prices["Yüksek fiyat"]],
                        ],
                        columns=["Seçenek", "Önerilen birim fiyat"],
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Önerilen birim fiyat": st.column_config.NumberColumn(format="%.2f TL")
                    },
                )
                st.caption(
                    f"Bu analizde düşük fiyat {format_try(scenario_prices['Düşük fiyat'])}, "
                    f"orta fiyat {format_try(scenario_prices['Orta fiyat'])}, "
                    f"yüksek fiyat {format_try(scenario_prices['Yüksek fiyat'])} olarak oluştu."
                )

        with section_4:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="section-title">3. Maliyet ve Kazanç - Birim Fiyata Göre Kazanç Oranı</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            ["Düşük fiyat", scenario_prices["Düşük fiyat"], margins["Düşük fiyat"]],
                            ["Orta fiyat", scenario_prices["Orta fiyat"], margins["Orta fiyat"]],
                            ["Yüksek fiyat", scenario_prices["Yüksek fiyat"], margins["Yüksek fiyat"]],
                        ],
                    columns=["Seçenek", "Birim fiyat", "Birim kazanç %"],
                ),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Birim fiyat": st.column_config.NumberColumn(format="%.2f TL"),
                        "Birim kazanç %": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                st.caption(
                    f"Birim kazanç oranı, satış fiyatından tahmini birim maliyet çıkarıldıktan sonra "
                    f"fiyatın yüzde kaçının kazanç olarak kaldığını gösterir. Bu analizde tahmini birim "
                    f"maliyet {format_try(estimated_unit_cost_try)}."
                )
                st.markdown("##### Geçmiş ihalelerde birim kazanç")
                st.caption(
                    "Birim kazanç oranı, geçmiş ihalede kazanan fiyat ile iç maliyet arasındaki yüzde ilişkiyi gösterir."
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            ["Ortalama birim kazanç", margin_benchmark["average"]],
                            ["Orta birim kazanç", margin_benchmark["median"]],
                            ["Düşük seviye", margin_benchmark["p25"]],
                            ["Yüksek seviye", margin_benchmark["p75"]],
                        ],
                        columns=["Gösterge", "Değer"],
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={"Değer": st.column_config.NumberColumn(format="%.2f")},
                )

        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">3. Maliyet ve kazanç</div>
                <div class="section-title">Yaklaşık Maliyete Göre Geçmiş Teklif Seviyesi</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="explain-box">
                    Kamu ihalelerinde “yaklaşık maliyet”, alıcının ihale öncesi tahmini maliyet seviyesidir.
                    Bu bölüm, benzer kazanılmış
                    ihalelerde kazanan fiyatların bu yaklaşık maliyetin ortalama ne kadar altında kaldığını
                    gösterir. Amaç, geçmişte kazanılan tekliflerin ne kadar agresif veya normal fiyatlandığını okumaktır.
                </div>
                """,
                unsafe_allow_html=True,
            )
            cost_cols = st.columns(3, gap="medium")
            with cost_cols[0]:
                metric_card(
                    "Ortalama yaklaşık maliyet altı",
                    format_pct(discount_benchmark["average"]),
                    f"Benzer {len(similar)} kazanılmış ihale",
                )
            with cost_cols[1]:
                metric_card(
                    "Orta yaklaşık maliyet altı",
                    format_pct(discount_benchmark["median"]),
                    "Benzer ihalelerin orta seviyesi",
                )
            with cost_cols[2]:
                metric_card(
                    "Tahmini toplam ihale tutarı",
                    format_try(scenario_prices["Orta fiyat"] * quantity),
                    "Birim fiyat x miktar",
                )
            st.caption(
                f"Tahmini toplam ihale tutarı, bu ihalenin yaklaşık parasal büyüklüğünü gösterir. "
                f"Hesap: model destekli orta birim fiyat ({format_try(scenario_prices['Orta fiyat'])}) "
                f"x girilen miktar ({int(quantity):,}).".replace(",", ".")
            )

        profile = success_profile_scores["profile"]
        st.markdown(
            """
            <div class="section-kicker">4. Kazanım profili analizi</div>
            <div class="section-title">One-Class ve KMeans Profil Modelleri</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Bu bölüm fiyat tahmini yapmaz. Buradaki amaç, yeni ihalenin geçmişte kazanılmış iş tiplerine "
            "ne kadar benzediğini ve hangi başarı profiline yakın durduğunu göstermektir."
        )
        st.markdown(
            f"""
            <div class="profile-card-grid">
                <div class="profile-card-panel">
                    <div class="section-kicker">One-Class modeli</div>
                    <div class="section-title">Kazanım Profili Yakınlığı</div>
                    <div class="profile-summary">
                        <div class="profile-status">{escape(success_profile_scores["one_class_label"])}</div>
                        <div class="profile-score">{success_profile_scores["one_class_score"]:.0f}/100</div>
                        <div class="profile-note">Geçmiş kazanım profiline benzerlik</div>
                    </div>
                    <span class="method-badge">Yöntem: One-Class / Isolation Forest</span>
                    <p>
                        Bu skor, yeni ihalenin geçmişte kazanılmış işlerin genel şekline ne kadar benzediğini gösterir.
                        Skor yükseldikçe ihale daha tanıdık; skor düştükçe daha sıra dışı görünür.
                    </p>
                </div>
                <div class="profile-card-panel">
                    <div class="section-kicker">Kümeleme modeli</div>
                    <div class="section-title">Başarı Profili</div>
                    <div class="profile-summary">
                        <div class="profile-status">En yakın başarı grubu</div>
                        <div class="profile-score">{escape(success_profile_scores["cluster_label"])}</div>
                        <div class="profile-note">{escape(profile["name"])}</div>
                    </div>
                    <span class="method-badge green">Yöntem: KMeans başarı grupları</span>
                    <p>
                        KMeans geçmiş kazanılmış ihaleleri 4 başarı grubuna ayırır. Bu ihale en çok
                        {escape(profile["name"])} grubuna benziyor. Grup {profile["count"]} ihale içerir;
                        tipik fiyat {escape(format_try(profile["median_price"]))}, tipik birim kazanç
                        {escape(format_pct(profile["median_margin"]))}. Eşleşme gücü
                        {success_profile_scores["cluster_score"]:.0f}/100 olduğu için bu sonuç
                        {escape(success_profile_scores["cluster_label"].lower())} olarak yorumlanır.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">5. Emsal kazanım değerlendirmesi</div>
                <div class="section-title">Emsal p(win) - Karar Destek Oranı</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="explain-box">
                    <b>Emsal p(win): {format_pct(empirical_pwin)}</b><br>
                    Bu yüzde, “bu ihale geçmişte kazandığımız işlere ve fiyat davranışımıza ne kadar yakın?”
                    sorusunun özetidir. 100'e yaklaştıkça yeni ihale, geçmiş kazanılmış emsallere daha çok benzer
                    ve önerilen fiyat geçmiş kazanım bandına daha iyi oturur. Veri setinde kaybedilen ihaleler
                    olmadığı için bu değer gerçek kazan/kaybet olasılığı değil, emsal bazlı kazanım göstergesidir.
                </div>
                """,
                unsafe_allow_html=True,
            )
            pwin_rows = pd.DataFrame(
                [
                    [
                        "Ortalama benzerlik",
                        "%30",
                        f"{average_similarity:.2f}",
                        f"{pwin_similarity_contribution:.1f} puan",
                        "Yeni ihale, en benzer 50 kazanılmış ihaleye ne kadar benziyor?",
                    ],
                    [
                        "Güçlü emsal oranı",
                        "%15",
                        f"{strong_similar_count}/{len(similar)}",
                        f"{pwin_strong_case_contribution:.1f} puan",
                        "Top-50 içinde benzerlik skoru 0.70 üstü olan güçlü emsal sayısı.",
                    ],
                    [
                        "Tarihsel fiyat bandı uyumu",
                        "%25",
                        price_fit["label"],
                        f"{pwin_price_fit_contribution:.1f} puan",
                        "Önerilen orta fiyat, geçmişte kazanılmış benzer fiyatların içinde mi?",
                    ],
                    [
                        "Kazanım profili yakınlığı",
                        "%15",
                        f"{success_profile_scores['one_class_label']} ({success_profile_scores['one_class_score']:.0f}/100)",
                        f"{pwin_one_class_contribution:.1f} puan",
                        "Yeni ihale, geçmişte kazandığımız işlerin genel şekline benziyor mu?",
                    ],
                    [
                        "Başarı grubu eşleşmesi",
                        "%15",
                        success_profile_scores["cluster_label"],
                        f"{pwin_cluster_contribution:.1f} puan",
                        "Yeni ihale, geçmiş kazanılmış iş gruplarından hangisine benziyor?",
                    ],
                ],
                columns=[
                    "Bileşen",
                    "p(win) içindeki ağırlık",
                    "Bu analizde sonuç",
                    "100 üzerinden katkı puanı",
                    "Ne anlatır?",
                ],
            )
            st.dataframe(
                pwin_rows,
                hide_index=True,
                width="stretch",
            )

            detail_cols = st.columns(3, gap="medium")
            with detail_cols[0]:
                mini_summary(
                    "One-Class yorumu",
                    success_profile_scores["one_class_label"],
                    f"{success_profile_scores['one_class_score']:.0f}/100 profil yakınlığı",
                )
                st.caption(
                    "One-Class / Isolation Forest modeli sadece kazanılmış ihalelerden öğrenir. Skor yüksekse yeni ihale "
                    "geçmiş kazanım profilimize benzer; skor düşükse bu ihale alışılmış kazanım profilimizin dışında kalır."
                )
            with detail_cols[1]:
                mini_summary(
                    "Başarı profili",
                    success_profile_scores["cluster_label"],
                    profile["name"],
                )
                st.caption(
                    f"Geçmiş kazanılmış ihaleler 4 başarı grubuna ayrılır. Seçilen grup {profile['count']} "
                    f"ihale içerir; tipik fiyat {format_try(profile['median_price'])}, tipik birim kazanç oranı "
                    f"{format_pct(profile['median_margin'])}."
                )
            with detail_cols[2]:
                mini_summary(
                    "Fiyat uyumu",
                    price_fit["level"],
                    price_fit["label"],
                )
                st.caption(
                    f"Model destekli orta fiyat {format_try(scenario_prices['Orta fiyat'])}. "
                    f"Benzer kazanılmış ihalelerde p25 {format_try(price_corridor['p25'])}, "
                    f"p75 {format_try(price_corridor['p75'])}, p90 {format_try(price_corridor['p90'])}."
                )

        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">6. Fırsat önceliği</div>
                <div class="section-title">İhale Öncelik Puanı</div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                build_gauge(final_attractiveness_score),
                config={"displayModeBar": False, "staticPlot": True},
            )
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-value">{final_attractiveness_score:.0f}</div>
                    <div class="score-label">{final_label}</div>
                    <p class="metric-note">
                        Bu puan, ihalenin teklif sürecinde ne kadar öncelikli ele alınabileceğini gösterir.
                        Fiyat tahmini veya kazanma olasılığı değildir.
                    </p>
                    <span class="confidence">Güven: {confidence}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Bu puan, satış/ihale ekibinin fırsatı önceliklendirmesi için kullanılır. "
                f"Katkılar: geçmiş emsale benzerlik {similarity_contribution:.1f} puan, "
                f"birim kazanç sağlığı {margin_contribution:.1f} puan, stratejik uyum "
                f"{strategic_fit_contribution:.1f} puan, rekabet koşulu "
                f"{competition_contribution:.1f} puan, teslim süresi "
                f"{delivery_contribution:.1f} puan. Toplam {final_attractiveness_score:.0f}/100."
            )

        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">7. Model kontrolü</div>
                <div class="section-title">5-Fold Backtest Metrikleri</div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        [
                            "Linear Regression",
                            models["linear_residuals"]["mae"],
                            models["linear_residuals"]["mape"],
                            models["linear_residuals"]["coverage"],
                        ],
                        [
                            "XGBoost",
                            models["xgboost_residuals"]["mae"],
                            models["xgboost_residuals"]["mape"],
                            models["xgboost_residuals"]["coverage"],
                        ],
                    ],
                    columns=["Model", "MAE", "MAPE %", "Koridor coverage %"],
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "MAE": st.column_config.NumberColumn(format="%.2f TL"),
                    "MAPE %": st.column_config.NumberColumn(format="%.2f"),
                    "Koridor coverage %": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(
                "MAE ortalama TL hatasını, MAPE ortalama yüzde hatayı, coverage ise geçmiş testlerde "
                "gerçek fiyatın model hata payı koridoruna düşme oranını gösterir."
            )

        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">8. Kısa açıklama</div>
                <div class="section-title">Bu Sonuç Ne Anlama Geliyor?</div>
                """,
                unsafe_allow_html=True,
            )
            model_note = (
                "Model tahminleri benzer ihale medyanıyla uyumlu görünüyor."
                if model_confidence == "Yüksek"
                else "Model tahminleri ile benzer ihale medyanı ayrışıyor; manuel inceleme önerilir."
                if model_confidence == "Düşük"
                else "Model tahminleri ve benzer ihale koridoru orta seviyede uyumlu."
            )
            corridor_calculation_note = (
                f"Düşük fiyat hesabında üç değer karşılaştırıldı: top-k p25 "
                f"({format_try(price_corridor['p25'])}), Linear düşük sınır "
                f"({format_try(model_corridor['linear_low'])}) ve XGBoost düşük sınır "
                f"({format_try(model_corridor['xgboost_low'])}). Bu üç değerin ortadaki değeri "
                f"{format_try(scenario_prices['Düşük fiyat'])} olduğu için nihai düşük fiyat bu oldu. "
                f"Orta fiyat hesabında top-k medyan ({format_try(price_corridor['median'])}), "
                f"Linear tahmin ({format_try(model_predictions['linear'])}) ve XGBoost tahmin "
                f"({format_try(model_predictions['xgboost'])}) karşılaştırıldı. Ortadaki değer "
                f"{format_try(scenario_prices['Orta fiyat'])} olduğu için nihai orta fiyat bu oldu. "
                f"Yüksek fiyat hesabında top-k p75 ({format_try(price_corridor['p75'])}), "
                f"Linear yüksek sınır ({format_try(model_corridor['linear_high'])}) ve XGBoost yüksek sınır "
                f"({format_try(model_corridor['xgboost_high'])}) karşılaştırıldı. Ortadaki değer "
                f"{format_try(scenario_prices['Yüksek fiyat'])} olduğu için nihai yüksek fiyat bu oldu."
            )
            explanation = (
                f"Emsal p(win) {format_pct(empirical_pwin)}. Bu oran; ortalama benzerlik "
                f"{average_similarity:.2f}, güçlü emsal sayısı {strong_similar_count}/{len(similar)}, "
                f"fiyat uyumu {price_fit['label'].lower()}, kazanım profili yakınlığı "
                f"{success_profile_scores['one_class_score']:.0f}/100 ve başarı profili yakınlığı "
                f"{success_profile_scores['cluster_score']:.0f}/100 sinyallerinden oluşur. "
                f"{len(similar)} benzer kazanılmış ihale bulundu. Güven seviyesi {confidence.lower()}. "
                f"Benzer ihale medyanı, en benzer 50 geçmiş ihalenin orta fiyatıdır ve bu analizde "
                f"{format_try(price_corridor['median'])}. Linear Regression modeli aynı ihale için "
                f"{format_try(model_predictions['linear'])} tahmin etti; bu model daha basit ve açıklanabilir "
                f"bir fiyat referansı verir. XGBoost modeli {format_try(model_predictions['xgboost'])} tahmin etti; "
                f"bu model ürün, bölge, miktar ve usul gibi değişkenlerin kombinasyon etkilerini yakalamaya çalışır. "
                f"Cluster analizi bu ihaleyi '{success_profile_scores['profile']['name']}' tipindeki geçmiş başarı "
                f"profiline en yakın buldu. One-Class model ise ihalenin genel kazanım profiline "
                f"{success_profile_scores['one_class_label'].lower()} olduğunu gösterdi. "
                f"Bu üç kaynak birlikte değerlendirildiğinde model destekli orta fiyat "
                f"{format_try(scenario_prices['Orta fiyat'])} oldu. Girilen "
                f"{format_try(estimated_unit_cost_try)} birim maliyete göre orta fiyat birim kazanç oranı "
                f"{format_pct(balanced_margin)}. İhale puanı {final_attractiveness_score:.0f}/100 ve sonuç "
                f"{final_label}. {model_note} {corridor_calculation_note}"
            )
            st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)
with llm_tab:
    latest_llm_context = st.session_state.get("latest_llm_context")
    if not latest_llm_context:
        st.info("LLM yorumu için önce Analiz tabında ihale bilgilerini girip Analiz Et'e tıklayın.")
    else:
        render_llm_chatbot(latest_llm_context)
