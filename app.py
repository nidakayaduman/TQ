"""Tender decision helper - Streamlit MVP."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "x_ilac_synthetic_tenders_2021_2025.csv"
PRIMARY_PRICE_FIELD = "inflation_adjusted_unit_price_2026_try"
REFERENCE_PRICE_FIELD = "winning_unit_price_try"
SIMILAR_TENDER_COUNT = 50
DISPLAY_TENDER_COUNT = 10
PRICE_MODEL_VERSION = "2026-price-model-without-year-v1"
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
    "tfidf": 0.45,
    "product_group": 0.12,
    "product_name": 0.10,
    "region": 0.08,
    "procedure_type": 0.05,
    "quantity": 0.10,
    "delivery_months": 0.05,
    "competitor_count": 0.05,
}


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

        .metric-card {
            min-height: 128px;
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
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
            min-height: 300px;
        }

        .score-value {
            color: var(--text);
            font-size: 4.2rem;
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

        @media (max-width: 900px) {
            .scope-pill { float: none; margin-top: 1rem; }
            .score-value { font-size: 3rem; }
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
    tfidf_score: float,
    product_group: float,
    product_name: float,
    region: float,
    procedure_type: float,
    quantity: float,
    delivery_months: float,
    competitor_count: float,
) -> float:
    return (
        SIMILARITY_WEIGHTS["tfidf"] * tfidf_score
        + SIMILARITY_WEIGHTS["product_group"] * product_group
        + SIMILARITY_WEIGHTS["product_name"] * product_name
        + SIMILARITY_WEIGHTS["region"] * region
        + SIMILARITY_WEIGHTS["procedure_type"] * procedure_type
        + SIMILARITY_WEIGHTS["quantity"] * quantity
        + SIMILARITY_WEIGHTS["delivery_months"] * delivery_months
        + SIMILARITY_WEIGHTS["competitor_count"] * competitor_count
    )


@st.cache_data
def load_dataset() -> pd.DataFrame:
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


@st.cache_resource
def build_tfidf_index(search_texts: tuple[str, ...]) -> tuple[TfidfVectorizer, Any]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(search_texts)
    return vectorizer, matrix


def get_similarity_index(df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    search_texts = tuple(df.apply(lambda row: combine_text(row, HISTORICAL_TEXT_FIELDS), axis=1))
    return build_tfidf_index(search_texts)


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
def train_price_models(df: pd.DataFrame, model_version: str) -> dict[str, Any]:
    _ = model_version
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
    vectorizer, matrix = get_similarity_index(df)
    query_text = combine_text(query, QUERY_TEXT_FIELDS)
    query_vector = vectorizer.transform([query_text])
    tfidf_scores = cosine_similarity(query_vector, matrix)[0]

    candidates = []
    for initial_rank, idx in enumerate(tfidf_scores.argsort()[::-1], start=1):
        row = df.iloc[int(idx)]
        tfidf_score = float(tfidf_scores[idx])
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
            tfidf_score,
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
                "initial_tfidf_rank": initial_rank,
                "tfidf_score": tfidf_score,
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
                "tfidf_score",
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
        return "Marj korunarak ilerlenebilir"
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
        height=230,
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
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        geçmiş kazanım hafızasına dayalı fiyat bandı, marj ve karar destek
        puanı üretmektir.

        **1. Veri kullanımı**

        Her geçmiş ihale satırı ürün, ürün grubu, alıcı kurum, bölge, ihale
        usulü, miktar, teslim süresi, tahmini rakip sayısı, kazanılmış fiyat,
        brüt marj ve stratejik uyum bilgilerini taşır. Ana fiyat alanı
        `inflation_adjusted_unit_price_2026_try` kolonudur; eski yıllardaki
        fiyatlar Mayıs 2026 TL seviyesine taşınmış haliyle karşılaştırılır.

        **2. Yeni ihale girdisi**

        Kullanıcı ürün adı, ürün grubu, bölge, ihale usulü, alıcı kurum,
        tahmini birim maliyet, miktar, teslim süresi ve tahmini rakip sayısını
        girer. Bu bilgiler yeni ihale sorgusu olarak kullanılır.

        **3. Benzerlik skoru**

        Sistem her geçmiş ihale için `overall_similarity_score` hesaplar. Metin
        alanları TF-IDF ile sayısallaştırılır ve cosine similarity ile metin
        yakınlığı ölçülür. Ürün grubu, bölge ve ihale usulü birebir eşleşme
        skoru alır. Ürün adı token ortaklığına göre daha esnek değerlendirilir.
        Miktar, teslim süresi ve rakip sayısı için yakınlık skoru şu mantıkla
        hesaplanır:

        ```text
        yakınlık = 1 - mutlak_fark / büyük_değer
        ```

        Nihai benzerlik formülü:

        ```text
        overall_similarity_score =
        0.45 * tfidf_score (ihale metni, ürün, kurum, bölge ve usul metinlerinin genel benzerliği)
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
        gösterilir.

        **4. Top-k fiyat benchmark'ı**

        En benzer 50 ihalenin Mayıs 2026 fiyatlarından `p25`, `median`, `p75`,
        ortalama ve dağılım hesaplanır. Bu bölüm geçmiş kazanılmış ihalelerden
        gelen açıklanabilir fiyat hafızasıdır.

        **5. Linear Regression ve XGBoost ne için kullanılır?**

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

        **6. Backtest ve hata payı**

        Sistem 5-fold cross validation ile modelleri geçmiş veri üzerinde test
        eder. Her kayıt için:

        ```text
        residual = gerçek_fiyat - tahmin_fiyatı
        ```

        Residual dağılımından `p25`, `median`, `p75`, MAE, MAPE ve coverage
        hesaplanır. Bu hata payları yeni ihaledeki model tahminlerine koridor
        oluşturmak için eklenir.

        **7. Nihai fiyat koridoru**

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

        **8. Marj hesabı**

        Girilen tahmini birim maliyet ile model destekli fiyatlar
        karşılaştırılır:

        ```text
        marj = (fiyat - tahmini_birim_maliyet) / fiyat * 100
        ```

        **9. İhale puanı**

        Genel puan 0-100 arasıdır ve şu bileşenlerden oluşur:

        ```text
        ihale_puanı =
        0.25 * benzerlik_puanı (seçilen top-k geçmiş ihaleler yeni ihaleye ne kadar benziyor)
        + 0.30 * marj_puanı (model destekli orta fiyatla beklenen marj ne kadar sağlıklı)
        + 0.20 * stratejik_uyum (benzer geçmiş ihalelerdeki stratejik fit ortalaması)
        + 0.15 * rekabet_puanı (tahmini rakip sayısı azaldıkça puan artar)
        + 0.10 * teslim_süresi_puanı (teslim süresi iş kuralına göre puanlanır)
        ```

        Sonuç etiketi puana göre verilir: `Güçlü fırsat`, `Orta fiyatla
        ilerlenebilir`, `Marj korunarak ilerlenebilir` veya `Manuel inceleme
        gerekir`.
        """
    )


df = load_dataset()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">TI</div>
        <div class="sidebar-title">TENDER IQ</div>
        <div class="sidebar-note">
            Geçmişte kazanılmış benzer ihalelere bakarak fiyat aralığı,
            marj ve ihale puanı gösterir. Kazanma ihtimali hesaplamaz.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Veri")
    st.write(f"{len(df):,} geçmiş kazanılmış ihale kaydı".replace(",", "."))
    st.write(f"{df['year'].min()}-{df['year'].max()} dönemini kapsar")
    st.write("Ürün, kurum, bölge, miktar, fiyat, marj ve teslim bilgileri içerir")
    st.write("Gerçek şirket verisi değil; X İlaç Şirketi için hazırlanmış sentetik demo verisidir")

header_left, header_right = st.columns([5, 1.4])
with header_left:
    st.markdown(
        """
        <div class="eyebrow">İhale Karar Yardımcısı</div>
        <h1 class="page-title">Geçmiş İhaleye Göre Fiyat Aralığı</h1>
        <p class="page-subtitle">
            Yeni bir ihale için benzer geçmiş kazanımları bulur, fiyatları
            Mayıs 2026 seviyesine taşır, olası fiyat aralığını ve marjı gösterir.
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
        kazanılmış ihalelerden fiyat ve marj kıyaslaması yapar.
    </div>
    """,
    unsafe_allow_html=True,
)

analysis_tab, how_it_works_tab = st.tabs(["Analiz", "Nasıl çalışır?"])

with how_it_works_tab:
    render_how_it_works_tab()

with analysis_tab:
    product_names = sorted(df["product_name"].dropna().unique())
    product_groups = sorted(df["product_group"].dropna().unique())
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
            product_group = st.selectbox(
                "Ürün grubu",
                product_groups,
                index=product_groups.index("IV Solution") if "IV Solution" in product_groups else 0,
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
        st.stop()

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

    models = train_price_models(df, PRICE_MODEL_VERSION)
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

    similarity_component = average_similarity * 100
    balanced_margin = margins["Orta fiyat"]
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

    st.markdown("---")

    top_metrics = st.columns(4, gap="medium")
    with top_metrics[0]:
        metric_card("Benzer ihale", str(len(similar)), f"Güven: {confidence}")
    with top_metrics[1]:
        metric_card("Model orta fiyat", format_try(scenario_prices["Orta fiyat"]), "Mayıs 2026 seviyesinde")
    with top_metrics[2]:
        metric_card("Orta marj", format_pct(balanced_margin), "Girilen maliyete göre")
    with top_metrics[3]:
        metric_card("Model güveni", model_confidence, f"Benzerlik: {average_similarity:.2f}")
    st.caption(
        "Özet kartları; en benzer 50 ihale, Linear Regression, XGBoost ve girilen maliyet "
        "birlikte değerlendirilerek hesaplanır. Bu skor kazanma ihtimali değil, fiyat/marj karar desteğidir."
    )
    st.caption(
        f"Model güveni {model_confidence.lower()} çünkü benzer ihale medyanı, Linear tahmin ve "
        f"XGBoost tahmin arasındaki ayrışma yaklaşık %{model_disagreement_pct:.1f}. "
        "Ayrışma yükseldikçe sistem sonucu daha dikkatli yorumlamak gerekir."
    )

    section_1, section_2 = st.columns([1.55, 1.0], gap="medium")

    with section_1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">1. Geçmiş örnekler</div>
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
                "Brüt marj %",
                "İndirim %",
                "Benzerlik",
            ]
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "O günkü fiyat": st.column_config.NumberColumn(format="%.2f TL"),
                    "Mayıs 2026 fiyatı": st.column_config.NumberColumn(format="%.2f TL"),
                    "Brüt marj %": st.column_config.NumberColumn(format="%.2f"),
                    "İndirim %": st.column_config.NumberColumn(format="%.2f"),
                    "Benzerlik": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(
                "Benzerlik skoru; metin benzerliği, ürün grubu, ürün adı, bölge, ihale usulü, "
                "miktar, teslim süresi ve tahmini rakip sayısını birlikte ölçer. "
                "Ağırlıklar: 0.45 TF-IDF metin benzerliği, 0.12 ürün grubu, 0.10 ürün adı, "
                "0.08 bölge, 0.05 ihale usulü, 0.10 miktar, 0.05 teslim süresi, 0.05 rakip sayısı."
            )

    with section_2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">4. Genel puan</div>
                <div class="section-title">İhale Puanı</div>
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
                        Puan; benzerlik, marj, geçmiş uyum, rakip sayısı ve teslim
                        süresi birlikte değerlendirilerek hesaplandı.
                    </p>
                    <span class="confidence">Güven: {confidence}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"İhale puanına katkılar: benzerlik {similarity_contribution:.1f} puan, "
                f"marj {margin_contribution:.1f} puan, stratejik uyum "
                f"{strategic_fit_contribution:.1f} puan, rekabet "
                f"{competition_contribution:.1f} puan, teslim süresi "
                f"{delivery_contribution:.1f} puan. Bu katkıların toplamı "
                f"{final_attractiveness_score:.0f}/100 olarak gösterilir."
            )

    section_3, section_4, section_5 = st.columns(3, gap="medium")

    with section_3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">2. Fiyat aralığı</div>
                <div class="section-title">Model Destekli Fiyat Aralığı</div>
                """,
                unsafe_allow_html=True,
            )
            corridor_rows = pd.DataFrame(
                [
                    ["Benzer ihale medyanı", price_corridor["median"], nominal_reference["median"]],
                    ["Linear Regression tahmini", model_predictions["linear"], None],
                    ["XGBoost tahmini", model_predictions["xgboost"], None],
                    ["Model destekli düşük", scenario_prices["Düşük fiyat"], None],
                    ["Model destekli orta", scenario_prices["Orta fiyat"], None],
                    ["Model destekli yüksek", scenario_prices["Yüksek fiyat"], None],
                    ["Top-k dağılım", price_corridor["std"], nominal_reference["std"]],
                ],
                columns=["Gösterge", "Mayıs 2026 fiyatı", "O günkü fiyat"],
            )
            st.dataframe(
                corridor_rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "Mayıs 2026 fiyatı": st.column_config.NumberColumn(format="%.2f TL"),
                    "O günkü fiyat": st.column_config.NumberColumn(format="%.2f TL"),
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
                <div class="section-kicker">3. Marj</div>
                <div class="section-title">Maliyete Göre Marj</div>
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
                    columns=["Seçenek", "Birim fiyat", "Marj %"],
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Birim fiyat": st.column_config.NumberColumn(format="%.2f TL"),
                    "Marj %": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption(
                f"Marj formülü: (fiyat - tahmini birim maliyet) / fiyat * 100. "
                f"Bu analizde tahmini birim maliyet {format_try(estimated_unit_cost_try)}."
            )
            st.markdown("##### Geçmiş ihalelerde marj")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Ortalama marj", margin_benchmark["average"]],
                        ["Orta marj", margin_benchmark["median"]],
                        ["Düşük seviye", margin_benchmark["p25"]],
                        ["Yüksek seviye", margin_benchmark["p75"]],
                    ],
                    columns=["Gösterge", "Değer"],
                ),
                hide_index=True,
                width="stretch",
                column_config={"Değer": st.column_config.NumberColumn(format="%.2f")},
            )

    with section_5:
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-kicker">Ek bilgi</div>
                <div class="section-title">Geçmiş İndirim ve Tutar</div>
                """,
                unsafe_allow_html=True,
            )
            metric_card(
                "Ortalama indirim",
                format_pct(discount_benchmark["average"]),
                f"Benzer {len(similar)} kazanılmış ihale",
            )
            st.write("")
            metric_card(
                "Orta indirim",
                format_pct(discount_benchmark["median"]),
                "Yaklaşık maliyete göre",
            )
            st.write("")
            metric_card(
                "Tahmini toplam tutar",
                format_try(scenario_prices["Orta fiyat"] * quantity),
                "Orta fiyat x miktar",
            )
            st.caption(
                f"Tahmini toplam tutar = model destekli orta fiyat "
                f"({format_try(scenario_prices['Orta fiyat'])}) x miktar ({int(quantity):,}).".replace(",", ".")
            )

    with st.container(border=True):
        st.markdown(
            """
            <div class="section-kicker">Model kontrolü</div>
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
            <div class="section-kicker">6. Kısa açıklama</div>
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
            f"{len(similar)} benzer kazanılmış ihale bulundu. Güven seviyesi {confidence.lower()}. "
            f"Benzer ihale medyanı, en benzer 50 geçmiş ihalenin orta fiyatıdır ve bu analizde "
            f"{format_try(price_corridor['median'])}. Linear Regression modeli aynı ihale için "
            f"{format_try(model_predictions['linear'])} tahmin etti; bu model daha basit ve açıklanabilir "
            f"bir fiyat referansı verir. XGBoost modeli {format_try(model_predictions['xgboost'])} tahmin etti; "
            f"bu model ürün, bölge, miktar ve usul gibi değişkenlerin kombinasyon etkilerini yakalamaya çalışır. "
            f"Bu üç kaynak birlikte değerlendirildiğinde model destekli orta fiyat "
            f"{format_try(scenario_prices['Orta fiyat'])} oldu. Girilen "
            f"{format_try(estimated_unit_cost_try)} birim maliyete göre orta fiyat marjı "
            f"{format_pct(balanced_margin)}. İhale puanı {final_attractiveness_score:.0f}/100 ve sonuç "
            f"{final_label}. {model_note} {corridor_calculation_note}"
        )
        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)
